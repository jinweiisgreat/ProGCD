import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.optim import SGD, lr_scheduler
from torch.nn import functional as F
from data.get_datasets import get_datasets
from data.augmentations import get_transform
from project_utils.loss_utils import ContrastiveLearningViewGenerator, StrongWeakView
from project_utils.cluster_utils import mixed_eval, AverageMeter
from data.get_datasets import get_datasets, get_class_splits
from models.cocoop_ovdet import CustomCLIP
from models.customloss import DistillLoss, AlignLoss
import clip
from tqdm import tqdm
from utils.gpu_utils import check_gpu
from loguru import logger
import time
import os
from pretrained_utils import *
from get_args import get_arguments

@logger.catch
def main():
    args = get_arguments()
    args.dataset_name = "cifar100"
    args.epochs = 100  
    args = get_class_splits(args)
    args.interpolation = 3
    args.crop_pct = 0.875
    args.image_size = 224
    args.feat_dim = 768
    args.num_mlp_layers = 3
    args.mlp_out_dim = 128
    n_ctx = 40 
    args.n_ctx = n_ctx

    train_transform, test_transform = get_transform(args.transform, image_size=args.image_size, args=args)
    train_transform = ContrastiveLearningViewGenerator(base_transform=train_transform, n_views=args.n_views)

    train_dataset, _, _, _ = get_datasets(args.dataset_name, train_transform, test_transform, args)
    
    label_len = len(train_dataset.labelled_dataset)
    unlabelled_len = len(train_dataset.unlabelled_dataset)
    sample_weights = [1 if i < label_len else label_len / unlabelled_len for i in range(len(train_dataset))]
    sample_weights = torch.DoubleTensor(sample_weights)
    sampler = torch.utils.data.WeightedRandomSampler(sample_weights, num_samples=len(train_dataset))

    train_loader = DataLoader(train_dataset, num_workers=0, batch_size=args.batch_size,sampler=sampler, shuffle=False,drop_last=False)

    pretrained_model_path = None
    cfg = setup_cfg(args)

    if pretrained_model_path != None:
        clip_model = torch.load(pretrained_model_path)
    else:
        model, preprocess = clip.load('/home/Lmodel/ViT-L-14.pt')
        model.cuda().eval()
        clip_model = CustomCLIP(cfg,args,model)

    optimizer = SGD(clip_model.text_encoder.ln_learn.parameters(), lr=args.lr, momentum=args.momentum)

    exp_lr_scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.epochs,
            eta_min=args.lr * 1e-3,
        )
    
    align_crit = AlignLoss()
    distill_crit = DistillLoss()
    count = 0

    currentTime = time.localtime()
    start_time = time.strftime("%Y_%m_%d_%H_%M_%S", currentTime)
    

    pth_save_path = f'./result/pretrained_model_save'
    log_save_path = f'./result/log'
    if not os.path.exists(pth_save_path):
        os.mkdir(pth_save_path+'/')
    if not os.path.exists(log_save_path):
        os.mkdir(log_save_path+'/')
    pth_save_path = pth_save_path + f'/{start_time}'
    log_save_path = log_save_path + f'/{start_time}'
    if not os.path.exists(pth_save_path):
        os.mkdir(pth_save_path+'/')
    if not os.path.exists(log_save_path):
        os.mkdir(log_save_path+'/')
    metriclog = open(log_save_path+'/train_log_with_clip.log', 'w')
    metriclog.write('lr: '+str(args.lr) + "\n")
    metriclog.write('batch_size: '+str(args.batch_size) + "\n")
    

    args.txt_file_name = f"./pth/{args.dataset_name}_a_photo_of_label.pth"
    
    metriclog.write('txt_file_name: '+args.txt_file_name + "\n")

    info = 'Batch_size: {} lr: {:.8f} txt_file: {} n_ctx:{}'.format(args.batch_size,exp_lr_scheduler.get_lr()[0],args.txt_file_name,args.n_ctx)

    print(info)
    for epoch in range(args.epochs):
        loss_record = AverageMeter()
        for batch_idx, batch in enumerate(train_loader):
            count += 1

            images, class_labels, uq_idxs, mask_lab = batch

            mask_lab = mask_lab[:, 0]
            mask_lab = mask_lab.bool()
            class_labels = class_labels.cuda()

            all_img_feats_1, all_txt_feats_1 = clip_model(images[0].cuda())
            all_img_feats_2, all_txt_feats_2 = clip_model(images[1].cuda())
            
            img_txt_loss_1,txt_img_loss_1 = align_crit(all_img_feats_1.cuda(),all_txt_feats_2.cuda(),args)
            img_txt_loss_2,txt_img_loss_2 = align_crit(all_img_feats_2.cuda(),all_txt_feats_1.cuda(),args)
            
            loss_align = img_txt_loss_1 + txt_img_loss_1 + img_txt_loss_2 + txt_img_loss_2
            total_loss = loss_align

            sup_labels = class_labels[mask_lab]
            
            sup_labels = sup_labels.contiguous().view(sup_labels.shape[0], -1)

            l_txt_feats_1 = all_txt_feats_1[mask_lab]
            l_txt_feats_2 = all_txt_feats_2[mask_lab]
           
            '''
            distill_loss
            '''
            txt_feats_gt = get_clip_text_feat_by_index(args.txt_file_name,uq_idxs[mask_lab]).cpu() 
            txt_feats_gt = txt_feats_gt.squeeze(1)

            #two views
            distill_loss_1 = distill_crit(l_txt_feats_1,txt_feats_gt,labels=sup_labels)
            distill_loss_2 = distill_crit(l_txt_feats_2,txt_feats_gt,labels=sup_labels)
            
            loss_distill = distill_loss_1 + distill_loss_2
            total_loss = loss_distill + loss_align
            pstr = ''
            pstr += f'loss_align: {loss_align.item():.4f} '
            pstr += f'loss_distill: {loss_distill.item():.4f} '
           
            loss_record.update(total_loss.item(), args.batch_size)
            optimizer.zero_grad()
            total_loss.backward()
            
            optimizer.step()
            exp_lr_scheduler.step()
            if batch_idx % args.print_freq == 0 or (batch_idx + 1) % len(train_loader) == 0:
                info = 'Epoch: [{}][{}/{}]\t loss {:.5f}\t {}'.format(epoch, batch_idx, len(train_loader), total_loss.item(), pstr)
                print(info)
                metriclog.write(info + "\n")
                metriclog.flush()

        info = 'Train Epoch: {} Avg Loss: {:.4f} Batch_size: {} lr: {:.8f} txt_file: {} n_ctx:{}'.format(epoch, loss_record.avg,args.batch_size,exp_lr_scheduler.get_lr()[0],args.txt_file_name,args.n_ctx)

        print(info)
        metriclog.write(info + "\n")
        metriclog.flush()
        print(f'start_time:{start_time}')
        
        save_epoch = epoch + 1
        if save_epoch == args.epochs:
            torch.save(
                clip_model, 
                f'{pth_save_path}/{args.dataset_name}_clip_ep{save_epoch}_coarse.pth'
            )
        if save_epoch % 15 == 0:    
            torch.cuda.empty_cache()
        torch.cuda.empty_cache()
            
    metriclog.close()   
    
def check_is_grad(model,optim_params=None):
    if optim_params !=None:
        for name, parms in model.named_parameters():
            if name in optim_params:
                print(f'-->name:{name} -->grad_requirs:{parms.requires_grad} -->grad_value:{parms.grad}')
    else:
        for name, parms in model.named_parameters():
            print(f'-->name:{name} -->grad_requirs:{parms.requires_grad} -->grad_value:{parms.grad}')

        
if __name__ == "__main__":
    main()