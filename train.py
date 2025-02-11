import argparse
import pickle
import math
import numpy as np
import torch
import torch.nn as nn
from torch.optim import SGD, lr_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd

from data.augmentations import get_transform
from data.get_datasets import get_datasets, get_class_splits

from util.general_utils import AverageMeter, init_experiment
from util.cluster_and_log_utils import log_accs_from_preds
from config import exp_root
from model_clip_sk import DINOHead, info_nce_logits, SupConLoss, DistillLoss, ContrastiveLearningViewGenerator, get_params_groups
import vision_transformer as vits
from loguru import logger
from torch.nn import functional as F

from model_utils import *
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE

import time

@logger.catch
def train(clip_model, projector,train_loader, test_loader, unlabelled_train_loader, args,optimizer=None):    
    optim_params = get_optim_params('ViT-L/14')
    change_need_grad(clip_model.model,optim_params=optim_params)
    
    weighted = WeightedGamma(args)
    # weighted = AverageFusion()
    # weighted = ConcatFusion(args.feat_dim)

    if optimizer == None:
        # optimizer = SGD(list(clip_model.model.parameters()) + list(projector.parameters())+list(weighted.parameters()), lr=args.lr, momentum=args.momentum)
        optimizer = SGD(list(clip_model.model.parameters()) + list(projector.parameters()), lr=args.lr, momentum=args.momentum)
    
    fp16_scaler = None
    if args.fp16:
        fp16_scaler = torch.cuda.amp.GradScaler()
        
    exp_lr_scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.epochs,
            eta_min=args.lr * 1e-3,
        )

    print("args.sinkhorn:", args.sinkhorn)
    cluster_criterion = DistillLoss(
                        args.warmup_teacher_temp_epochs,
                        args.epochs,
                        args.n_views,
                        args.warmup_teacher_temp,
                        args.teacher_temp,
                        sinkhorn = args.sinkhorn
                    )

    warm_up_epoch = args.warm_up_epoch
    
    epoch_times = []
    epoch_memory = []

    for epoch in range(args.epochs):
        loss_record = AverageMeter()
        
        clip_model.train()
        
        start_time = time.time()
        
        for batch_idx, batch in enumerate(train_loader):
            images, class_labels, uq_idxs, mask_lab = batch
            mask_lab = mask_lab[:, 0]
            
            all_img_feats_1, all_txt_feats_1 = clip_model(images[0].cuda())
            all_img_feats_2, all_txt_feats_2 = clip_model(images[1].cuda())
            
            fusion_feat_view1 = weighted(all_img_feats_1,all_txt_feats_1)
            fusion_feat_view2 = weighted(all_img_feats_2,all_txt_feats_2)
            fusion_feat = [fusion_feat_view1,fusion_feat_view2]
            
            class_labels, mask_lab = class_labels.cuda(non_blocking=True), mask_lab.cuda(non_blocking=True).bool()
            
            fusion_feat = torch.cat(fusion_feat,dim=0).cuda(non_blocking=True)
            fusion_proj, fusion_out = projector(fusion_feat.float())

            with torch.cuda.amp.autocast(fp16_scaler is not None):
                
                student_out = fusion_out #[batch_size*2,100]
                student_proj = fusion_proj #[batch_size*2,256]
                loss = 0             
                teacher_out = student_out.detach() #[batch_size*2, 100]
                
                if epoch >= warm_up_epoch:
                    # clustering, sup
                    sup_logits = torch.cat([f[mask_lab] for f in (student_out / 0.1).chunk(2)], dim=0)
                    sup_labels = torch.cat([class_labels[mask_lab] for _ in range(2)], dim=0)                  
                    cls_loss = nn.CrossEntropyLoss()(sup_logits, sup_labels)
                        
                    cluster_loss = cluster_criterion(student_out, teacher_out, epoch-warm_up_epoch)
                    avg_probs_softmax = (student_out / 0.1).softmax(dim=1)
                    avg_probs = avg_probs_softmax.mean(dim=0)
                    me_max_loss = - torch.sum(torch.log(avg_probs**(-avg_probs))) + math.log(float(len(avg_probs)))

                    print("args.memax_weight:", args.memax_weight)
                    cluster_loss += args.memax_weight * me_max_loss 
                    
                    loss += (1 - args.sup_weight) * cluster_loss + args.sup_weight * cls_loss
                    
                else:
                    cls_loss = torch.Tensor([0.0]).cuda()
                    cluster_loss = torch.Tensor([0.0]).cuda()
                
                contrastive_logits, contrastive_labels = info_nce_logits(features=student_proj)
                contrastive_loss = torch.nn.CrossEntropyLoss()(contrastive_logits, contrastive_labels)

                student_sup_proj = torch.cat([f[mask_lab].unsqueeze(1) for f in student_proj.chunk(2)], dim=1)
                student_sup_proj = torch.nn.functional.normalize(student_sup_proj, dim=-1)
                sup_con_labels = class_labels[mask_lab]
                sup_con_loss = SupConLoss()(student_sup_proj, labels=sup_con_labels)
                
                pstr = ''
                pstr += f'cls_loss: {cls_loss.item():.4f} '
                pstr += f'cluster_loss: {cluster_loss.item():.4f} '
                pstr += f'sup_con_loss: {sup_con_loss.item():.4f} '
                pstr += f'contrastive_loss: {contrastive_loss.item():.4f} '

                loss += (1 - args.sup_weight) * contrastive_loss + args.sup_weight * sup_con_loss
                
            # Train acc
            loss_record.update(loss.item(), class_labels.size(0))
            optimizer.zero_grad()
            if fp16_scaler is None:
                loss.backward()
                optimizer.step()
            else:
                fp16_scaler.scale(loss).backward()
                fp16_scaler.step(optimizer)
                fp16_scaler.update()
             
            if batch_idx % args.print_freq == 0:
                args.logger.info('Epoch: [{}][{}/{}]\t loss {:.5f}\t {}'
                            .format(epoch, batch_idx, len(train_loader), loss.item(), pstr))


        args.logger.info('Train Epoch: {} Avg Loss: {:.4f} lr: {:.4f}'.format(epoch, loss_record.avg, exp_lr_scheduler.get_lr()[0]))
        
        end_time = time.time()
        
        elapsed_time = end_time - start_time
        epoch_times.append(elapsed_time)
        print(f"running time: {elapsed_time:.2f} sec")  # average：492s 11834 MB
        epoch_memory.append(torch.cuda.memory_allocated(0) / 1024**2)
        get_gpu_memory(0)
        
        # Step schedule
        exp_lr_scheduler.step()
        
        if epoch >= warm_up_epoch:
            args.logger.info('Testing on unlabelled examples in the training data...')
            all_acc, old_acc, new_acc = test(clip_model, projector,unlabelled_train_loader, epoch=epoch,weighted=weighted,save_name=f'{args.dataset_name}_CLIP_ACC_train', args=args)  
            args.logger.info('Train Accuracies: All {:.4f} | Old {:.4f} | New {:.4f}'.format(all_acc, old_acc, new_acc))
           
            save_dict = {
                'model': clip_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch + 1,
            }
            
            torch.save(save_dict, args.model_path)
            args.logger.info("model saved to {}.".format(args.model_path))

# def get_gpu_memory(gpu_index = 0):
#     if torch.cuda.is_available():
#         reserved_memory = torch.cuda.memory_reserved(gpu_index) / 1024**2
#         allocated_memory = torch.cuda.memory_allocated(gpu_index) / 1024**2
#         print(f"GPU : {reserved_memory:.2f} MiB reserved, {allocated_memory:.2f} MiB allocated")
#     else:
#         print("CUDA is not available")
        

def test(model, test_loader, epoch, save_name, args):

    model.eval()

    preds, targets = [], []
    mask = np.array([])
    for batch_idx, (images, label, _) in enumerate(tqdm(test_loader)):
        images = images.cuda(non_blocking=True)
        with torch.no_grad():
            _, logits = model(images)
            preds.append(logits.argmax(1).cpu().numpy())
            targets.append(label.cpu().numpy())
            mask = np.append(mask, np.array([True if x.item() in range(len(args.train_classes)) else False for x in label]))

    preds = np.concatenate(preds)
    targets = np.concatenate(targets)
    all_acc, old_acc, new_acc = log_accs_from_preds(y_true=targets, y_pred=preds, mask=mask,
                                                    T=epoch, eval_funcs=args.eval_funcs, save_name=save_name,
                                                    args=args)

    return all_acc, old_acc, new_acc

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='cluster', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--eval_funcs', nargs='+', help='Which eval functions to use', default=['v2', 'v2p'])

    parser.add_argument('--warmup_model_dir', type=str, default=None)
    parser.add_argument('--dataset_name', type=str, default='scars', help='options: cifar10, cifar100, imagenet_100, cub, scars, fgvc_aricraft, herbarium_19')
    parser.add_argument('--prop_train_labels', type=float, default=0.5)
    parser.add_argument('--use_ssb_splits', action='store_true', default=True)

    parser.add_argument('--grad_from_block', type=int, default=11)
    parser.add_argument('--lr', type=float, default=0.1)
    # parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    # parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--exp_root', type=str, default=exp_root)
    parser.add_argument('--transform', type=str, default='imagenet')
    parser.add_argument('--sup_weight', type=float, default=0.35)
    parser.add_argument('--n_views', default=2, type=int)
    
    parser.add_argument('--memax_weight', type=float, default=2)
    parser.add_argument('--warmup_teacher_temp', default=0.07, type=float, help='Initial value for the teacher temperature.')
    parser.add_argument('--teacher_temp', default=0.04, type=float, help='Final value (after linear warmup)of the teacher temperature.')
    parser.add_argument('--warmup_teacher_temp_epochs', default=30, type=int, help='Number of warmup epochs for the teacher temperature.')

    parser.add_argument('--fp16', action='store_true', default=False)
    parser.add_argument('--print_freq', default=10, type=int)
    parser.add_argument('--exp_name', default=None, type=str)
    parser.add_argument('--warm_up_epoch', default=0, type=str)
    parser.add_argument('--sinkhorn', default=0.2, type=float)
    
    
    # ----------------------
    # INIT
    # ----------------------
    args = parser.parse_args()
    device = torch.device('cuda:0')
    args = get_class_splits(args)

    args.num_labeled_classes = len(args.train_classes)
    
    # ============================== the number of unlabeled classes ==============================
    args.num_unlabeled_classes = len(args.unlabeled_classes)
    # =============================================================================================

    init_experiment(args, runner_name=['simgcd'])
    args.logger.info(f'Using evaluation function {args.eval_funcs[0]} to print results')
    
    torch.backends.cudnn.benchmark = True

    # ----------------------
    # BASE MODEL
    # ----------------------
    args.interpolation = 3
    args.crop_pct = 0.875

    # NOTE: Hardcoded image size as we do not finetune the entire ViT model
    args.image_size = 224
    args.feat_dim = 768
    args.num_mlp_layers = 3
    args.mlp_out_dim = args.num_labeled_classes + args.num_unlabeled_classes

    args.logger.info('model build')

    # --------------------
    # CONTRASTIVE TRANSFORM
    # --------------------
    train_transform, test_transform = get_transform(args.transform, image_size=args.image_size, args=args)
    train_transform = ContrastiveLearningViewGenerator(base_transform=train_transform, n_views=args.n_views)
    
    # --------------------
    # DATASETS
    # --------------------
    train_dataset, test_dataset, unlabelled_train_examples_test, datasets = get_datasets(args.dataset_name,
                                                                                         train_transform,
                                                                                         test_transform,
                                                                                         args)

    # --------------------
    # SAMPLER
    # Sampler which balances labelled and unlabelled examples in each batch
    # --------------------
    label_len = len(train_dataset.labelled_dataset)
    unlabelled_len = len(train_dataset.unlabelled_dataset)
    sample_weights = [1 if i < label_len else label_len / unlabelled_len for i in range(len(train_dataset))]
    sample_weights = torch.DoubleTensor(sample_weights)
    sampler = torch.utils.data.WeightedRandomSampler(sample_weights, num_samples=len(train_dataset))

    # --------------------
    # DATALOADERS
    # --------------------
    train_loader = DataLoader(train_dataset, num_workers=0, batch_size=args.batch_size,sampler=sampler, shuffle=False,drop_last=True)

    test_loader_unlabelled = DataLoader(unlabelled_train_examples_test, num_workers=args.num_workers,
                                        batch_size=256, shuffle=False, pin_memory=False,drop_last=True)
  
    # ----------------------
    # PROJECTION HEAD
    # ----------------------
    projector = DINOHead(in_dim=args.feat_dim, out_dim=args.mlp_out_dim, nlayers=args.num_mlp_layers).cuda()

    pretrained_model_path = './result/pretrained_model_save/cub_clip_ep100.pth' # cub
    
    clip_model = torch.load(pretrained_model_path)
   
    # ----------------------
    # TRAIN
    # ----------------------
    train(clip_model, projector, train_loader, None, test_loader_unlabelled, args,optimizer=None)


    
    
