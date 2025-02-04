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

    '''
    初始化optimizer
    '''
    if optimizer == None:
        #判断是不是读取预训练的模型，不是的话读取optimizer
        # gamma学习
        # optimizer = SGD(list(clip_model.model.parameters()) + list(projector.parameters())+list(weighted.parameters()), lr=args.lr, momentum=args.momentum)
        optimizer = SGD(list(clip_model.model.parameters()) + list(projector.parameters()), lr=args.lr, momentum=args.momentum)
    
    fp16_scaler = None
    if args.fp16:
        fp16_scaler = torch.cuda.amp.GradScaler()


    '''
    余弦退火动态调节lr
    '''
    exp_lr_scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.epochs,
            eta_min=args.lr * 1e-3,
        )


    '''
    初始化DistillLoss
    '''
    print("args.sinkhorn:", args.sinkhorn)
    cluster_criterion = DistillLoss(
                        args.warmup_teacher_temp_epochs,
                        args.epochs,
                        args.n_views,
                        args.warmup_teacher_temp,
                        args.teacher_temp,
                        sinkhorn = args.sinkhorn
                    )

    
    '''
    warm_up的epoch数
    '''
    warm_up_epoch = args.warm_up_epoch #默认不warm_up
    

    epoch_times = []
    epoch_memory = []
    '''
    正式训练
    '''
    for epoch in range(args.epochs):
        loss_record = AverageMeter()
        
        clip_model.train()
        
        start_time = time.time()
        
        for batch_idx, batch in enumerate(train_loader):
            images, class_labels, uq_idxs, mask_lab = batch
            mask_lab = mask_lab[:, 0]
            '''
            训练的时候，mask_lab用于区分的是labeled 和unlabeled
            测试的时候，mask用于区分old和novel
            '''
            #images是图像，class_labels是对应的gt真值，
            #uq_idxs是生成dataset，DataLoader打乱数据之前为每个数据赋予的id，可以用于获得对应的class_labels
            #mask_lab用于区分哪些数据是labeled 和 unlabeled
            
            
            '''
            1.获得两个view的img和txt的特征
            '''
            all_img_feats_1, all_txt_feats_1 = clip_model(images[0].cuda())
            all_img_feats_2, all_txt_feats_2 = clip_model(images[1].cuda())
            

            '''
            ======================= 获得融合特征_加权求和 ==============================
            '''
            fusion_feat_view1 = weighted(all_img_feats_1,all_txt_feats_1)
            fusion_feat_view2 = weighted(all_img_feats_2,all_txt_feats_2)
            fusion_feat = [fusion_feat_view1,fusion_feat_view2]
            
            
            class_labels, mask_lab = class_labels.cuda(non_blocking=True), mask_lab.cuda(non_blocking=True).bool()
            fusion_feat = torch.cat(fusion_feat,dim=0).cuda(non_blocking=True)#shape为[batch_size*2,768]
            '''
            batch_size*2是因为两个view
            '''
            

            '''
            3.通过fusion_out获得proj和out
            '''
            fusion_proj, fusion_out = projector(fusion_feat.float())
            #fusion_proj的shape是#[batch_size*2,256]，它经过了mlp
            #fusion_out的shape是#[batch_size*2,100]，它经过了线性层（也就是prototypes），prototypes就是用一个线性层表示的
            #256是projection的bottleneck的输出

        
            with torch.cuda.amp.autocast(fp16_scaler is not None):
                
                student_out = fusion_out #[batch_size*2,100]
                student_proj = fusion_proj #[batch_size*2,256]
                #换个名字，用SimGCD原来的变量名

                
                loss = 0  #最终的loss              
                teacher_out = student_out.detach() #[batch_size*2, 100]
                #detach用于保证梯度不传递
                
                
                '''
                warmp_up时，这里不训练
                '''
                if epoch >= warm_up_epoch:
                    '''
                    传统监督学习
                    '''
                    # clustering, sup
                    sup_logits = torch.cat([f[mask_lab] for f in (student_out / 0.1).chunk(2)], dim=0)
                    sup_labels = torch.cat([class_labels[mask_lab] for _ in range(2)], dim=0)                  
                    cls_loss = nn.CrossEntropyLoss()(sup_logits, sup_labels)
                        
                        
                    '''
                    半监督学习，teacher那支更加sharp
                    '''
                    #epoch-warm_up_epoch是为了减去没有计算distill loss的epoch
                    #因为teacher的τ温度是持续变化的，一开始warm_up的时候不变
                    cluster_loss = cluster_criterion(student_out, teacher_out, epoch-warm_up_epoch)
                    avg_probs_softmax = (student_out / 0.1).softmax(dim=1)
                    avg_probs = avg_probs_softmax.mean(dim=0)
                    me_max_loss = - torch.sum(torch.log(avg_probs**(-avg_probs))) + math.log(float(len(avg_probs)))
                    #计算mean_entropy_max，正则项
                    
                    '''
                    加上正则项
                    '''
                    print("args.memax_weight:", args.memax_weight)
                    cluster_loss += args.memax_weight * me_max_loss 
                    #这里是+=，所以上面的cluster_loss有用，这里是减去me_max_loss(上面有负号)
                    
                    '''
                    监督学习loss+半监督学习，加权
                    '''
                    loss += (1 - args.sup_weight) * cluster_loss + args.sup_weight * cls_loss
                    #加权
                    
                else:
                    cls_loss = torch.Tensor([0.0]).cuda()
                    cluster_loss = torch.Tensor([0.0]).cuda()
                


                '''
                represent learning, unsup
                自监督对比学习
                '''
                contrastive_logits, contrastive_labels = info_nce_logits(features=student_proj)
                contrastive_loss = torch.nn.CrossEntropyLoss()(contrastive_logits, contrastive_labels)

                '''
                representation learning, sup
                监督对比学习
                '''
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
             
            #看情况可以在这边加参数，来保证log里面看得见
            if batch_idx % args.print_freq == 0:
                args.logger.info('Epoch: [{}][{}/{}]\t loss {:.5f}\t {}'
                            .format(epoch, batch_idx, len(train_loader), loss.item(), pstr))


        args.logger.info('Train Epoch: {} Avg Loss: {:.4f} lr: {:.4f}'.format(epoch, loss_record.avg, exp_lr_scheduler.get_lr()[0]))
        
        end_time = time.time()
        
        elapsed_time = end_time - start_time
        epoch_times.append(elapsed_time)
        print(f"程序运行时间: {elapsed_time:.2f} 秒")  # average：492秒 11834 MB
        epoch_memory.append(torch.cuda.memory_allocated(0) / 1024**2)
        get_gpu_memory(0)
        
        # Step schedule
        exp_lr_scheduler.step()
        
    
        '''
        超过warm_up_epoch才进行测试
        '''
        if epoch >= warm_up_epoch:
            args.logger.info('Testing on unlabelled examples in the training data...')
            all_acc, old_acc, new_acc = test(clip_model, projector,unlabelled_train_loader, epoch=epoch,weighted=weighted,save_name=f'{args.dataset_name}_CLIP_ACC_train', args=args)  
            args.logger.info('Train Accuracies: All {:.4f} | Old {:.4f} | New {:.4f}'.format(all_acc, old_acc, new_acc))
           

            '''
            按照dict形式保存模型
            '''
            save_dict = {
                'model': clip_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch + 1,
            }
            
            torch.save(save_dict, args.model_path)
            args.logger.info("model saved to {}.".format(args.model_path))

'''
测试用的函数，很重要，画图也部分借用这个函数
'''
'''
训练的时候，mask_lab用于区分的是labeled 和unlabeled
测试的时候，mask用于区分old和novel
'''
def get_gpu_memory(gpu_index = 0):
    if torch.cuda.is_available():
        reserved_memory = torch.cuda.memory_reserved(gpu_index) / 1024**2
        allocated_memory = torch.cuda.memory_allocated(gpu_index) / 1024**2
        print(f"GPU : {reserved_memory:.2f} MiB reserved, {allocated_memory:.2f} MiB allocated")
    else:
        print("CUDA is not available")
        
def test(model, projector, test_loader, epoch, weighted, save_name, args):
    model.eval()
    
    print("testing...")
    
    preds, targets = [], []
    mask = np.array([])
    
    class_prediction_counts = np.zeros(200) # imagenet_100 100 # cifar 100 cub:200 aircraft:100 scars:196
    
    # 初始化混淆矩阵
    confusion_matrix = np.zeros((2, 2), dtype=int)
    # print(len(test_loader.dataset.targets))
    
    all_features = [] # 用于存储所有的特征
    all_labels = [] # 用于存储所有的标签
    
    for batch_idx, (images, label, _) in enumerate(tqdm(test_loader)):
        images = images.cuda(non_blocking=True)
        all_img_feats, all_txt_feats = model(images.cuda())
        fusion_feat = weighted(all_img_feats,all_txt_feats)
        
        with torch.no_grad():
            # _, logits = model(images)
            _, logits = projector(fusion_feat.float())
            
            batch_preds = logits.argmax(1).cpu().numpy()
            preds.append(logits.argmax(1).cpu().numpy()) # preds存储的是预测结果的索引
            
            all_features.append(fusion_feat.cpu().numpy())
            all_labels.append(label.cpu().numpy())
            #与传统分类器一样，每个样本选择logits中最大的那个
            targets.append(label.cpu().numpy())
            # 更新每个类的预测次数
            for pred in batch_preds:
                class_prediction_counts[pred] += 1
                
            mask = np.append(mask, np.array([True if x.item() in range(len(args.train_classes)) else False for x in label]))
            
            # 更新混淆矩阵
            for p, true in zip(batch_preds, label.cpu().numpy()):
                if true < 100:  # 实际为旧类
                    if p < 100:
                        confusion_matrix[0, 0] += 1  # TP
                    else:
                        confusion_matrix[0, 1] += 1  # FN
                else:  # 实际为新类
                    if p < 100:
                        confusion_matrix[1, 0] += 1  # FP
                    else:
                        confusion_matrix[1, 1] += 1  # TN

    preds = np.concatenate(preds)
    targets = np.concatenate(targets)
    all_acc, old_acc, new_acc = log_accs_from_preds(y_true=targets, y_pred=preds, mask=mask,
                                                    T=epoch, eval_funcs=args.eval_funcs, save_name=save_name,
                                                    args=args)
    # 只取范围 (80, 100) 的类进行可视化
    selected_counts = class_prediction_counts[100:200] # cub: [100:200] # āircraft: [50:100] # scars: [98,196] # imagenet_100: [50:100]

    # 根据预测次数从大到小排序
    sorted_indices = np.argsort(selected_counts)[::-1]
    sorted_counts = selected_counts[sorted_indices]
    
    if epoch in [0, 145, 199]: # aircraft: [49]
        
    # 可视化预测分布
        plt.style.use('default')
        plt.figure(figsize=(20, 16),dpi=300)
        plt.bar(range(100), sorted_counts, color='deepskyblue') # cifar100: 20, cub: 100, aircraft: 50, scars: 98, imagenet_100: 50
        plt.xlabel('Class Index (100-199)',fontsize=20)
        plt.ylabel('Instance Count',fontsize=20)
        plt.title('Per-Class Prediction Distributions (100-200)',fontsize=20)
        plt.xticks(range(0, 100, 10),fontsize=20)  # 确保使用排序后的类索引
        plt.grid(axis='y')
        plt.savefig(f'./result_visualaization/{save_name}_class_distribution_epoch_{epoch}.png')  # 保存图像
        plt.show()  # 显示图像

        
        
        # 可视化混淆矩阵
        confusion_matrix_percent = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis] * 100
        labels = np.array([[f"{value:.2f}%" for value in row] for row in confusion_matrix_percent])
        format_func = np.vectorize(lambda x: f"{x:.2f}%")
        labels = format_func(confusion_matrix_percent)
        plt.figure(figsize=(20, 16),dpi=300)
        sns.heatmap(confusion_matrix_percent, annot=labels, fmt="", cmap="Greens", cbar=False, annot_kws={"fontsize": 40},
                    xticklabels=[f"{args.dataset_name} (Old)", f"{args.dataset_name} (New)"],
                    yticklabels=[f"{args.dataset_name} (Old)", f"{args.dataset_name} (New)"])
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.ylabel('Ground truth',fontsize=40)
        plt.xlabel('Predicition',fontsize=40)
        plt.title('Confusion_Matrix', fontsize=40)
        plt.savefig(f'./result_visualaization/{save_name}_confusionMatrix_epoch_{epoch}.png')  # 如果需要保存图像，可以设置文件名
        plt.show()
        
        # t-SNE 可视化
        # 将特征和标签转换为numpy数组
        all_features = np.concatenate(all_features)
        all_labels = np.concatenate(all_labels)
        
        # 选择需要可视化的类别，这里选择范围内的类
        selected_mask = (all_labels >= 100) & (all_labels < 200)
        selected_features = all_features[selected_mask]
        selected_labels = all_labels[selected_mask]
        
        # 为了加快t-SNE的计算速度，可以随机采样部分数据点
        # if len(selected_labels) > 1000:
        #     np.random.seed(42)  # 设置随机种子以保证可重复性
        #     indices = np.random.choice(len(selected_labels), 1000, replace=False)
        #     selected_features = selected_features[indices]
        #     selected_labels = selected_labels[indices]
        
        # 执行t-SNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
        tsne_results = tsne.fit_transform(selected_features)
        
        # 绘制t-SNE结果
        plt.figure(figsize=(16, 12), dpi=300)
        scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=selected_labels, cmap='viridis', s=10)
        # plt.colorbar(scatter,ticks=range(80, 100))
        plt.colorbar()
        plt.title('t-SNE Visualization of Features (Classes 100-199)', fontsize=20)
        plt.savefig(f'./result_visualaization/{save_name}_tsne_epoch_{epoch}.png')
        plt.show()
        
        
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
    # args.num_unlabeled_classes = 100
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
    #它输出x_proj, logits
    #具体内容看DINOHead的forward

    '''
    获取一阶段的预训练模型，这里的路径就是一阶段预训练模型的存储路径
    '''

    # pretrained_model_path = get_pretrained_model(args)
    # pretrained_model_path = './result/pretrained_model_save/2024_09_13_20_07_28/cifar100_clip_ep100.pth' # cifar100
    pretrained_model_path = './result/pretrained_model_save/2024_10_06_02_46_54/cub_clip_ep100.pth' # cub
    # pretrained_model_path = './result/pretrained_model_save/2024_10_06_18_28_33/aircraft_clip_ep50.pth' # aircraft
    # pretrained_model_path = './result/pretrained_model_save/2024_10_07_20_52_30/scars_clip_ep50.pth' #scars
    # pretrained_model_path = './result/pretrained_model_save/2024_10_19_12_35_38/imagenet_100_clip_ep50.pth' # imagenet_100
    # pretrained_model_path = './result/pretrained_model_save/2024_11_18_12_46_28/cifar100_clip_ep100_coarse.pth' # cifar100_coarse
    
    
    clip_model = torch.load(pretrained_model_path)
   
    # ----------------------
    # TRAIN
    # ----------------------
    '''
    开始训练
    '''
    train(clip_model, projector, train_loader, None, test_loader_unlabelled, args,optimizer=None)


    
    
