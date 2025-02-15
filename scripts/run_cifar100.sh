set -e
set -x

python ./SimGCD_ovdet/train.py \
    --dataset_name 'cifar100' \
    --batch_size 128 \
    --grad_from_block 11 \
    --epochs 200 \
    --num_workers 0 \
    --use_ssb_splits \
    --sup_weight 0.35 \
    --weight_decay 5e-5 \
    --transform 'imagenet' \
    --lr 0.1 \
    --eval_funcs 'v2' \
    --warmup_teacher_temp 0.07 \
    --teacher_temp 0.04 \
    --warmup_teacher_temp_epochs 30 \
    --memax_weight 2 \
    --sinkhorn 0 \
    --exp_name cifar100_Progcd
