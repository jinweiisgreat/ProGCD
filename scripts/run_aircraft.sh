set -e
set -x

python ./train.py \
    --dataset_name 'aircraft' \
    --batch_size 128 \
    --grad_from_block 11 \
    --epochs 50 \
    --num_workers 0 \
    --use_ssb_splits \
    --sup_weight 0.35 \
    --weight_decay 5e-5 \
    --transform 'imagenet' \
    --lr 0.05 \
    --eval_funcs 'v2' \
    --warmup_teacher_temp 0.07 \
    --teacher_temp 0.04 \
    --warmup_teacher_temp_epochs 10 \
    --memax_weight 1 \
    --sinkhorn 0.2 \
    --exp_name aircraft_Progcd

    # warmup_teacher_temp_epochs: 50->10; 100->20; 200->30;
