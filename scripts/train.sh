timestamp=`date "+%Y%0m%0d_%T"`
model_dir="checkpoint"
data_dir="/home/black/atbm/data"
wandb_run_name="atbm"
s="123"
lr="5e-5"

CUDA_VISIBLE_DEVICES=0 python train.py \
        --model_dir $model_dir \
        --data_dir $data_dir\
        --model_type roberta-base \
        --wandb_run_name $wandb_run_name \
        --seed $s \
        --num_train_epochs 10 \
        --train_batch_size 256 \
        --max_seq_len 64 \
        --learning_rate $lr \
        --early_stopping 25
