timestamp=`date "+%Y%0m%0d_%T"`
model_dir="checkpoint"
data_dir="/atbm/data"
wandb_run_name="atbm"
s="123"
lr="5e-5"

CUDA_VISIBLE_DEVICES=2 python train.py \
        --model_dir $model_dir \
        --data_dir $data_dir\
        --model_type malware-url \
        --wandb_run_name $wandb_run_name \
        --seed $s \
        --num_train_epochs 50 \
        --train_batch_size 512 \
        --max_seq_len 128 \
        --learning_rate $lr \
        --early_stopping 250 \
        --d_model 768 \
        --hidden_size 768 \
        --num_heads 12 \
        --dropout_rate 0.1 \
        --eps 1e9
