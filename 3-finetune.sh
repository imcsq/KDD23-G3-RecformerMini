export CUDA_VISIBLE_DEVICES=1

names="Scientific Instruments Arts Office Games Pet"

for name in $names
do
echo $name
python finetune.py \
    --model_name_or_path kiddothe2b/longformer-mini-1024 \
    --pretrain_ckpt "pretrain_ckpt/seqrec_pretrain_ckpt-v5-epoch=0-avg_val_accuracy=0.1368.bin" \
    --data_path finetune_data/$name \
    --num_train_epochs 128 \
    --batch_size 80 \
    --device 0 \
    --fp16 \
    --finetune_negative_sample_size -1 \
    --max_token_num 512 \
    --learning_rate 1e-4
done
