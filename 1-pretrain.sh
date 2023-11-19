# Use distributed data parallel
CUDA_VISIBLE_DEVICES=2 python lightning_pretrain.py \
    --model_name_or_path kiddothe2b/longformer-mini-1024 \
    --train_file pretrain_data/train.json \
    --dev_file pretrain_data/dev.json \
    --item_attr_file pretrain_data/meta_data.json \
    --output_dir result/recformer_pretraining \
    --num_train_epochs 32 \
    --gradient_accumulation_steps 8 \
    --preprocessing_num_workers 4 \
    --dataloader_num_workers 4 \
    --batch_size 80 \
    --learning_rate 1e-5 \
    --temp 0.05 \
    --device 1 \
    --fp16 \
    --fix_word_embedding
