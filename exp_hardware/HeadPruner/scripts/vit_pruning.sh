WORK_NUM=$(cat /proc/cpuinfo | grep "processor" | sort | uniq | wc -l)

TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 ViT_pruner.py \
    --output_dir ./beans_outputs/ \
    --remove_unused_columns False \
    --do_eval \
    --learning_rate 2e-5 \
    --num_train_epochs 5 \
    --per_device_train_batch_size 196 \
    --per_device_eval_batch_size 196 \
    --dataloader_num_workers ${WORK_NUM} \
    --dataloader_pin_memory False \
    --logging_strategy steps \
    --logging_steps 10 \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --load_best_model_at_end True \
    --save_total_limit 3 \
    --model_name_or_path google/vit-base-patch16-384 \
    --seed 1337