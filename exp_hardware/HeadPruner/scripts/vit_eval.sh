MASK_STR=$1
MODEL_DIR=$2
WORK_NUM=$(cat /proc/cpuinfo | grep "processor" | sort | uniq | wc -l)

sudo TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 /opt/miniconda/bin/python -m torch.distributed.launch --nproc_per_node=8 ViT_trainer.py \
    --output_dir ./vit_finetune/ \
    --remove_unused_columns False \
    --do_eval \
    --head_mask_str ${MASK_STR} \
    --learning_rate 0.003 \
    --max_steps 30000 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 256 \
    --dataloader_num_workers ${WORK_NUM} \
    --dataloader_pin_memory False \
    --logging_strategy steps \
    --logging_steps 200 \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --load_best_model_at_end True \
    --save_total_limit 3 \
    --model_name_or_path ${MODEL_DIR} \
    --seed 1337