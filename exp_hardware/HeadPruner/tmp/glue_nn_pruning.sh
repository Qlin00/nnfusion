WORK_NUM=$(cat /proc/cpuinfo | grep "processor" | sort | uniq | wc -l)

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python glue_nn_pruner.py \
    --output_dir ./glue_nn_pruning/ \
    --task_name mnli \
    --do_eval \
    --do_train \
    --head_mask_str empty \
    --num_train_epochs 3 \
    --learning_rate 2e-5 \
    --final_threshold 0.3 \
    --block_size_row 64 \
    --block_size_col 768 \
    --max_seq_length 3 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 256 \
    --dataloader_num_workers ${WORK_NUM} \
    --dataloader_pin_memory False \
    --logging_strategy steps \
    --logging_steps 500 \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --load_best_model_at_end True \
    --save_total_limit 3 \
    --model_name_or_path aloxatel/bert-base-mnli \
    --distil_teacher_name_or_path aloxatel/bert-base-mnli \
    --seed 1337
