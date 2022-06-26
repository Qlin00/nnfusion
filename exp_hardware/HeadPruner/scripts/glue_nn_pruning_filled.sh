export TASK_NAME=mnli

TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 glue_nn_pruner.py \
  --model_name_or_path fluency_final_0.100_32x32 \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --max_seq_length 128 \
  --per_device_train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 6 \
  --head_mask_str empty \
  --final_threshold 1.0 \
  --logging_steps 100 \
  --distil_teacher_name_or_path /tmp/bert-base-mnli/ \
  --block_size_row 32 \
  --block_size_col 32 \
  --evaluation_strategy epoch \
  --report_to azure_ml wandb \
  --output_dir /tmp/kk_$TASK_NAME/