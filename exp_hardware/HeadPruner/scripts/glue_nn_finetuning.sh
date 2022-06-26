echo finetuing $1 save to $2
export TASK_NAME=mnli
python  glue_nn_pruner.py \
  --model_name_or_path $1 \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --max_seq_length 128 \
  --per_device_train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 12 \
  --head_mask_str empty \
  --final_threshold 1.0 \
  --logging_steps 100 \
  --distil_teacher_name_or_path gchhablani/bert-base-cased-finetuned-mnli \
  --block_size_row 32 \
  --block_size_col 32 \
  --evaluation_strategy epoch \
  --output_dir $2
