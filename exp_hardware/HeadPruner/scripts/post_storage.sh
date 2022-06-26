OUTPUT_DIR=/blob/Experiments/Pruning/nn_pruning/GLUE/mnli/$AMLT_JOB_NAME/$AMLT_EXPERIMENT_NAME
mkdir -p $OUTPUT_DIR

# log copy
cp -r $AZUREML_CR_HT_CAP_user_logs_PATH $OUTPUT_DIR

# ckpt copy
cp -r /tmp/code/fluency_final_0.100_32x32 $OUTPUT_DIR
cp -r /tmp/code/fluency_final_1.000_32x32 $OUTPUT_DIR
