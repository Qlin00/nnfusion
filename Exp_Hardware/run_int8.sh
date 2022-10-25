rm -rf ~/.cache/nnfusion/kernel_cache.db
prefix=balance_bert_large_n_16_m_32_align64
out_dir=nnfusion_cfg_test
mkdir -p $out_dir
#cp ${prefix}/config $out_dir
cp ${prefix}/Constants/* $out_dir
cp ${prefix}/* $out_dir
python prepare_cfg_int8.py --in_dir $prefix  --out_dir $out_dir
pushd $out_dir
nnfusion model_tesa.onnx -f onnx -flayernorm_fusion=1 -fgelu_fusion=1 -fspargen_cfg config
popd

