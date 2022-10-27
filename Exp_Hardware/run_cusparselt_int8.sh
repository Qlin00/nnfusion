rm -rf ~/.cache/nnfusion/kernel_cache.db
prefix=cusparselt_int8_bert_large_n_16_m_32_align1
out_dir=nnfusion_cfg_test
mkdir $out_dir
cp ${prefix}/* $out_dir
pushd $out_dir
#nnfusion model_tesa.onnx -f onnx -flayernorm_fusion=1 -fgelu_fusion=1 -fspargen_cfg config -fsparse_dot_transpose=true
nnfusion model_tesa.onnx -f onnx -flayernorm_fusion=0 -fgelu_fusion=1 -fspargen_cfg config
popd
cp CusparseLtCMakeLists.txt ${out_dir}/nnfusion_rt/cuda_codegen/CMakeLists.txt

