tuning_step=2000
nnfusion model.onnx -f onnx -fkernel_tuning_steps=$tuning_step -fantares_mode=1 -fantares_codegen_server="127.0.0.1:10088" -fir_based_fusion=true -fkernel_fusion_level=0 -fblockfusion_level=0 
