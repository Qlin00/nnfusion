nnfusion mlp_norelu_small.onnx -f onnx  -fblockfusion_level=0 -fquantize_cfg mlp_coarse_quantize_cfg -fdefault_device=ROCm
