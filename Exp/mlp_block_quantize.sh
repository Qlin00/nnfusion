nnfusion mlp_norelu.onnx -f onnx -fblockquantize_cfg mlp_norelu_block_config -fblockfusion_level=0 -fdefault_device=ROCm
