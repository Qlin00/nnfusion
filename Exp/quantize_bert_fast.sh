nnfusion bert_original.onnx -f onnx  -fblockfusion_level=0 -fquantize_cfg bert_cfg   -fdefault_device=ROCm -fgelu_fusion=true
