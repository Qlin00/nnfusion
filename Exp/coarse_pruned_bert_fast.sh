nnfusion bert_pruned.onnx -f onnx  -fblockfusion_level=1  -fgelu_fusion=true -flayernorm_fusion=true -fdefault_device=ROCm
