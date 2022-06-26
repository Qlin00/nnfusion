"""
# Export the ViT ONNX model
python -m transformers.onnx --model=google/vit-base-patch16-384 --feature image-classification onnxm/
"""
import time
from typing import Union, List
from transformers.onnx import OnnxConfig
from transformers import ViTConfig, ViTForImageClassification, ViTFeatureExtractor, PreTrainedTokenizerBase, PreTrainedModel, is_torch_available, TensorType

import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

from copy import deepcopy
import datasets
import numpy as np
import torch
from datasets import load_dataset
from PIL import Image
from scipy.special import softmax
import torchvision.transforms as transforms
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    ToTensor,
)

# import wandb

import transformers
from transformers import (
    MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING,
    AutoConfig,
    AutoFeatureExtractor,
    AutoModelForImageClassification,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    ViTForImageClassification,
    ViTFeatureExtractor,
    PreTrainedModel,
    is_torch_available,
    TensorType,
)
from data.dataset import ImageFolder

def pil_loader(path: str):
    with open(path, "rb") as f:
        im = Image.open(f)
        return im.convert("RGB")


def build_dataset(zip_dir: str, img_prefix: str, info_path: str, transform):
    return ImageFolder(
        zip_dir,
        ann_file=info_path,
        img_prefix=img_prefix,
        transform=transform,
    )

class ViTOnnxConfig(OnnxConfig):
    @property
    def inputs(self):
        return OrderedDict(
            [
                ("pixel_values", {0: "batch", 1: "sequence"}),
            ]
        )

    @property
    def atol_for_validation(self) -> float:
        return 1e-3

def validate_model_outputs(
    config: OnnxConfig,
    preprocessor: Union["PreTrainedTokenizer", "FeatureExtractionMixin"],
    onnx_model,
    onnx_named_outputs: List[str],
    tokenizer: "PreTrainedTokenizer" = None,
):
    from onnxruntime import InferenceSession, SessionOptions

    print("Validating ONNX model...")

    if isinstance(preprocessor, PreTrainedTokenizerBase) and tokenizer is not None:
        raise ValueError("You cannot provide both a tokenizer and a preprocessor to validatethe model outputs.")
    if tokenizer is not None:
        warnings.warn(
            "The `tokenizer` argument is deprecated and will be removed in version 5 of Transformers. Use"
            " `preprocessor` instead.",
            FutureWarning,
        )
        print("Overwriting the `preprocessor` argument with `tokenizer` to generate dummmy inputs.")
        preprocessor = tokenizer

    feature_extractor = preprocessor

    # Define torchvision transforms to be applied to each image.
    normalize = Normalize(
        mean=feature_extractor.image_mean, std=feature_extractor.image_std
    )
    _val_transforms = Compose(
        [
            Resize(feature_extractor.size),
            CenterCrop(feature_extractor.size),
            ToTensor(),
            normalize,
        ]
    )

    def val_transforms(example_batch):
        """Apply _val_transforms across a batch."""
        example_batch["pixel_values"] = [
            _val_transforms(pil_img.convert("RGB"))
            for pil_img in example_batch["image"]
        ]
        return example_batch

    test_dataset = build_dataset(
        "/tmp/imagenet/", "val.zip@/", "val_map.txt", _val_transforms
    )

    dataset = {"test": test_dataset}
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Create ONNX Runtime session
    options = SessionOptions()
    session = InferenceSession(onnx_model.as_posix(), options, providers=["CPUExecutionProvider"])

    # We flatten potential collection of inputs (i.e. past_keys)
    onnx_inputs = []
    num = 0
    for reference_model_inputs in test_dataloader:
        onnx_input = {}
        for name, value in reference_model_inputs.items():
            if name == "labels":
                continue
            if isinstance(value, (list, tuple)):
                value = config.flatten_output_collection_property(name, value)
                onnx_input.update({tensor_name: pt_tensor.numpy() for tensor_name, pt_tensor in value.items()})
            else:
                onnx_input[name] = value.numpy()
        onnx_inputs.append(onnx_input)
        num += 1
        if num > 100:
            break

    print("Start Run ONNX model...")
    # Compute outputs from the ONNX model
    start = time.time()
    for onnx_input in onnx_inputs:
        onnx_outputs = session.run(onnx_named_outputs, onnx_input)
    end = time.time()
    print(end - start)

if __name__ == '__main__':
    config = ViTConfig.from_pretrained("google/vit-base-patch16-384")
    onnx_config = ViTOnnxConfig(config, "image-classification")
    feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-384")
    validate_model_outputs(onnx_config, feature_extractor, Path("../onnx/model_int8.onnx"),['logits'])