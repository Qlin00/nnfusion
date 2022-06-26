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
)
from data.dataset import ImageFolder
from head_pruner import HeadPruner

""" Fine-tuning a 🤗 Transformers model for image classification"""

logger = logging.getLogger(__name__)

MODEL_CONFIG_CLASSES = list(MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


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


class ViTHeadPruner(HeadPruner):
    def __init__(
        self,
        trainer,
        method: str,
        valid_dataset,
        test_dataset,
        config: dict,
        params: dict,
    ):
        self.trainer = trainer
        super(ViTHeadPruner, self).__init__(
            trainer.model, method, valid_dataset, test_dataset, config, params
        )

    def predict(self, model, dataset, config: dict):
        """return pred_prob, pred_labels, metrics"""
        trainer = self.trainer
        trainer.model = model

        metrics = trainer.evaluate(dataset)
        # pred_prob = softmax(pred_prob, -1)
        # pred_labels = np.argmax(pred_prob, axis=2)
        # sparity = get_sparity(model)
        # metrics = {**metrics, **sparity}
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
        return [], [], metrics

    def mask_head(self, model, heads: list, config: dict):
        """return model"""
        model = deepcopy(self.model)
        model.prune_heads(
            {layer_id: head_ids for layer_id, head_ids in enumerate(heads)}
        )
        return model


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        default="google/vit-base-patch16-224-in21k",
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={
            "help": "If training from scratch, pass a model type from the list: "
            + ", ".join(MODEL_TYPES)
        },
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained config name or path if not the same as model_name"
        },
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where do you want to store the pretrained models downloaded from s3"
        },
    )
    model_revision: str = field(
        default="main",
        metadata={
            "help": "The specific model version to use (can be a branch name, tag name or commit id)."
        },
    )
    feature_extractor_name: str = field(
        default=None, metadata={"help": "Name or path of preprocessor config."}
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )


def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.tensor([example["labels"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, training_args = parser.parse_args_into_dataclasses()

    # if training_args.local_rank in [-1, 0]:
    #     wandb.init(project="ViT-pruning-experiment")
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None

    # Prepare label mappings.
    # We'll include these in the model's config to get human readable labels in the Inference API.
    # labels = dataset["train"].features["labels"].names
    # label2id, id2label = dict(), dict()
    # for i, label in enumerate(labels):
    #     label2id[label] = str(i)
    #     id2label[str(i)] = label

    # Load the accuracy metric from the datasets package
    metric = datasets.load_metric("accuracy")

    # Define our compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with a
    # predictions and label_ids field) and has to return a dictionary string to float.
    def compute_metrics(p):
        """Computes accuracy on a batch of predictions"""
        return metric.compute(
            predictions=np.argmax(p.predictions, axis=1), references=p.label_ids
        )

    # config = AutoConfig.from_pretrained(
    #     model_args.config_name or model_args.model_name_or_path,
    #     # num_labels=len(labels),
    #     # label2id=label2id,
    #     # id2label=id2label,
    #     finetuning_task="image-classification",
    #     cache_dir=model_args.cache_dir,
    #     revision=model_args.model_revision,
    #     use_auth_token=True if model_args.use_auth_token else None,
    # )
    model = ViTForImageClassification.from_pretrained(model_args.model_name_or_path)

    feature_extractor = ViTFeatureExtractor.from_pretrained(
        model_args.model_name_or_path
    )

    # Define torchvision transforms to be applied to each image.
    normalize = Normalize(
        mean=feature_extractor.image_mean, std=feature_extractor.image_std
    )
    _train_transforms = Compose(
        [
            RandomResizedCrop(feature_extractor.size),
            RandomHorizontalFlip(),
            ToTensor(),
            normalize,
        ]
    )
    _val_transforms = Compose(
        [
            Resize(feature_extractor.size),
            CenterCrop(feature_extractor.size),
            ToTensor(),
            normalize,
        ]
    )

    def train_transforms(example_batch):
        """Apply _train_transforms across a batch."""
        example_batch["pixel_values"] = [
            _train_transforms(pil_img.convert("RGB"))
            for pil_img in example_batch["image"]
        ]
        return example_batch

    def val_transforms(example_batch):
        """Apply _val_transforms across a batch."""
        example_batch["pixel_values"] = [
            _val_transforms(pil_img.convert("RGB"))
            for pil_img in example_batch["image"]
        ]
        return example_batch

    valid_dataset = build_dataset(
        "/tmp/imagenet/", "train.zip@/", "dev_map.txt", _val_transforms
    )
    test_dataset = build_dataset(
        "/tmp/imagenet/", "val.zip@/", "val_map.txt", _val_transforms
    )

    dataset = {"validation": valid_dataset, "test": test_dataset}

    # Initalize our trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=None,
        eval_dataset=dataset["validation"] if training_args.do_eval else None,
        compute_metrics=compute_metrics,
        tokenizer=feature_extractor,
        data_collator=collate_fn,
    )
    return trainer, dataset, training_args

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        metrics = trainer.evaluate(dataset["test"])
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)


def prunning_head():
    trainer, dataset, training_args = main()
    pruner = ViTHeadPruner(
        trainer,
        "beam_search_union",
        dataset["validation"],
        dataset["test"],
        {},
        {
            "layer_num": trainer.model.config.num_hidden_layers,
            "head_num": trainer.model.config.num_attention_heads,
            "metrics_name": "eval_accuracy",
        },
    )
    pruner.beam_search_union(
        {
            "heads": [
                [1, 2, 3, 5, 6, 7, 8, 10, 11],
                [2, 4, 5, 7, 8, 9],
                [4, 5],
                [1, 3, 4, 11],
                [2, 3, 8],
                [0, 3, 4, 8, 9, 11],
                [2, 3, 4, 5, 7, 8, 10],
                [0, 1, 3, 7, 9],
                [0, 1, 2, 6, 7, 10, 11],
                [0, 3, 4, 5, 6, 7, 8, 10],
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
                [1, 10],
            ],
            "mask_idx": 32,
            "local_rank": training_args.local_rank,
        }
    )


if __name__ == "__main__":
    prunning_head()
