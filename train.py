import argparse
import time
import os
from typing import List, Set, Dict

import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, EvaluationStrategy
from transformers import Trainer, TrainingArguments

from utils import read_from_file, get_offsets_mapping, encode_tags
from datasets import PosDataset
from models import BertCRFForTokenClassification, XLMRobertaCRFForTokenClassification


def preprocess(train_file, valid_file) -> (List[List[str]], List[List[str]], List[List[str]], List[List[str]],
                                           Set[str], Dict[str, int], Dict[int, str]):
    """Prepare data.

    :param train_file: Path of the train set.
    :param valid_file: Path of the valid set.
    :return: train_srcs: List of token lists (i.e. sentences) for training.
             valid_srcs: List of token lists (i.e. sentences) for validation.
             train_trgs: List of label lists for training.
             valid_trgs: List of label lists for validation.
             unique_tags: Label set.
             tag2id_dict: Mapping: tag -> tag id.
             id2tag_dict: Mapping: tag id -> tag.
    """

    train_srcs, train_trgs = read_from_file(train_file)
    valid_srcs, valid_trgs = read_from_file(valid_file)

    unique_tags = set(tag for tag_doc in (train_trgs + valid_trgs) for tag in tag_doc)
    tag2id_dict = {tag: id for id, tag in enumerate(unique_tags)}
    id2tag_dict = {id: tag for tag, id in tag2id_dict.items()}

    return (
        train_srcs, valid_srcs, train_trgs, valid_trgs,
        unique_tags, tag2id_dict, id2tag_dict
    )


def train(model_name, add_crf, train_file, valid_file, output_dir, logging_dir, epoch):
    """Train a model."""

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Prepares for data.
    (
        train_srcs, valid_srcs, train_trgs, valid_trgs,
        unique_tags, tag2id_dict, id2tag_dict
    ) = preprocess(train_file, valid_file)

    # Tokenization.
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print(tokenizer, "\n")
    train_encodings = tokenizer(
        train_srcs,
        padding=True, truncation=True,
        is_split_into_words=True
    )
    valid_encodings = tokenizer(
        valid_srcs,
        padding=True, truncation=True,
        is_split_into_words=True
    )

    # Solves the mismatch btw tokens and labels due to tokenization.
    train_offsets_mapping = get_offsets_mapping(train_srcs[:], train_encodings, tokenizer)
    valid_offsets_mapping = get_offsets_mapping(valid_srcs[:], valid_encodings, tokenizer)
    train_mask, train_trgs_ = encode_tags(train_trgs, tag2id_dict, train_offsets_mapping)
    valid_mask, valid_trgs_ = encode_tags(valid_trgs, tag2id_dict, valid_offsets_mapping)

    # Creates a model.
    model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=len(unique_tags))
    if add_crf:
        print("Switching to the crf model")

        train_encodings["pos_mask"] = train_mask
        valid_encodings["pos_mask"] = valid_mask

        if model.__class__.__name__ == "BertForTokenClassification":
            model = BertCRFForTokenClassification.from_pretrained(model_name, num_labels=len(unique_tags)).to(device)
        elif model.__class__.__name__ == "XLMRobertaForTokenClassification":
            model = XLMRobertaCRFForTokenClassification.from_pretrained(model_name, num_labels=len(unique_tags)).to(device)
        else:
            raise NotImplementedError
    print(model, "\n")

    # Creates datasets.
    train_dataset = PosDataset(train_encodings, train_trgs_)
    valid_dataset = PosDataset(valid_encodings, valid_trgs_)

    # Training.
    training_args = TrainingArguments(
        output_dir=output_dir,  # output directory
        num_train_epochs=epoch,  # total number of training epochs
        per_device_train_batch_size=16,  # batch size per device during training
        per_device_eval_batch_size=64,  # batch size for evaluation
        warmup_steps=500,  # number of warmup steps for learning rate scheduler
        weight_decay=0.01,  # strength of weight decay
        logging_dir=logging_dir,  # directory for storing logs
        logging_steps=10,

        # Eval steps. Will save a ckpt every `eval_steps`.
        evaluation_strategy=EvaluationStrategy.STEPS,
        eval_steps=2_000,

        # Eval epochs. Will save a ckpt every `num_train_epochs`.
        # evaluation_strategy=EvaluationStrategy.EPOCH,

        save_total_limit=10,
        metric_for_best_model="loss",
        load_best_model_at_end=True,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset
    )
    trainer.train()

    # Saves id2tag dict for inferring.
    with open(f"{output_dir}/id2tag.txt", "w") as f:
        f.write(str(id2tag_dict))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True, help="Train set")
    parser.add_argument("--valid", required=True, help="Valid set")
    parser.add_argument("--model", required=True)
    parser.add_argument("--crf", action="store_true", default=False)
    parser.add_argument("--epoch", default=3, type=int)
    parser.add_argument("--notes", default="")
    args = parser.parse_args()

    add_crf = args.crf

    fmt_time = time.strftime("%mm%dd%HH%MM")
    output_dir = f"results&model={args.model}{f'&CRF' if add_crf else ''}&epoch={args.epoch}&train={os.path.basename(args.train)}&valid={os.path.basename(args.valid)}{f'&notes={args.notes}' if args.notes else ''}&time={fmt_time}".replace(
        '/', '#')
    logging_dir = f"logs&model={args.model}{f'&CRF' if add_crf else ''}&epoch={args.epoch}&train={os.path.basename(args.train)}&valid={os.path.basename(args.valid)}&notes={f'&notes={args.notes}' if args.notes else ''}&time={fmt_time}".replace(
        '/', '#')

    train(
        model_name=args.model,
        add_crf=add_crf,
        train_file=args.train,
        valid_file=args.valid,
        output_dir=output_dir,
        logging_dir=logging_dir,
        epoch=args.epoch
    )
