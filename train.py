import re
import argparse
from typing import List, Set, Dict, Tuple

import numpy as np
import torch
import time
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import Trainer, TrainingArguments


def read_from_file(file_path) -> (List[List[str]], List[List[str]]):
    """Read data from file.

    :return: token_docs: List of word lists (i.e. sentences).
             tag_docs: List of tag lists.
    """

    with open(file_path) as f:
        # TODO: Support unk
        raw_text = f.read().strip().replace("'", '"').replace("|", '"').replace("é", "e")
    raw_docs = re.split(r"\n\t?\n", raw_text)

    token_docs = []
    tag_docs = []
    for doc in raw_docs:
        tokens = []
        tags = []

        for line in doc.split("\n"):
            token, tag = line.split("\t")

            tokens.append(token)
            tags.append(tag)

        token_docs.append(tokens)
        tag_docs.append(tags)

    return token_docs, tag_docs


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


def get_offsets_mapping(srcs, encodings, tokenizer) -> List[List[Tuple[int, int]]]:
    """Get the position of each token.

    :param srcs: List of token lists (i.e. sentences).
    :param encodings: Encoding of `srcs`.
    :param tokenizer: Tokenizer for the model.
    :return: Position of each token in each sentence (token list).

    The position of a token is in the form of (word_idx, token_idx). word_idx is
    the idx of the word consisting the token, and token_idx is the idx of the token
    in the word. For example, if the original word is "@huggingface" and the tokens are
    @, hugging, ##face, the positions are (0, 1), (1, 8), (8, 12).
    """

    input_ids = encodings["input_ids"]
    # "sentence" here refers to a list of tokens.
    sentence_list = [tokenizer.convert_ids_to_tokens(input_ids_) for input_ids_ in input_ids]

    offset_mapping = []
    for i in range(len(sentence_list)):
        offset_mapping.append([])
        end = 0
        current_string = ""
        for token in sentence_list[i]:
            # TODO: Support unk
            if token in ["[CLS]", "[SEP]", "[PAD]"]:
                offset_mapping[-1].append((0, 0))
                continue

            if token.startswith("##"):
                token = token[2:]

            token_len = len(token)
            offset_mapping[-1].append((end, end + token_len))
            end += token_len
            current_string += token

            if current_string == srcs[i][0].lower():
                end = 0
                current_string = ""
                srcs[i] = srcs[i][1:]

    return offset_mapping


def encode_tags(tags, tag2id, offsets_mapping):
    """Mask labels of special tokens and those starting with ##.

    :param tags: List of tag lists.
    :param tag2id: Mapping: tag -> id.
    :param offsets_mapping: Position of each token in each sentence (token list)
    :return: List of label lists whose lengths are the same as the corresponding token lists.
    """

    # TODO: Support param for tag id lists.
    labels = [[tag2id[tag] for tag in doc] for doc in tags]
    encoded_labels = []

    for doc_labels, doc_offset in zip(labels, offsets_mapping):
        # create an empty array of -100
        doc_enc_labels = np.ones(len(doc_offset), dtype=int) * -100
        arr_offset = np.array(doc_offset)

        # set labels whose first offset position is 0 and the second is not 0
        doc_enc_labels[(arr_offset[:, 0] == 0) & (arr_offset[:, 1] != 0)] = doc_labels
        encoded_labels.append(doc_enc_labels.tolist())

    return encoded_labels


class WNUTDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def train(model_name, train_file, valid_file, output_dir, logging_dir):
    """Train a model."""
    # Prepares for data.
    (
        train_srcs, valid_srcs, train_trgs, valid_trgs,
        unique_tags, tag2id_dict, id2tag_dict
    ) = preprocess(train_file, valid_file)

    # Tokenization.
    tokenizer = AutoTokenizer.from_pretrained(model_name)
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
    train_trgs_ = encode_tags(train_trgs, tag2id_dict, train_offsets_mapping)
    valid_trgs_ = encode_tags(valid_trgs, tag2id_dict, valid_offsets_mapping)

    # Creates datasets.
    train_dataset = WNUTDataset(train_encodings, train_trgs_)
    valid_dataset = WNUTDataset(valid_encodings, valid_trgs_)

    # Creates a model.
    model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=len(unique_tags))

    # Training.
    training_args = TrainingArguments(
        output_dir=output_dir,  # output directory
        num_train_epochs=3,  # total number of training epochs
        per_device_train_batch_size=16,  # batch size per device during training
        per_device_eval_batch_size=64,  # batch size for evaluation
        warmup_steps=500,  # number of warmup steps for learning rate scheduler
        weight_decay=0.01,  # strength of weight decay
        logging_dir=logging_dir,  # directory for storing logs
        logging_steps=10,
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
    args = parser.parse_args()

    model_name = "cahya/bert-base-indonesian-1.5G"
    output_dir = f'./ind_results_{time.strftime("%m_%d_%H_%M")}'
    logging_dir = f'./logs_{time.strftime("%m_%d_%H_%M")}'

    train(
        model_name=model_name,
        train_file=args.train,
        valid_file=args.valid,
        output_dir=output_dir,
        logging_dir=logging_dir
    )
