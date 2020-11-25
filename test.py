import argparse
import os
import time
import re

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import Trainer

from utils import get_offsets_mapping, read_from_file, encode_tags, pack
from datasets import PosDataset
from models import BertCRFForTokenClassification, XLMRobertaCRFForTokenClassification


def evaluate(test_file, checkpoint_dir, notes):
    """Evaluation with the trained model and test data."""

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    if checkpoint_dir.endswith("/"):
        checkpoint_dir = checkpoint_dir[:-1]
    base_dir = os.path.dirname(checkpoint_dir)

    model_name = re.compile(r"model=(.+?)&").search(base_dir).group(1).replace("#", "/")
    add_crf = True if re.compile(f"&CRF&").search(base_dir) is not None else False

    with open(os.path.join(base_dir, "id2tag.txt")) as f:
        id2tag_dict = eval(f.read().strip())
    tag2id_dict = {v: k for k, v in id2tag_dict.items()}

    # Prepares for data.
    test_srcs, test_trgs = read_from_file(test_file)

    # Creates a tokenizer.
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print(tokenizer, "\n")
    test_encodings = tokenizer(
        test_srcs,
        padding=True, truncation=True,
        is_split_into_words=True
    )

    # Solves the mismatch btw tokens and labels due to tokenization.
    test_offsets_mapping = get_offsets_mapping(test_srcs[:], test_encodings, tokenizer)
    test_mask, test_trgs_ = encode_tags(test_trgs, tag2id_dict, test_offsets_mapping)

    # Creates a model.
    model = AutoModelForTokenClassification.from_pretrained(checkpoint_dir, return_dict=True).to(device)
    if add_crf:
        print("Switching to the crf model")

        test_encodings["pos_mask"] = test_mask

        if model.__class__.__name__ == "BertForTokenClassification":
            model = BertCRFForTokenClassification.from_pretrained(checkpoint_dir, return_dict=True).to(device)
        elif model.__class__.__name__ == "XLMRobertaForTokenClassification":
            model = XLMRobertaCRFForTokenClassification.from_pretrained(checkpoint_dir, return_dict=True).to(device)
        else:
            raise NotImplementedError
    print(model, "\n")

    # Creates a dataset.
    test_dataset = PosDataset(test_encodings, test_trgs_)

    # Evaluation.
    if not add_crf:
        trainer = Trainer(model=model)
        output = trainer.predict(test_dataset)

        outs_out = np.argmax(output.predictions, axis=2).tolist()

        outs = [[] for _ in range(len(outs_out))]
        for seq_i in range(len(outs_out)):
            for tag_i in range(len(outs_out[seq_i])):
                if test_mask[seq_i][tag_i]:
                    outs[seq_i].append(id2tag_dict[outs_out[seq_i][tag_i]])
    else:
        outs = []

        loader = DataLoader(
            dataset=test_dataset,
            batch_size=32,
            shuffle=False,
        )

        model.eval()
        for i, batch in enumerate(loader):
            predictions = model(
                input_ids=batch["input_ids"].to(device) if "input_ids" in batch.keys() else None,
                token_type_ids=batch["token_type_ids"].to(device) if "token_type_ids" in batch.keys() else None,
                attention_mask=batch["attention_mask"].to(device) if "attention_mask" in batch.keys() else None,
                pos_mask=batch["pos_mask"].to(device) if "pos_mask" in batch.keys() else None
            )
            predictions = pack(predictions)

            for prediction in predictions:
                outs.append([id2tag_dict[id] for id in prediction])

    trgs = test_trgs
    count_right = 0
    count_all = 0
    with open(os.path.join(checkpoint_dir, f"results&test={os.path.basename(test_file)}{f'&notes={notes}' if notes else ''}&time={fmt_time}.txt"), "w") as f:
        for trg_seq, out_seq in zip(trgs, outs):
            for trg, out in zip(trg_seq, out_seq):
                print(f"{trg}\t{out}")
                f.write(f"{trg}\t{out}\n")

                if trg == out:
                    count_right += 1
                count_all += 1

            print()
            f.write("\n")

    # bert, epoch=3: 94.86
    # bert-crf, epoch=3: 94.96
    # bert-crf, epoch=5: 94.53
    with open(os.path.join(checkpoint_dir, f"accuracy&test={os.path.basename(test_file)}{f'&notes={notes}' if notes else ''}&time={fmt_time}.txt"), "w") as f:
        accuracy = count_right / count_all

        print(f"accuracy: {accuracy:.2%}")
        f.write(f"accuracy: {accuracy:.2%}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", required=True, help="Test set")
    parser.add_argument("--ckpt-dir", required=True, help="Checkpoint directory of the model to be tested")
    parser.add_argument("--notes", default="")
    args = parser.parse_args()

    fmt_time = time.strftime("%m_%d_%H_%M")

    evaluate(
        test_file=args.test,
        checkpoint_dir=args.ckpt_dir,
        notes=args.notes
    )
