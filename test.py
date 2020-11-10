import argparse
import os

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
from train import get_offsets_mapping, read_from_file


def evaluate(test_file, tokenizer_name, checkpoint_dir):
    """Evaluation with the trained model and test data."""

    if checkpoint_dir.endswith("/"):
        checkpoint_dir = checkpoint_dir[:-1]
    base_dir = os.path.dirname(checkpoint_dir)

    with open(os.path.join(base_dir, "id2tag.txt")) as f:
        id2tag_dict = eval(f.read().strip())

    # Prepares for data.
    test_srcs, test_trgs = read_from_file(test_file)

    # Creates a tokenizer.
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # Creates a model.
    model = AutoModelForTokenClassification.from_pretrained(checkpoint_dir, return_dict=True)

    # Evaluation.
    count_wrong = 0
    count = 0
    with open(os.path.join(base_dir, "results.txt"), "w") as f:
        # TODO: Do with whole data when gpu resources are available
        for i in range(len(test_srcs)):
            # Tokenization.
            tokens = tokenizer(
                test_srcs[i],
                padding=True, truncation=True,
                is_split_into_words=True,
                return_tensors="pt"
            )

            # Gets predictions.
            outputs = model(**tokens).logits
            predictions = torch.argmax(outputs, dim=2)

            # Solves the unmatch btw tokens and labels due to tokenization.
            offsets_mapping = get_offsets_mapping([test_srcs[i]], tokens, tokenizer)[0]
            # TODO: Merge with encode_tags
            doc_enc_labels = np.ones(len(offsets_mapping), dtype=int) * -100
            assert len(doc_enc_labels) == len(predictions.tolist()[0])
            for j in range(len(offsets_mapping)):
                if offsets_mapping[j][0] == 0 and offsets_mapping[j][1] != 0:
                    doc_enc_labels[j] = predictions.tolist()[0][j]

            # Writes results to file.
            for token, trg in zip(test_srcs[i], test_trgs[i]):
                # Ignores special tokens and those starting with ##.
                while doc_enc_labels[0] == -100:
                    doc_enc_labels = doc_enc_labels[1:]

                out = id2tag_dict[doc_enc_labels[0]]

                if trg != out:
                    count_wrong += 1
                count += 1

                print(f"{token}\t{out}")
                f.write(f"{token}\t{out}\n")

                doc_enc_labels = doc_enc_labels[1:]
            print()
            f.write("\n")

    with open(os.path.join(base_dir, "accuracy.txt"), "w") as f:
        accuracy = 1 - (count_wrong / count)

        print(f"accuracy: {accuracy:.2%}")
        f.write(f"accuracy: {accuracy:.2%}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", required=True, help="Test set")
    parser.add_argument("--checkpoint-dir", required=True, help="Checkpoint directory of the model to be tested")
    args = parser.parse_args()

    tokenizer_name = "cahya/bert-base-indonesian-1.5G"

    evaluate(
        test_file=args.test,
        tokenizer_name=tokenizer_name,
        checkpoint_dir=args.checkpoint_dir,
    )
