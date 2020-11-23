import re
import argparse
import os

from sklearn.model_selection import train_test_split


def split_data(data_file, valid_ratio, test_ratio, folds):
    """Split data into train : valid : test."""

    def _read_from_file():
        with open(data_file) as f:
            raw_text = f.read().strip()
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

    def _write(file, srcs, trgs):
        with open(file, "w") as f:
            assert len(srcs) == len(trgs)
            for i in range(len(srcs)):
                for token, tag in zip(srcs[i], trgs[i]):
                    f.write(f"{token}\t{tag}\n")
                f.write("\n")

    token_docs, tag_docs = _read_from_file()

    span = int(len(token_docs) / folds)
    for i in range(folds):
        print(f"fold {i + 1}: ")

        train_valid_srcs, test_srcs, train_valid_trgs, test_trgs = train_test_split(token_docs, tag_docs, test_size=0.1,
                                                                                    shuffle=False)
        train_srcs, valid_srcs, train_trgs, valid_trgs = train_test_split(train_valid_srcs, train_valid_trgs,
                                                                          test_size=(valid_ratio) / (1 - test_ratio),
                                                                          shuffle=False)

        assert len(train_srcs) == len(train_trgs)
        assert len(valid_srcs) == len(valid_trgs)
        assert len(test_srcs) == len(test_trgs)

        print(f"train: {len(train_srcs)}")
        print(f"valid: {len(valid_srcs)}")
        print(f"test: {len(test_srcs)}")

        base_name = os.path.splitext(data_file)[0]
        fold_ext = f'.{i + 1}' if folds > 1 else ''
        _write(f"{base_name}.train{fold_ext}", train_srcs, train_trgs)
        _write(f"{base_name}.valid{fold_ext}", valid_srcs, valid_trgs)
        _write(f"{base_name}.test{fold_ext}", test_srcs, test_trgs)

        token_docs = token_docs[span:] + token_docs[:span]
        tag_docs = tag_docs[span:] + tag_docs[:span]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-file", required=True, help="Data file to split")
    parser.add_argument("--valid_ratio", default=0.1, type=float, help="Ratio of the valid set")
    parser.add_argument("--test_ratio", default=0.1, type=float, help="Ratio of the test set")
    parser.add_argument("--folds", default=1, type=int)
    args = parser.parse_args()

    split_data(
        data_file=args.data_file,
        valid_ratio=args.valid_ratio,
        test_ratio=args.test_ratio,
        folds=args.folds
    )
