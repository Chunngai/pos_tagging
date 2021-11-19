import argparse

from plot_conf_mx import plot_confusion_matrix


def plot(id2tag_file, rst_file, normalize, title, precise, save):
    # Gets id-tag mapping.
    with open(id2tag_file) as f:
        id2tag_dict = eval(f.read())
    id2tag_dict = dict(sorted(id2tag_dict.items(), key=lambda item: item[1]))
    tag2id_dict = {v: k for k, v in id2tag_dict.items()}
    tags = list(tag2id_dict)

    tag_len = len(tags)

    print(f"id2tag: {id2tag_dict}")
    print(f"tag2id: {tag2id_dict}")
    print(f"tags: {tags}")

    # Generates conf mx.
    conf_mx = [[0 for _ in range(tag_len)] for _ in range(tag_len)]

    trgs = []
    outs = []
    with open(rst_file) as f:
        lines = f.readlines()
        for line in lines:
            if line == "\n":
                continue

            trg, out = line.strip().split("\t")
            trgs.append(trg)
            outs.append(out)

            conf_mx[tag2id_dict[trg]][tag2id_dict[out]] += 1

    # Plots the matrix.
    plot_confusion_matrix(
        y_true=trgs,
        y_pred=outs,
        labels=tags,
        normalize=normalize,
        title=title,
        precise=precise,
        save=save
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--id2tag-file", required=True)
    parser.add_argument("--result-file", required=True)
    parser.add_argument("--normalize", action="store_true", default=False)
    parser.add_argument("--title", type=str, default="")
    parser.add_argument("--save", action="store_true", default=False)
    parser.add_argument("--precise", action="store_true", default=False)
    args = parser.parse_args()

    plot(
        id2tag_file=args.id2tag_file,
        rst_file=args.result_file,
        normalize=args.normalize,
        title=args.title,
        precise=args.precise,
        save=args.save,
    )

