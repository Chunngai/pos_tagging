import re
from typing import List, Tuple

import numpy as np


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
            if token in [tokenizer.cls_token, tokenizer.sep_token, tokenizer.pad_token]:
                offset_mapping[-1].append((0, 0))
                continue

            if token.startswith("##"):
                token = token[2:]
            elif token.startswith("▁"):
                token = token[1:]

            token_len = len(token)
            offset_mapping[-1].append((end, end + token_len))
            end += token_len
            current_string += token

            if current_string == srcs[i][0].lower().replace(" ", ""):
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
    mask = []
    encoded_labels = []

    for doc_labels, doc_offset in zip(labels, offsets_mapping):
        # create an empty array of -100
        doc_enc_labels = np.ones(len(doc_offset), dtype=int) * -100
        arr_offset = np.array(doc_offset)

        # set labels whose first offset position is 0 and the second is not 0
        doc_enc_labels[(arr_offset[:, 0] == 0) & (arr_offset[:, 1] != 0)] = doc_labels
        encoded_labels.append(doc_enc_labels.tolist())

        mask_seq = []
        for offset in doc_offset:
            if offset[0] == 0 and offset[1] != 0:
                mask_seq.append(True)
            else:
                mask_seq.append(False)
        mask.append(mask_seq)

    return mask, encoded_labels
