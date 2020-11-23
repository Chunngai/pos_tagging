import torch
import torch.nn as nn
from transformers import BertForTokenClassification, BertModel, RobertaForTokenClassification, RobertaModel
from torchcrf import CRF


class BertCRFForTokenClassification(BertForTokenClassification):
    """BertForTokenClassification with a CRF layer on top."""

    def __init__(self, config):
        super(BertCRFForTokenClassification, self).__init__(config)

        # Bert for token classification.
        self.bert = BertModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # CRF.
        self.crf = CRF(num_tags=config.num_labels, batch_first=True)

    def forward(
            self,
            input_ids=None,
            token_type_ids=None,
            attention_mask=None,
            pos_mask=None,
            labels=None
    ):
        # Outputs from the bert.
        outputs = self.bert(input_ids, token_type_ids, attention_mask)
        sequence_output = outputs[0]

        # Dropout.
        sequence_output = self.dropout(sequence_output)

        # Linear transformation from bert_hidden_size to num_labels.
        logits = self.classifier(sequence_output)

        emissions = logits

        # Removes [CLS].
        emissions = emissions[:, 1:, :]
        mask = pos_mask[:, 1:]

        mask_clone = mask.clone().detach()

        # Moves -100 to the end.
        batch_size, seq_len, _ = emissions.size()
        for batch_idx in range(batch_size):
            emission_i = emissions[batch_idx]
            mask_i = mask[batch_idx]

            token_idx = 0
            while mask[batch_idx][token_idx:].any():
                if not mask_i[token_idx]:
                    emissions[batch_idx] = torch.cat([
                        emission_i[:token_idx],
                        emission_i[token_idx + 1:],
                        emission_i[token_idx:token_idx + 1]]
                    )
                    mask[batch_idx] = torch.cat([
                        mask_i[:token_idx],
                        mask_i[token_idx + 1:],
                        mask_i[token_idx:token_idx + 1]])
                else:
                    token_idx += 1

        if labels is not None:
            # Removes [CLS].
            labels = labels[:, 1:]

            # Moves -100 to the end.
            for batch_idx in range(batch_size):
                labels_i = labels[batch_idx]
                mask_clone_i = mask_clone[batch_idx]

                token_idx = 0
                while mask_clone[batch_idx][token_idx:].any():
                    if not mask_clone_i[token_idx]:
                        labels[batch_idx] = torch.cat([
                            labels_i[:token_idx],
                            labels_i[token_idx + 1:],
                            labels_i[token_idx:token_idx + 1]]
                        )
                        mask_clone[batch_idx] = torch.cat([
                            mask_clone_i[:token_idx],
                            mask_clone_i[token_idx + 1:],
                            mask_clone_i[token_idx:token_idx + 1]]
                        )
                    else:
                        token_idx += 1

            # -100 -> 0.
            zeros = torch.zeros_like(labels)
            labels = torch.where(labels == -100, zeros, labels)

            log_likelihood = self.crf(emissions, labels, mask=mask, reduction="mean")
            sequence_of_tags = self.crf.decode(emissions, mask=mask)

            return -log_likelihood, sequence_of_tags
        else:
            sequence_of_tags = self.crf.decode(emissions, mask=mask)
            return sequence_of_tags


class RobertaCRFForTokenClassification(RobertaForTokenClassification):
    """RobertaForTokenClassification with a CRF layer on top."""

    def __init__(self, config):
        super(RobertaCRFForTokenClassification, self).__init__(config)

        # Roberta for token classification.
        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # CRF.
        self.crf = CRF(num_tags=config.num_labels, batch_first=True)

    def forward(
            self,
            input_ids=None,
            token_type_ids=None,
            attention_mask=None,
            pos_mask=None,
            labels=None
    ):
        # Outputs from the bert.
        outputs = self.roberta(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask
        )
        sequence_output = outputs[0]

        # Dropout.
        sequence_output = self.dropout(sequence_output)

        # Linear transformation from roberta_hidden_size to num_labels.
        logits = self.classifier(sequence_output)

        emissions = logits

        # Removes [CLS].
        emissions = emissions[:, 1:, :]
        mask = pos_mask[:, 1:]

        mask_clone = mask.clone().detach()

        # Moves -100 to the end.
        batch_size, seq_len, _ = emissions.size()
        for batch_idx in range(batch_size):
            emission_i = emissions[batch_idx]
            mask_i = mask[batch_idx]

            token_idx = 0
            while mask[batch_idx][token_idx:].any():
                if not mask_i[token_idx]:
                    emissions[batch_idx] = torch.cat([
                        emission_i[:token_idx],
                        emission_i[token_idx + 1:],
                        emission_i[token_idx:token_idx + 1]]
                    )
                    mask[batch_idx] = torch.cat([
                        mask_i[:token_idx],
                        mask_i[token_idx + 1:],
                        mask_i[token_idx:token_idx + 1]])
                else:
                    token_idx += 1

        if labels is not None:
            # Removes [CLS].
            labels = labels[:, 1:]

            # Moves -100 to the end.
            for batch_idx in range(batch_size):
                labels_i = labels[batch_idx]
                mask_clone_i = mask_clone[batch_idx]

                token_idx = 0
                while mask_clone[batch_idx][token_idx:].any():
                    if not mask_clone_i[token_idx]:
                        labels[batch_idx] = torch.cat([
                            labels_i[:token_idx],
                            labels_i[token_idx + 1:],
                            labels_i[token_idx:token_idx + 1]]
                        )
                        mask_clone[batch_idx] = torch.cat([
                            mask_clone_i[:token_idx],
                            mask_clone_i[token_idx + 1:],
                            mask_clone_i[token_idx:token_idx + 1]]
                        )
                    else:
                        token_idx += 1

            # -100 -> 0.
            zeros = torch.zeros_like(labels)
            labels = torch.where(labels == -100, zeros, labels)

            log_likelihood = self.crf(emissions, labels, mask=mask, reduction="mean")
            sequence_of_tags = self.crf.decode(emissions, mask=mask)

            return -log_likelihood, sequence_of_tags
        else:
            sequence_of_tags = self.crf.decode(emissions, mask=mask)
            return sequence_of_tags
