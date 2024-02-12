from typing import Optional

import torch
import torch.nn as nn
from transformers import PretrainedConfig
from transformers.models.roberta.modeling_roberta import (
    RobertaModel,
    RobertaPreTrainedModel,
)

from components.models.module import MLPLayer


class APKModel(RobertaPreTrainedModel):
    _keys_to_ignore_on_save = [r"lm_head.decoder.weight", r"lm_head.decoder.bias"]
    _keys_to_ignore_on_load_missing = [
        r"position_ids",
        r"lm_head.decoder.weight",
        r"lm_head.decoder.bias",
    ]

    def __init__(self, config: PretrainedConfig, args: None):
        super().__init__(config)
        self.args = args
        self.config = config

        self.mlp = MLPLayer(config)
        self.activation = nn.Sigmoid()
        self.loss = nn.BCEWithLogitsLoss()

        self.roberta = RobertaModel(config, add_pooling_layer=True)

        # The LM head weights require special treatment only when they are tied with the word embeddings
        self.update_keys_to_ignore(config, ["lm_head.decoder.weight"])

        # Initialize weights and apply final processing
        self.post_init()

    def get_output(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            **kwargs
    ):
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
        ).pooler_output

        outputs = self.mlp(outputs)
        outputs = self.activation(outputs)

        return outputs

    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.LongTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            **kwargs
    ):
        outputs = self.get_output(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        if not kwargs.get('is_train', ''):
            return outputs

        outputs = torch.gather(outputs, 1, labels)

        loss = self.loss(outputs, labels.float())

        return loss
