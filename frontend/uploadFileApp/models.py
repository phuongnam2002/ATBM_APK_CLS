import os
import uuid
import torch
import string
import torch.nn as nn
from typing import Optional
from django.db import models
from APKFileUpload import settings
from androguard.core.dex import DEX
from androguard.core.apk import APK
from transformers import PretrainedConfig
from transformers.models.roberta.modeling_roberta import (
    RobertaModel,
    RobertaPreTrainedModel,
)


def preprocessing(text: str):
    text = text.lower()
    text = text.strip(string.punctuation)

    return text


def load_apk(file_path: str):
    apk = APK(file_path)

    # Extract static data
    package = apk.get_package()
    receivers = apk.get_receivers()
    providers = apk.get_providers()
    activities = apk.get_activities()
    permissions = apk.get_permissions()

    # Extract code
    codes = []

    classes_dex = apk.get_dex()

    dvm = DEX(classes_dex)

    classes = dvm.get_classes()

    for x in classes:
        methods = dvm.get_methods_class(x)
        codes.extend(methods)

        break

    data = []

    data.extend(package)
    data.extend(activities)
    data.extend(permissions)
    data.extend(providers)
    data.extend(receivers)
    data.extend(codes)

    return " ".join(data)


def convert_text_to_features(
        text: str,
        tokenizer,
        max_seq_len: int = 128,
        special_tokens_count: int = 2,
):
    unk_token = tokenizer.unk_token

    cls_token = tokenizer.cls_token

    sep_token = tokenizer.sep_token

    pad_token_id = tokenizer.pad_token_id

    # Normalize text
    text = preprocessing(text)

    text = text.split()  # Some are spaced twice

    tokens = []
    # Tokenizer
    for word in text:
        word_tokens = tokenizer.tokenize(word)
        if not word_tokens:
            word_tokens = [unk_token]  # For handling the bad-encoded word
        tokens.extend(word_tokens)

    # Truncate data
    if len(tokens) > max_seq_len - special_tokens_count:
        tokens = tokens[: (max_seq_len - special_tokens_count)]

    # Add [SEP] token
    tokens += [sep_token]

    # Add [CLS] token
    tokens = [cls_token] + tokens

    # Convert tokens to ids
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    attention_mask = [1] * len(input_ids)

    # TODO use smart padding in here
    # Zero-pad up to the sequence length. This is static method padding
    padding_length = max_seq_len - len(input_ids)
    input_ids = input_ids + ([pad_token_id] * padding_length)
    attention_mask = attention_mask + ([0] * padding_length)

    assert len(input_ids) == max_seq_len, "Error with input length {} vs {}".format(
        len(input_ids), max_seq_len
    )

    return input_ids, attention_mask


def user_directory_path(instance, filename):
    ext = filename.split('.')[-1]
    filename = '{}.{}'.format(uuid.uuid4().hex[:10], ext)
    file_path = os.path.join(settings.FILE_UPLOAD_URL, filename)
    return file_path


# Create your models here.
class File(models.Model):
    file = models.FileField(upload_to=user_directory_path, null=True)


class MLPLayer(nn.Module):
    def __init__(self, config: PretrainedConfig):
        super().__init__()

        self.linear = nn.Linear(config.hidden_size, 2)

    def forward(self, x):
        x = self.linear(x)

        return x


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
    ):
        outputs = self.get_output(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        return outputs
