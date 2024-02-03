import os
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from utils.utils import logger
from utils.io import load_jsonl
from components.dataset.utils import convert_text_to_features


class APKDataset(Dataset):
    def __init__(self, args, tokenizer: PreTrainedTokenizer, mode: str = "train"):
        super().__init__()
        self.args = args
        self.tokenizer = tokenizer

        # Reading data
        file_path = os.path.join(
            self.args.data_dir, mode, "data.jsonl"
        )
        logger.info("LOOKING AT {}".format(file_path))

        self.data = load_jsonl(file_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        datapoint = self.data[index]

        text = datapoint['text']
        label = datapoint['label']

        input_ids, attention_mask = convert_text_to_features(
            text=text,
            tokenizer=self.tokenizer,
            max_seq_len=self.args.max_seq_len,
        )

        return (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(attention_mask, dtype=torch.long),
            torch.tensor([label], dtype=torch.long),
        )
