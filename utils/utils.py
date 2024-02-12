import os
import logging

from transformers import (
    AutoTokenizer,
    RobertaConfig
)
from components.models.model import APKModel

MODEL_CLASSES = {
    "roberta-base": (RobertaConfig, APKModel, AutoTokenizer),
    "roberta-large": (RobertaConfig, APKModel, AutoTokenizer),
    "malware-url": (RobertaConfig, APKModel, AutoTokenizer),
}

MODEL_PATH_MAP = {
    "roberta-base": "roberta-base",
    "roberta-large": "roberta-large",
    "malware-url": "elftsdmr/malware-url-detect"
}


def _setup_logger():
    log_format = logging.Formatter("[%(asctime)s %(levelname)s] %(message)s")
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)

    data_dir = './logs/'
    os.makedirs(data_dir, exist_ok=True)
    file_handler = logging.FileHandler('{}/log.txt'.format(data_dir))
    file_handler.setFormatter(log_format)

    logger.handlers = [console_handler, file_handler]

    return logger


logger = _setup_logger()


def load_tokenizer(args):
    return MODEL_CLASSES[args.model_type][2].from_pretrained(
        args.model_name_or_path,
        use_fast=args.use_fast_tokenizer
    )
