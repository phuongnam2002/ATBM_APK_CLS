import os
import torch
import wandb
import argparse
from transformers import set_seed
from trainers.trainer import APKTrainer
from components.dataset.dataset import APKDataset
from utils.utils import MODEL_CLASSES, MODEL_PATH_MAP, load_tokenizer, logger


def main(args):
    logger.info("Args={}".format(str(args)))

    set_seed(args.seed)

    # Pre Setup
    args.device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f'Running on {args.device}')

    # Load tokenizer and model
    tokenizer = load_tokenizer(args)
    config_class, model_class, _ = MODEL_CLASSES[args.model_type]

    model_config = config_class.from_pretrained(args.model_name_or_path)
    model = model_class.from_pretrained(
        args.model_name_or_path,
        torch_dtype=args.compute_dtype,
        config=model_config,
        device_map=args.device,
        args=args,
    )

    logger.info(model)
    logger.info(model.dtype)
    logger.info("Vocab size: {}".format(len(tokenizer)))

    # Load data
    train_dataset = APKDataset(args, tokenizer, "train")

    trainer = APKTrainer(
        args=args,
        model=model,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
    )

    wandb.init(
        project=args.wandb_project, name=args.wandb_run_name, config=vars(args)
    )

    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_dir",
        default=None,
        required=True,
        type=str,
        help="Path to save, load model",
    )
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        help="The input data dir",
    )
    parser.add_argument(
        "--model_type",
        default="roberta-base",
        type=str,
        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()),
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="namdp",
        help="Name of the Weight and Bias project.",
    )
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default="test-source",
        help="Name of the run for Weight and Bias.",
    )
    parser.add_argument(
        "--wandb_watch",
        type=str,
        default="false",
        help="Whether to enable tracking of gradients and model topology in Weight and Bias.",
    )
    parser.add_argument(
        "--wandb_log_model",
        type=str,
        default="false",
        help="Whether to enable model versioning in Weight and Bias.",
    )

    # Hyperparameters for training
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for initialization.",
    )
    parser.add_argument(
        "--num_train_epochs",
        default=2.0,
        type=float,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--train_batch_size",
        default=32,
        type=int,
        help="Batch size used for training.",
    )
    parser.add_argument(
        "--dataloader_drop_last",
        type=bool,
        default=True,
        help="Toggle whether to drop the last incomplete batch in the dataloader.",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help="Number of workers for the dataloader.",
    )
    parser.add_argument(
        "--dataloader_pin_memory",
        type=bool,
        default=True,
        help="Toggle whether to use pinned memory in the dataloader.",
    )

    # Tokenizer Configuration
    parser.add_argument(
        "--max_seq_len",
        default=64,
        type=int,
        help="The maximum total input sequence length for input after tokenization.",
    )
    parser.add_argument(
        "--use_fast_tokenizer",
        default=False,
        type=bool,
        help="Whether to use the fast tokenizer. If set to True, a faster tokenizer will be used for tokenizing the input data. This can improve the performance of tokenization but may sacrifice some tokenization quality. If set to False, a slower but more accurate tokenizer will be used. Default value is True.",
    )

    # Optimizer Configuration
    parser.add_argument(
        "--learning_rate",
        default=1e-5,
        type=float,
        help="The initial learning rate for Adam.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        default="cosine",
        type=str,
        help="Type of learning rate scheduler to use. Available options are: 'cosine', 'step', 'plateau'. "
             "The default is 'cosine', which uses a cosine annealing schedule. ",
    )
    parser.add_argument(
        "--weight_decay", default=0.0, type=float, help="Weight decay if we apply some."
    )
    parser.add_argument(
        "--max_grad_norm", default=1.0, type=float, help="Max gradient norm."
    )
    parser.add_argument(
        "--warmup_steps", default=100, type=int, help="Linear warmup over warmup_steps."
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument(
        "--early_stopping",
        type=int,
        default=50,
        help="Number of unincreased validation step to wait for early stopping",
    )

    # Model Configuration
    parser.add_argument(
        "--compute_dtype",
        type=torch.dtype,
        default=torch.float,
        help="Used in quantization configs. Do not specify this argument manually.",
    )

    args = parser.parse_args()
    args.model_name_or_path = MODEL_PATH_MAP[args.model_type]
    # Check if parameter passed or if set within environ
    args.use_wandb = len(args.wandb_project) > 0 or (
            "WANDB_PROJECT" in os.environ and len(os.environ["WANDB_PROJECT"]) > 0
    )
    # Only overwrite environ if wandb param passed
    if len(args.wandb_project) > 0:
        os.environ["WANDB_PROJECT"] = args.wandb_project
    if len(args.wandb_watch) > 0:
        os.environ["WANDB_WATCH"] = args.wandb_watch
    if len(args.wandb_log_model) > 0:
        os.environ["WANDB_LOG_MODEL"] = args.wandb_log_model

    main(args)
