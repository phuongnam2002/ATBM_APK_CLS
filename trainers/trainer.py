import torch
import wandb
from tqdm import tqdm
from typing import Optional
from lion_pytorch import Lion
from tqdm.auto import tqdm, trange
from torch.utils.data import DataLoader
from transformers.optimization import get_scheduler
from transformers.trainer_pt_utils import get_parameter_names
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from utils.utils import logger
from utils.early_stopping import EarlyStopping
from components.dataset.dataset import APKDataset


class APKTrainer:
    def __init__(
            self,
            args,
            model: Optional[torch.nn.Module],
            train_dataset: Optional[APKDataset] = None,
            tokenizer: Optional[PreTrainedTokenizerBase] = None,
    ):
        self.args = args

        self.model = model

        self.tokenizer = tokenizer
        self.train_dataset = train_dataset

    def train(self):
        train_dataloader = DataLoader(
            self.train_dataset,
            shuffle=True,
            batch_size=self.args.train_batch_size,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

        if self.args.max_steps > 0:
            t_total = self.args.max_steps
            self.args.num_train_epochs = (
                    self.args.max_steps // len(train_dataloader)
                    + 1
            )
        else:
            t_total = (len(train_dataloader) * self.args.num_train_epochs)

        # Prepare optimizer and schedule (linear warmup and decay)
        optimizer = self.get_optimizer()

        scheduler = get_scheduler(
            self.args.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=self.args.warmup_steps,
            num_training_steps=t_total,
        )

        # Train!
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(self.train_dataset))
        logger.info("  Num Epochs = %d", self.args.num_train_epochs)
        logger.info("  Total train batch size = %d", self.args.train_batch_size)
        logger.info("  Total optimization steps = %d", t_total)

        global_step = 0

        self.model.zero_grad()

        train_iterator = trange(int(self.args.num_train_epochs), desc="Epoch")
        early_stopping = EarlyStopping(patience=self.args.early_stopping, verbose=True)

        # Automatic Mixed Precision
        scaler = torch.cuda.amp.GradScaler()

        for _ in train_iterator:
            epoch_iterator = tqdm(
                train_dataloader, desc="Iteration", position=0, leave=True
            )
            logger.info(f"Epoch {_}")

            for step, batch in enumerate(epoch_iterator):
                self.model.train()
                batch = tuple(t.to(self.args.device) for t in batch)  # GPU or CPU

                inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "labels": batch[2],
                    "is_train": True
                }

                with torch.cuda.amp.autocast():
                    loss = self.model(**inputs)

                scaler.scale(loss).backward()
                wandb.log({"Train Loss": loss.item()})

                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.args.max_grad_norm
                )

                scaler.step(optimizer)
                scheduler.step()
                scaler.update()

                self.model.zero_grad()
                global_step += 1

                early_stopping(loss.item(), self.model, self.args)

                if early_stopping.early_stop:
                    logger.info('Early stopping')
                    break

                if 0 < self.args.max_steps < global_step:
                    epoch_iterator.close()
                    break

            if 0 < self.args.max_steps < global_step or early_stopping.early_stop:
                train_iterator.close()
                break
        return

    def get_optimizer(self):
        decay_parameters = get_parameter_names(self.model, [torch.nn.LayerNorm])
        decay_parameters = [name for name in decay_parameters if "bias" not in name]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in self.model.named_parameters() if n in decay_parameters
                ],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if n not in decay_parameters
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = Lion(
            optimizer_grouped_parameters,
            lr=self.args.learning_rate
        )
        return optimizer
