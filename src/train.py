import os
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from tqdm import tqdm

from src.data import base_dataset
from src.metrics import Evaluators
from src.models import HydraNet


class BaseTrainer:
    def __init__(self, cfg):
        self.cfg = cfg
        current_time = datetime.now()
        formatted_date = current_time.strftime("%Y-%m-%d")
        formatted_datetime = formatted_date + "_" + current_time.strftime("%H-%M")
        self.workspace = os.path.join(
            self.cfg["general"]["workspace"], formatted_datetime
        )
        if not os.path.exists(self.workspace):
            os.makedirs(self.workspace)

        self.max_epoch = self.cfg["general"]["epoch"]
        self.evaluator = Evaluators()
        self.net = HydraNet(self.cfg)

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        self.net = self.net.to(self.device)

        criterion = self.cfg["train"]["criterion"]
        if criterion == "nll":
            self.loss_func = nn.NLLLoss()
        elif criterion == "ce":
            self.loss_func = nn.CrossEntropyLoss()
        else:
            raise ValueError(f"Unknown criterion: {criterion}")

        self.criterion = criterion

        params = [
            {"params": self.net.parameters(), "lr": self.cfg["train"]["initial_lr"]}
        ]
        optim_params = self.cfg["optim"]
        optimizer_name = self.cfg["train"]["optimizer"]
        if optimizer_name == "adam":
            self.optimizer = optim.Adam(params, **optim_params)
        elif optimizer_name == "sgd":
            self.optimizer = optim.SGD(params, **optim_params)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")

        lr_strategy = self.cfg["train"]["lr_strategy"]
        if lr_strategy == "cosine":
            self.scheduler = lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=100)
        elif lr_strategy == "none":
            self.scheduler = None
        else:
            raise ValueError(f"Unknown lr_strategy: {lr_strategy}")

        self.train_loader = base_dataset(self.cfg, "train")
        self.val_loader = base_dataset(self.cfg, "val")

    def train_epoch(self, epoch):
        self.net.train()
        train_map = []
        for inputs, labels in tqdm(
            self.train_loader,
            desc=f"Train Epoch {epoch}",
            leave=False,
        ):
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            self.optimizer.zero_grad()
            output, _ = self.net(inputs)
            if self.criterion == "nll":
                loss = self.loss_func(torch.log(output), labels.long())
            else:
                loss = self.loss_func(output, labels.long())
            loss.backward()
            self.optimizer.step()
            pred = output.argmax(dim=1, keepdim=True)
            acc = self.evaluator.accuracy(labels.cpu(), pred.cpu())
            train_map.append(acc)

        if self.scheduler is not None:
            self.scheduler.step()

        print(
            "Train Epoch: {}\ttrain_Loss: {:.6f} mAP:{:.4f}".format(
                epoch, loss.item(), np.mean(train_map)
            )
        )

    def val(self, epoch):
        self.net.eval()
        test_loss = 0
        test_map = []
        with torch.no_grad():
            for inputs, labels in tqdm(
                self.val_loader,
                desc=f"Val Epoch {epoch}",
                leave=False,
            ):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                labels = labels.long()
                output, _ = self.net(inputs)

                if self.criterion == "nll":
                    test_loss += self.loss_func(torch.log(output), labels)
                else:
                    test_loss += self.loss_func(output, labels)
                pred = output.argmax(dim=1, keepdim=True)
                acc = self.evaluator.accuracy(labels.cpu(), pred.cpu())
                test_map.append(acc)
        test_loss /= len(self.val_loader.dataset)

        print(
            "Val set: Average loss: {:.4f}, mAP: {:.4f}".format(
                test_loss.item(), np.mean(test_map)
            )
        )
        return np.round(np.mean(test_map), 4)

    def train(
        self,
    ):
        for epoch in range(1, self.max_epoch + 1):
            self.train_epoch(epoch)
            val_map = self.val(epoch)
            if os.path.exists(self.cfg["general"]["workspace"]) == False:
                os.makedirs(self.cfg["general"]["workspace"])
            save_epoch_path = os.path.join(
                self.workspace, str(epoch) + "-" + str(val_map) + ".pt"
            )
            torch.save(self.net.state_dict(), save_epoch_path)
            latest_path = os.path.join(self.workspace, "latest.pt")
            torch.save(self.net.state_dict(), latest_path)
