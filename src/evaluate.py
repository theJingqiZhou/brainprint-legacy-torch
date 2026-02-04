import time

import numpy as np
import onnxruntime
import torch
from tqdm import tqdm

from src.data import base_dataset
from src.metrics import Evaluators
from src.models import HydraNet


class Test:
    def __init__(self, cfg):
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        # load data
        self.test_loader = base_dataset(cfg, "test")
        self.suffix = cfg["test"]["model_path"].split(".")[-1]

        if self.suffix == "pt":
            self.model = HydraNet(cfg)
            self.model.load_state_dict(
                torch.load(cfg["test"]["model_path"], map_location=self.device)
            )
            self.model = self.model.to(self.device)
            self.model.eval()

        elif self.suffix == "onnx":
            providers = ["CPUExecutionProvider"]
            if torch.cuda.is_available():
                providers.insert(0, "CUDAExecutionProvider")
            self.onnx_session = onnxruntime.InferenceSession(
                cfg["test"]["model_path"],
                providers=providers,
            )
        # init evaluator
        self.evaluator = Evaluators()

    def test_onnx(
        self,
    ):
        times = []
        preds = []
        targets = []
        for data, target in tqdm(
            self.test_loader,
            desc="Test (onnx)",
            leave=False,
        ):
            data = data.to(self.device)
            start_time = time.time()
            output, _ = self.onnx_session.run(None, {"input": data.cpu().numpy()})
            end_time = time.time()
            times.append(end_time - start_time)
            output = np.array(output)
            pred = output.argmax(axis=1)

            preds.append(pred)
            targets.append(target)

        preds = np.concatenate(preds, axis=0)
        targets = torch.cat(targets, dim=0)
        targets = targets.numpy()

        acc = self.evaluator.accuracy(targets, preds)
        print("accuracy: ", acc)
        precision = self.evaluator.precision(targets, preds)
        print("precision: ", precision)
        recall = self.evaluator.recall(targets, preds)
        print("recall: ", recall)
        print("infer time: ", sum(times), np.mean(times))

    def test_pth(
        self,
    ):
        times = []
        preds = []
        targets = []

        with torch.no_grad():
            for data, target in tqdm(
                self.test_loader,
                desc="Test (pth)",
                leave=False,
            ):
                data, target = data.to(self.device), target.to(self.device)
                start_time = time.time()
                output, _ = self.model(data)
                end_time = time.time()
                times.append(end_time - start_time)
                pred = output.argmax(dim=1, keepdim=True)

                preds.append(pred)
                targets.append(target)
            preds = torch.cat(preds, dim=0)
            targets = torch.cat(targets, dim=0)

            preds = preds.cpu().numpy()
            targets = targets.cpu().numpy()

            acc = self.evaluator.accuracy(targets, preds)
            print("accuracy: ", acc)
            precision = self.evaluator.precision(targets, preds)
            print("precision: ", precision)
            recall = self.evaluator.recall(targets, preds)
            print("recall: ", recall)

            print("infer time: ", sum(times), np.mean(times))

    def run(
        self,
    ):
        if self.suffix == "pt":
            self.test_pth()
        elif self.suffix == "onnx":
            self.test_onnx()
