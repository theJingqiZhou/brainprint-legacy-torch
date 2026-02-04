import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from src.deploy import Deploy
from src.runtime_config import CONFIG_DEFAULT, CONFIG_MAT
from src.test import Test
from src.trainer import BaseTrainer

PROFILE = "default"  # "default" | "mat"
RUNNER = "train"  # "train" | "test" | "deploy"


if PROFILE == "default":
    cfg = CONFIG_DEFAULT
elif PROFILE == "mat":
    cfg = CONFIG_MAT
else:
    raise ValueError(f"Unknown profile: {PROFILE}")


if RUNNER == "train":
    trainer = BaseTrainer(cfg)
    trainer.train()
elif RUNNER == "test":
    test = Test(cfg)
    test.run()
elif RUNNER == "deploy":
    deploy = Deploy(cfg)
    deploy.run()
else:
    raise ValueError(f"Unknown runner: {RUNNER}")
