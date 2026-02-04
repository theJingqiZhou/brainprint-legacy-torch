import numpy as np
import scipy.io as sio

from src.config import CONFIG_DEFAULT, CONFIG_MAT
from src.inference import Inference

if __name__ == "__main__":
    PROFILE = "mat"  # "default" | "mat"
    if PROFILE == "default":
        cfg = CONFIG_DEFAULT
    elif PROFILE == "mat":
        cfg = CONFIG_MAT
    else:
        raise ValueError(f"Unknown profile: {PROFILE}")

    inference = Inference(cfg)

    lines = [
        "data2/test_data/real_time_data/2024-07-23-10-56-11.mat",
        "data2/test_data/real_time_data/2024-07-23-10-56-23.mat",
        "data2/test_data/real_time_data/2024-07-23-10-56-33.mat",
        "data2/test_data/real_time_data/2024-07-23-10-56-43.mat",
        "data2/test_data/real_time_data/2024-07-23-10-57-01.mat",
        "data2/test_data/real_time_data/2024-07-23-10-57-14.mat",
        "data2/test_data/real_time_data/2024-07-23-10-57-38.mat",
        "data2/test_data/real_time_data/2024-07-23-10-58-04.mat",
        "data2/test_data/real_time_data/2024-07-23-10-58-16.mat",
        "data2/test_data/real_time_data/2024-07-23-10-58-31.mat",
        "data2/test_data/real_time_data/2024-07-23-10-58-47.mat",
        "data2/test_data/real_time_data/2024-07-23-10-58-59.mat",
        "data2/test_data/real_time_data/2024-07-23-10-59-09.mat",
    ]
    for line in lines:
        if line.endswith("npy"):
            data = np.load(line)
        elif line.endswith("mat"):
            data = sio.loadmat(line)["data"]
        identity_map = inference.infer(data)
        print(identity_map)
