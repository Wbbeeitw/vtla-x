"""Tiny smoke dataset in the EXACT physical-intelligence/libero schema (v2.0),
written with the container's lerobot 0.1.0 so the format is guaranteed
compatible, then verified via openpi's own create_torch_dataset.
"""
import numpy as np
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

REPO = "smoke_libero_pi"
FEATURES = {
    "image": {"dtype": "image", "shape": (256, 256, 3), "names": ["height", "width", "channel"]},
    "wrist_image": {"dtype": "image", "shape": (256, 256, 3), "names": ["height", "width", "channel"]},
    "state": {"dtype": "float32", "shape": (8,), "names": ["state"]},
    "actions": {"dtype": "float32", "shape": (7,), "names": ["actions"]},
}

ds = LeRobotDataset.create(REPO, fps=10, features=FEATURES, use_videos=False)
rng = np.random.default_rng(0)
for ep in range(5):
    for t in range(20):
        ds.add_frame({
            "image": rng.integers(0, 255, (256, 256, 3), dtype=np.uint8),
            "wrist_image": rng.integers(0, 255, (256, 256, 3), dtype=np.uint8),
            "state": np.zeros(8, np.float32),
            "actions": (rng.standard_normal(7) * 0.01).astype(np.float32),
            "task": "smoke wine rack",
        })
    ds.save_episode()
print("dataset built:", REPO, flush=True)

# acceptance: openpi's own loader must read it
import sys
sys.path.insert(0, "/workspace/RLinf")
from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config
import openpi.training.data_loader as openpi_data_loader

cfg = get_openpi_config("pi05_libero", repo_id=REPO)
data_config = cfg.data.create(cfg.assets_dirs, cfg.model)
dataset = openpi_data_loader.create_torch_dataset(
    data_config, cfg.model.action_horizon, cfg.model
)
sample = dataset[0]
print("openpi loader OK | sample keys:", sorted(sample.keys()), flush=True)
