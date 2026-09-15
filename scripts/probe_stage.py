import sys, os
sys.path.insert(0, "/workspace/RLinf")
sys.path.insert(0, "/opt/venv/openpi/libero")
sys.path.insert(0, "/workspace/RLinf/scripts")
os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np, torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
import openpi.transforms as transforms
from rlinf.envs.sim.libero.tactile_patch import apply_touch_patch, TactileReader
apply_touch_patch()
from rlinf.models.embodiment.openpi import get_model
from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config
from collect_policy import make_env, to_env_obs
from libero.libero import benchmark
with initialize_config_dir(config_dir="/workspace/RLinf/evaluations/libero", version_base=None):
    cfg = compose(config_name="vtla_triage_goal_openpi_pi05_eval")
model_cfg = OmegaConf.merge(cfg.rollout.model, OmegaConf.create({
    "model_path": "/data_vtlax/checkpoints/RLinf-Pi05-PPO-LIBERO-130"}))
model = get_model(model_cfg)
model.eval()
tc = get_openpi_config("pi05_libero", model_path=model_cfg.model_path)
bench = benchmark.get_benchmark_dict()["libero_goal"](task_order_index=0)
env = make_env(bench.get_task_bddl_file_path(0))
reader = TactileReader(env.sim)
obs = env.reset()
env_obs = to_env_obs(obs, "put the wine bottle on the wine rack")
data = model.obs_processor(env_obs)

def shapes(d):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out[k] = {kk: np.shape(vv) for kk, vv in v.items()}
        elif isinstance(v, (list, tuple)):
            out[k] = [np.shape(x) for x in v]
        else:
            out[k] = np.shape(v)
    return out

print("stage obs_processor:", shapes(data), flush=True)
data_config = tc.data.create(tc.assets_dirs, tc.model)
stages = [
    ("repack", [*data_config.repack_transforms.inputs]),
    ("data_transforms", [*data_config.data_transforms.inputs]),
    ("model_transforms", [*data_config.model_transforms.inputs]),
]
for name, ts in stages:
    for t in ts:
        tname = type(t).__name__
        try:
            data = t(data)
            print(f"after {tname}:", shapes(data), flush=True)
        except Exception as e:
            print(f"FAIL at {tname}: {type(e).__name__} {str(e)[:80]}", flush=True)
            if "image" in data:
                print("  data[image]:", {k: np.shape(v) for k, v in data["image"].items()}, flush=True)
            raise SystemExit
