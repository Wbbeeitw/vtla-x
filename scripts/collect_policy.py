"""PPO policy-driven demo collector for the wine-rack task with tactile channels.

Uses the SAME model wrapper and env class as the proven RLinf eval path:
  get_model(rollout.model)  ->  wrapped pi05 with predict_action_batch(env_obs)
  LiberoEnv(cfg.env.eval)   ->  obs assembly (main_images/wrist_images/states/
                                task_descriptions), init-state pool, success checks
plus per-step tactile recording via tactile_patch.TactileReader.

Usage:
  python scripts/collect_policy.py --n-demos 2     # trial
  python scripts/collect_policy.py --n-demos 50    # collection
Writes LeRobot dataset `wine_rack_tactile` under HF_LEROBOT_HOME
(=/data_vtlax/datasets in the vtla_x container).
"""
import argparse
import os
import sys

sys.path.insert(0, "/workspace/RLinf")
sys.path.insert(0, "/opt/venv/openpi/libero")
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from rlinf.envs.sim.libero.tactile_patch import apply_touch_patch, TactileReader

apply_touch_patch()  # must precede env construction

from rlinf.models.embodiment.openpi import get_model  # noqa: E402
from rlinf.envs.sim.libero.libero_env import LiberoEnv  # noqa: E402
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402

PROMPT = "put the wine bottle on the wine rack"
GRASP_TOUCH_MIN = 1.0
PLACE_TOUCH_MIN = 0.5


def tactile_event_ok(frames):
    grasp_max = max(
        (f["tactile"].max() for f in frames if f["state_name"] in ("close", "lift")),
        default=0.0,
    )
    place_max = max(
        (f["tactile"].max() for f in frames if f["state_name"] == "place"),
        default=0.0,
    )
    return grasp_max >= GRASP_TOUCH_MIN and place_max >= PLACE_TOUCH_MIN


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-demos", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str, default="wine_rack_tactile")
    ap.add_argument("--config-dir", type=str,
                    default="/workspace/RLinf/evaluations/libero")
    ap.add_argument("--config-name", type=str,
                    default="vtla_triage_goal_openpi_pi05_eval")
    args = ap.parse_args()

    with initialize_config_dir(config_dir=args.config_dir, version_base=None):
        cfg = compose(config_name=args.config_name)
    model_cfg = OmegaConf.merge(cfg.rollout.model, OmegaConf.create({
        "model_path": "/data_vtlax/checkpoints/RLinf-Pi05-PPO-LIBERO-130",
    }))

    # single-env config for collection
    env_cfg = OmegaConf.merge(cfg.env.eval, OmegaConf.create({
        "total_num_envs": 1,
        "group_size": 1,
        "task_id_filter": [9],
        "use_fixed_reset_state_ids": False,  # diverse inits for demos
    }))

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    env = LiberoEnv(env_cfg, num_envs=1, seed_offset=0,
                    total_num_processes=1, worker_info=None)
    model = get_model(model_cfg)
    model.eval()
    reader = TactileReader(env.sim)

    ds = LeRobotDataset.create(args.out, fps=20, features={
        "image": {"dtype": "video", "shape": (256, 256, 3),
                  "names": ["height", "width", "channel"]},
        "wrist_image": {"dtype": "video", "shape": (256, 256, 3),
                        "names": ["height", "width", "channel"]},
        "state": {"dtype": "float32", "shape": (8,), "names": ["state"]},
        "action": {"dtype": "float32", "shape": (7,), "names": ["action"]},
        "observation.tactile": {"dtype": "float32", "shape": (60,),
                                "names": ["tactile"]},
        "observation.wrist_ft": {"dtype": "float32", "shape": (6,),
                                 "names": ["ft"]},
    })

    n_saved, n_fail, ep = 0, 0, 0
    while n_saved < args.n_demos and ep < args.n_demos * 6:
        obs = env.reset()
        frames = []
        steps = 0
        done = False
        while not done and steps < 320:
            actions, _ = model.predict_action_batch(env_obs=obs, mode="eval")
            chunk = actions.detach().cpu().numpy()
            if chunk.ndim == 3:
                chunk = chunk[0]
            for a in chunk:
                frames.append({
                    "image": obs["main_images"][0]
                        if isinstance(obs["main_images"], list)
                        else obs["main_images"],
                    "wrist_image": obs["wrist_images"][0]
                        if isinstance(obs["wrist_images"], list)
                        else obs["wrist_images"],
                    "state": obs["states"][0] if len(obs["states"].shape) > 1
                        else obs["states"],
                    "action": a.copy(),
                    "observation.tactile": reader.tactile(env.sim).copy(),
                    "observation.wrist_ft": reader.wrist_ft(env.sim).copy(),
                    "task": PROMPT,
                })
                obs, _, _, _ = env.step(a)
                steps += 1
                if env._check_success():
                    done = True
                    break
        ok = done and steps < 320  # success terminates early; truncation = fail

        if ok and tactile_event_ok(frames):
            n_saved += 1
            for f in frames:
                ds.add_frame(dict(f, task=PROMPT))
            ds.save_episode()
            print(f"[{n_saved}/{args.n_demos}] saved ({len(frames)} steps)", flush=True)
        else:
            n_fail += 1
            print(f"episode discarded (ok={ok}, {len(frames)} steps)", flush=True)
        ep += 1

    print(f"DONE: saved={n_saved} failed={n_fail}", flush=True)


if __name__ == "__main__":
    main()
