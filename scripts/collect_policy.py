"""PPO policy-driven demo collector for the wine-rack task with tactile channels.

Uses the raw robosuite env (OffScreenRenderEnv) directly + the RLinf model
wrapper (get_model -> predict_action_batch), replicating rlinf's libero_env
obs assembly (state = eef_pos + axisangle(eef_quat) + gripper_qpos = 8d).

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
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from rlinf.envs.sim.libero.tactile_patch import apply_touch_patch, TactileReader

apply_touch_patch()  # must precede env construction

from libero.libero import benchmark  # noqa: E402
from libero.libero.envs import OffScreenRenderEnv  # noqa: E402
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402
from robosuite.utils.transform_utils import quat2axisangle  # noqa: E402

from rlinf.models.embodiment.openpi import get_model  # noqa: E402

PROMPT = "put the wine bottle on the wine rack"
GRASP_TOUCH_MIN = 1.0
PLACE_TOUCH_MIN = 0.5


def make_env(bddl):
    return OffScreenRenderEnv(
        bddl_file_name=bddl,
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        camera_names=["agentview", "robot0_eye_in_hand"],
        camera_heights=256,
        camera_widths=256,
    )


def build_model(config_dir, config_name):
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name=config_name)
    model_cfg = OmegaConf.merge(cfg.rollout.model, OmegaConf.create({
        "model_path": "/data_vtlax/checkpoints/RLinf-Pi05-PPO-LIBERO-130",
    }))
    model = get_model(model_cfg)
    model.eval()
    return model


def to_env_obs(obs, prompt):
    state = np.concatenate([
        obs["robot0_eef_pos"],
        quat2axisangle(obs["robot0_eef_quat"]),
        obs["robot0_gripper_qpos"],
    ]).astype(np.float32)
    return {
        "main_images": [obs["agentview_image"]],
        "wrist_images": [obs["robot0_eye_in_hand_image"]],
        "states": state[None],
        "actions": np.zeros((1, 7), np.float32),  # placeholder; replaced by flow noise
        "task_descriptions": [prompt],
        "extra_view_images": None,
    }


def check_success(env):
    check = getattr(env, "check_success", None) or env._check_success
    return check()


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

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    model = build_model(args.config_dir, args.config_name)

    bench_cls = benchmark.get_benchmark_dict()["libero_goal"]
    bench = bench_cls(task_order_index=0)
    bddl = bench.get_task_bddl_file_path(0)
    init_states = bench.get_task_init_states(0)

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
        env = make_env(bddl)
        reader = TactileReader(env.sim)
        env.seed(args.seed + ep)
        obs = env.reset()

        frames = []
        steps = 0
        done = False
        while not done and steps < 320:
            env_obs = to_env_obs(obs, PROMPT)
            with torch.no_grad():
                actions, _ = model.predict_action_batch(env_obs=env_obs,
                                                        mode="eval")
            chunk = actions.detach().cpu().numpy()
            if chunk.ndim == 3:
                chunk = chunk[0]
            for a in chunk:
                env_obs = to_env_obs(obs, PROMPT)
                tactile = reader.tactile(env.sim)
                ft = reader.wrist_ft(env.sim)
                frames.append({
                    "image": obs["agentview_image"].copy(),
                    "wrist_image": obs["robot0_eye_in_hand_image"].copy(),
                    "state": env_obs["states"][0],
                    "action": a.copy(),
                    "observation.tactile": tactile.copy(),
                    "observation.wrist_ft": ft.copy(),
                    "task": PROMPT,
                })
                obs, _, _, _ = env.step(a)
                steps += 1
                if check_success(env):
                    done = True
                    break
        env.close()

        ok = done and steps < 320  # early termination = success
        if ok and tactile_event_ok(frames):
            n_saved += 1
            for f in frames:
                ds.add_frame(f)
            ds.save_episode()
            print(f"[{n_saved}/{args.n_demos}] saved ({len(frames)} steps)",
                  flush=True)
        else:
            n_fail += 1
            print(f"episode discarded (ok={ok}, {len(frames)} steps)", flush=True)
        ep += 1

    print(f"DONE: saved={n_saved} failed={n_fail}", flush=True)


if __name__ == "__main__":
    main()
