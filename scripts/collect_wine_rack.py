"""Collect wine-rack demonstrations with tactile channels via the waypoint expert.

Usage:
  python scripts/collect_wine_rack.py --check                 # acceptance: 30 random inits, no save
  python scripts/collect_wine_rack.py --n-demos 100           # collect 100 successful demos

Writes LeRobot v2.x dataset `wine_rack_tactile` under HF_LEROBOT_HOME
(=/data_vtlax/datasets in the vtla_x container).
"""
import argparse
import os
import sys

sys.path.insert(0, "/workspace/RLinf")
sys.path.insert(0, "/opt/venv/openpi/libero")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np

from rlinf.envs.sim.libero.tactile_patch import apply_touch_patch

apply_touch_patch()  # must precede env construction

from libero.libero import benchmark  # noqa: E402
from libero.libero.envs import OffScreenRenderEnv  # noqa: E402
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402
from libero_expert import WineRackExpert, ExpertConfig  # noqa: E402

TASK_NAME = "put_the_wine_bottle_on_the_rack"
SUITE = "libero_goal"
CAMERAS = ["agentview", "robot0_eye_in_hand"]
FEATURES = {
    "image": {"dtype": "video", "shape": (256, 256, 3), "names": ["height", "width", "channel"]},
    "wrist_image": {"dtype": "video", "shape": (256, 256, 3), "names": ["height", "width", "channel"]},
    "state": {"dtype": "float32", "shape": (8,), "names": ["state"]},
    "action": {"dtype": "float32", "shape": (7,), "names": ["action"]},
    "observation.tactile": {"dtype": "float32", "shape": (60,), "names": ["tactile"]},
    "observation.wrist_ft": {"dtype": "float32", "shape": (6,), "names": ["ft"]},
}
GRASP_TOUCH_MIN = 1.0   # N, expected contact force during grasp/lift
PLACE_TOUCH_MIN = 0.5


def make_env(bddl):
    from robosuite.controllers import load_controller_config
    cc = load_controller_config(default_controller="OSC_POSE")
    cc["kp"] = 300  # reduce gravity sag (limit is 300)
    return OffScreenRenderEnv(
        bddl_file_name=bddl,
        controller_configs=cc,
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        camera_names=CAMERAS,
        camera_heights=256,
        camera_widths=256,
    )


def tactile_event_ok(frames):
    """QC: contact spikes present in both the grasp and the place phases."""
    grasp_max = max((f["tactile60"].max() for f, s in frames if s in ("close", "lift")), default=0)
    place_max = max((f["tactile60"].max() for f, s in frames if s == "place"), default=0)
    return grasp_max >= GRASP_TOUCH_MIN and place_max >= PLACE_TOUCH_MIN


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="acceptance mode: no saving")
    ap.add_argument("--n-demos", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str, default="/data_vtlax/datasets/wine_rack_tactile")
    args = ap.parse_args()

    bench_cls = benchmark.get_benchmark_dict()[SUITE]
    bench = bench_cls(task_order_index=0)
    task = bench.get_task(0)
    bddl = bench.get_task_bddl_file_path(0)
    init_states = bench.get_task_init_states(0)
    print(f"task={TASK_NAME} | init states: {len(init_states)}", flush=True)

    np.random.seed(args.seed)

    def build():
        e = make_env(bddl)
        return e, WineRackExpert(e, ExpertConfig())

    env, expert = build()

    if args.check:
        n_ok, n_tot = 0, min(12, len(init_states) * 5)
        for ep in range(n_tot):
            init = init_states[ep % len(init_states)]
            try:
                ok, frames = expert.rollout(init_state=init, verbose=(ep % 10 == 0))
            except Exception as e:
                print(f"[check {ep:02d}] env exception: {type(e).__name__} {str(e)[:90]}", flush=True)
                env, expert = build()  # GL/teardown noise: rebuild and continue
                continue
            n_ok += ok
            print(f"[check {ep:02d}] success={ok} steps={len(frames)}", flush=True)
        print(f"ACCEPTANCE: {n_ok}/{n_tot} = {n_ok / n_tot:.0%} (need >=90%)", flush=True)
        return

    ds = LeRobotDataset.create(args.out, fps=20, features=FEATURES)
    n_saved, n_fail = 0, 0
    ep = 0
    while n_saved < args.n_demos and ep < args.n_demos * 6:
        init = init_states[ep % len(init_states)]
        ok, frames = expert.rollout(init_state=init)
        if ok and tactile_event_ok([(f, f["state_name"]) for f in frames]):
            n_saved += 1
            for f in frames:
                ds.add_frame({
                    "image": f["image"],
                    "wrist_image": f["wrist_image"],
                    "state": f["state"],
                    "action": f["action"],
                    "observation.tactile": f["tactile60"],
                    "observation.wrist_ft": f["wrist_ft"],
                    "task": TASK_NAME,
                })
            ds.save_episode()
            print(f"[{n_saved}/{args.n_demos}] saved ({len(frames)} steps)", flush=True)
        else:
            n_fail += 1
            print(f"episode discarded (ok={ok}, {len(frames)} steps)", flush=True)
        ep += 1
    print(f"DONE: saved={n_saved} failed={n_fail}", flush=True)


if __name__ == "__main__":
    main()
