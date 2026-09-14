"""Waypoint-scripted expert for the LIBERO wine-rack task
(put_the_wine_bottle_on_the_rack, libero_goal).

Serves OSC_POSE deltas toward ground-truth waypoints (sim gives free object
poses) with a fixed state machine. Acceptance bar: >=90% success over 30
random inits before any data collection.

Tuning knobs live in ExpertConfig; the acceptance loop
(scripts/collect_wine_rack.py --check) sweeps random inits to verify.
"""
import time

import numpy as np


class ExpertConfig:
    max_steps: int = 320
    k_pos: float = 8.0            # proportional gain on position servo
    max_delta: float = 0.08       # per-step pos delta clip (m)
    grasp_z_off: float = 0.20     # grasp the bottle's UPPER body: the arm
                                  # cannot descend near its base, and a
                                  # high carry hangs the bottle into the slot
    pregrasp_z_off: float = 0.26
    lift_dx: float = 0.0
    lift_dz: float = 0.05
    rack_above_dz: float = 0.15
    rack_place_dz: float = 0.02
    close_steps: int = 15
    release_steps: int = 10
    state_timeout: int = 80
    comp_max: float = 0.15       # max 3-dim sag/load compensation (norm)
    slow_factor: float = 0.5      # speed scale during rack descent


class WineRackExpert:
    """States: pregrasp -> approach -> close -> lift -> above_rack -> place -> release -> done."""

    def __init__(self, env, cfg: ExpertConfig | None = None):
        self.env = env
        self.cfg = cfg or ExpertConfig()
        sim = env.sim
        m = sim.model
        self.bottle_id = m.body_name2id("wine_bottle_1_main")
        self.rack_id = m.site_name2id("wine_rack_1_top_region")
        self.eef_id = m.body_name2id("gripper0_eef")
        self.comp = np.zeros(3)  # eef BODY (controller frame origin); pos via body_xpos
        self.t0 = time.time()

    # ---- ground truth -------------------------------------------------
    def _bottle_pos(self, sim):
        return sim.data.body_xpos[self.bottle_id].copy()

    def _rack_pos(self, sim):
        return sim.data.site_xpos[self.rack_id].copy()

    def _eef_pos(self, sim):
        return sim.data.body_xpos[self.eef_id].copy()

    # ---- core servo ---------------------------------------------------
    def _servo_action(self, sim, target, gripper, scale=1.0):
        cur = self._eef_pos(sim)
        delta = np.clip(self.cfg.k_pos * (target - cur) * scale,
                        -self.cfg.max_delta, self.cfg.max_delta)
        return np.concatenate([delta, np.zeros(3), [gripper]]).astype(np.float32)

    # ---- episode driver -------------------------------------------------
    def rollout(self, verbose=False, init_state=None):
        """Run one episode; returns (success, frames) where frames is a list of
        per-step dicts {action, tactile60, wrist_ft, image, wrist_image, state}.
        The caller (collector) owns recording and LeRobot writing."""
        env, cfg = self.env, self.cfg
        from rlinf.envs.sim.libero.tactile_patch import TactileReader
        reader = TactileReader(env.sim)

        obs = env.reset()
        self.comp = np.zeros(3)  # per-episode reset: sag compensation must not leak
        if init_state is not None:
            sim = env.sim
            nq, nv = sim.model.nq, sim.model.nv
            sim.data.qpos[:] = init_state[1:1 + nq]
            sim.data.qvel[:] = init_state[1 + nq:1 + nq + nv]
            sim.forward()
            obs, _, _, _ = env.step(np.zeros(7))
        frames = []
        state, t_in_state, gripper = "pregrasp", 0, 1.0
        grasp_pos = None

        for t in range(cfg.max_steps):
            sim = env.sim
            bottle = self._bottle_pos(sim)
            rack = self._rack_pos(sim)

            if state == "pregrasp":
                target = bottle + np.array([0, 0, cfg.pregrasp_z_off])
                gripper = 1.0
                if np.linalg.norm(self._eef_pos(sim) - target) < 0.02:
                    state, t_in_state = "approach", 0
            elif state == "approach":
                target = bottle + np.array([0, 0, cfg.grasp_z_off])
                gripper = 1.0
                if np.linalg.norm(self._eef_pos(sim) - target) < 0.012:
                    state, t_in_state = "close", 0
                    grasp_pos = self._eef_pos(sim).copy()
            elif state == "close":
                target = grasp_pos
                gripper = -1.0
                if t_in_state >= cfg.close_steps:
                    state, t_in_state = "lift", 0
            elif state == "lift":
                target = grasp_pos + np.array([cfg.lift_dx, 0, cfg.lift_dz])
                gripper = -1.0
                if np.linalg.norm(self._eef_pos(sim) - target) < 0.02:
                    state, t_in_state = "above_rack", 0
            elif state == "above_rack":
                target = rack + np.array([0, 0, cfg.rack_above_dz])
                gripper = -1.0
                if np.linalg.norm(self._eef_pos(sim) - target) < 0.02:
                    state, t_in_state = "place", 0
            elif state == "place":
                target = rack + np.array([0, 0, cfg.rack_place_dz])
                gripper = -1.0
                if np.linalg.norm(self._eef_pos(sim) - target) < 0.015:
                    state, t_in_state = "release", 0
            elif state == "release":
                target = self._eef_pos(sim)
                gripper = 1.0
                if t_in_state >= cfg.release_steps:
                    state, t_in_state = "retreat", 0
            elif state == "retreat":
                target = self._eef_pos(sim) + np.array([0, 0, 0.10])
                gripper = 1.0

            target = target.copy() + self.comp
            action = self._servo_action(sim, target, gripper,
                                        scale=cfg.slow_factor if state == "place" else 1.0)
            obs, _, _, _ = env.step(action)
            t_in_state += 1

            frames.append({
                "action": action.copy(),
                "tactile60": reader.tactile(sim).copy(),
                "wrist_ft": reader.wrist_ft(sim).copy(),
                "image": obs["agentview_image"].copy(),
                "wrist_image": obs["robot0_eye_in_hand_image"].copy(),
                "state": np.concatenate([
                    obs["robot0_eef_pos"],
                    obs["robot0_eef_quat"],
                    obs["robot0_gripper_qpos"][:1],
                ]).astype(np.float32),
                "state_name": state,
            })

            if verbose and t % 40 == 0:
                print(f"  t={t} state={state}", flush=True)

            check = getattr(env, "check_success", None) or env._check_success
            if check():
                return True, frames
            if t_in_state >= cfg.state_timeout:
                cur = self._eef_pos(sim)
                shortfall = target - cur
                if np.linalg.norm(shortfall) > 0.01 and np.linalg.norm(self.comp) < cfg.comp_max:
                    self.comp = self.comp + shortfall * 0.8
                    if np.linalg.norm(self.comp) > cfg.comp_max:
                        self.comp = self.comp / np.linalg.norm(self.comp) * cfg.comp_max
                    t_in_state = 0
                    print(f"  [comp {np.round(self.comp,3)}] state={state}", flush=True)
                else:
                    print(f"  [timeout {state}] eef={np.round(cur,3)} target="
                          f"{np.round(target,3)} dist={np.linalg.norm(cur - target):.3f} "
                          f"bottle={np.round(self._bottle_pos(sim),3)}", flush=True)
                    return False, frames

        return False, frames
