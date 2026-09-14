"""Waypoint-scripted expert for the LIBERO wine-rack task
(put_the_wine_bottle_on_the_rack, libero_goal).

Serves OSC_POSE deltas toward ground-truth waypoints (sim gives free object
poses) with a fixed state machine + load-sag compensation.

Key behaviors:
- High grasp (bottle upper body): the arm cannot descend near its base.
- 3-dim load-sag compensation accumulates observed shortfalls (resets per
  episode).
- above_rack proceeds to descend after a bounded wait (no perfect-alignment
  requirement); place releases unconditionally after a bounded descent.
- Acceptance bar: >=90% success over 30 random inits before any data collection.
"""
import time

import numpy as np


class ExpertConfig:
    max_steps: int = 600
    k_pos: float = 8.0            # proportional gain on position servo
    max_delta: float = 0.08       # per-step pos delta clip (m)
    grasp_z_off: float = 0.15     # grasp the bottle's UPPER body: the arm
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
    comp_max: float = 0.15        # max 3-dim sag/load compensation (norm)
    place_steps: int = 150        # bounded descent before unconditional release
    slow_factor: float = 0.5      # speed scale during rack descent


class WineRackExpert:
    def __init__(self, env, cfg: ExpertConfig | None = None):
        self.env = env
        self.cfg = cfg or ExpertConfig()
        sim = env.sim
        m = sim.model
        self.bottle_id = m.body_name2id("wine_bottle_1_main")
        self.rack_id = m.site_name2id("wine_rack_1_top_region")
        self.eef_id = m.body_name2id("gripper0_eef")  # eef BODY (controller frame)
        self.comp = np.zeros(3)

    def _bottle_pos(self, sim):
        return sim.data.body_xpos[self.bottle_id].copy()

    def _rack_pos(self, sim):
        return sim.data.site_xpos[self.rack_id].copy()

    def _eef_pos(self, sim):
        return sim.data.body_xpos[self.eef_id].copy()

    def _servo_action(self, sim, target, gripper, scale=1.0):
        cur = self._eef_pos(sim)
        delta = np.clip(self.cfg.k_pos * (target - cur) * scale,
                        -self.cfg.max_delta, self.cfg.max_delta)
        return np.concatenate([delta, np.zeros(3), [gripper]]).astype(np.float32)

    def _check(self, env):
        check = getattr(env, "check_success", None) or env._check_success
        return check()

    def rollout(self, verbose=False, init_state=None):
        """One episode; returns (success, frames). frames carry per-step
        {action, tactile60, wrist_ft, image, wrist_image, state, state_name}."""
        env, cfg = self.env, self.cfg
        from rlinf.envs.sim.libero.tactile_patch import TactileReader
        reader = TactileReader(env.sim)

        obs = env.reset()
        self.comp = np.zeros(3)  # per-episode reset: compensation must not leak
        self.comp_adjust = 0
        if init_state is not None:
            sim = env.sim
            nq, nv = sim.model.nq, sim.model.nv
            sim.data.qpos[:] = init_state[1:1 + nq]
            sim.data.qvel[:] = init_state[1 + nq:1 + nq + nv]
            sim.forward()
            obs, _, _, _ = env.step(np.zeros(7))
        frames = []

        state, t_state = "pregrasp", 0
        grasp_pos = None
        comp_printed = False

        for t in range(cfg.max_steps):
            sim = env.sim
            bottle = self._bottle_pos(sim)
            rack = self._rack_pos(sim)
            eef = self._eef_pos(sim)

            # ---- state targets -------------------------------------------
            if state == "pregrasp":
                target = bottle + np.array([0, 0, cfg.pregrasp_z_off])
                gripper = 1.0
            elif state == "approach":
                target = bottle + np.array([0, 0, cfg.grasp_z_off])
                gripper = 1.0
            elif state == "close":
                target = eef.copy()
                gripper = -1.0
            elif state == "lift":
                target = (grasp_pos if grasp_pos is not None else eef) + \
                    np.array([cfg.lift_dx, 0, cfg.lift_dz])
                gripper = -1.0
            elif state == "above_rack":
                target = rack + np.array([0, 0, cfg.rack_above_dz])
                gripper = -1.0
            elif state == "place":
                target = rack + np.array([0, 0, cfg.rack_place_dz])
                gripper = -1.0
            elif state == "release":
                target = eef.copy()
                gripper = 1.0
            else:  # retreat
                target = eef + np.array([0, 0, 0.10])
                gripper = 1.0

            goal = target + self.comp

            # ---- transitions ---------------------------------------------
            if state == "pregrasp":
                if np.linalg.norm(eef - goal) < 0.03:
                    state, t_state = "approach", 0
            elif state == "approach":
                if np.linalg.norm(eef - goal) < 0.015:
                    state, t_state = "close", 0
                    grasp_pos = eef.copy()
            elif state == "close":
                if t_state >= cfg.close_steps:
                    state, t_state = "lift", 0
                    grasp_pos = eef.copy()
            elif state == "lift":
                if np.linalg.norm(eef - goal) < 0.03:
                    state, t_state = "above_rack", 0
            elif state == "above_rack":
                # carry sag: correct once, then descend regardless — the
                # place phase lets the bottle contact the slot (tactile).
                if np.linalg.norm(eef - goal) < 0.03:
                    state, t_state = "place", 0
                elif t_state >= cfg.state_timeout // 2 and self.comp_adjust < 3:
                    self.comp = np.clip(self.comp + (goal - eef) * 0.8,
                                        -cfg.comp_max, cfg.comp_max)
                    self.comp_adjust += 1
                    t_state = 0
            elif state == "place":
                if np.linalg.norm(eef - goal) < 0.02 or t_state >= cfg.place_steps:
                    state, t_state = "release", 0
            elif state == "release":
                if t_state >= cfg.release_steps:
                    state, t_state = "retreat", 0

            # ---- hard timeouts (pre-grasp phases must not grind) ---------
            if state in ("pregrasp", "approach", "close", "lift") and \
                    t_state >= cfg.state_timeout:
                if verbose:
                    print(f"  [timeout {state}] eef={np.round(eef,3)} "
                          f"goal={np.round(goal,3)}", flush=True)
                return False, frames

            scale = cfg.slow_factor if state in ("lift", "above_rack", "place") else 1.0
            action = self._servo_action(sim, goal, gripper, scale=scale)
            obs, _, _, _ = env.step(action)
            t_state += 1

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
                print(f"  t={t} state={state} eef={np.round(eef,3)} "
                      f"goal={np.round(goal,3)}", flush=True)

            if self._check(env):
                return True, frames

        return False, frames
