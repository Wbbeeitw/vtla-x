"""Empirical touch-sensor diagnostic: run the PPO policy until it grasps,
then dump the ACTUAL contact table (geom pairs + contact positions in the
finger-tip frame) vs our 60 taxel sites, plus F/T readings."""
import sys, os
sys.path.insert(0, "/workspace/RLinf")
sys.path.insert(0, "/opt/venv/openpi/libero")
sys.path.insert(0, "/workspace/RLinf/scripts")
os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np
import torch
from rlinf.envs.sim.libero.tactile_patch import apply_touch_patch, TactileReader
apply_touch_patch()
from libero.libero import benchmark
from collect_policy import make_env, build_model, to_env_obs, check_success
from rlinf.envs.sim.libero.utils import quat2axisangle

bench = benchmark.get_benchmark_dict()["libero_goal"](task_order_index=0)
env = make_env(bench.get_task_bddl_file_path(9))
reader = TactileReader(env.sim)
sim = env.sim

print("== sensors present ==")
names = [str(n) for n in sim.model.sensor_names]
touch = [n for n in names if "touch" in n]
print(f"total sensors={len(names)} touch sensors={len(touch)} first3={touch[:3]}")

# addresses sanity
t0 = reader.tactile(sim)
print("tactile60 init:", t0.min(), t0.max(), "| ft:", np.round(reader.wrist_ft(sim), 3))

model = build_model("/workspace/RLinf/evaluations/libero",
                    "vtla_triage_goal_openpi_pi05_eval")
device = next(model.parameters()).device

init_qpos_holder = bench.get_task_init_states(9)[0]
env.seed(7)
obs = env.reset()
sim = env.sim  # re-fetch after reset (bindings may be rebuilt)
reader = TactileReader(sim)
mdl = getattr(sim, "model", None) or sim.sim.model
dat = getattr(sim, "data", None) or sim.sim.data
nq, nv = mdl.nq, mdl.nv
dat.qpos[:] = init_qpos_holder[1:1 + nq]
dat.qvel[:] = init_qpos_holder[1 + nq:1 + nq + nv]
sim.forward()
settle = np.zeros(7); settle[-1] = -1.0
for _ in range(15):
    obs, _, _, _ = env.step(settle)

geom_names = [mdl.geom_id2name(i) for i in range(mdl.ngeom)]
body_of_geom = sim.model.geom_bodyid
tip1 = sim.model.body_name2id("gripper0_finger_joint1_tip")
tip2 = sim.model.body_name2id("gripper0_finger_joint2_tip")

for t in range(140):
    env_obs = to_env_obs(obs, "put the wine bottle on the rack")
    with torch.no_grad():
        actions, _ = model.predict_action_batch(env_obs=env_obs, mode="eval")
    chunk = np.asarray(actions.detach().cpu().numpy())
    if chunk.ndim == 3:
        chunk = chunk[0]
    for a in chunk:
        obs, _, _, _ = env.step(a)
        tac = reader.tactile(sim)
        if t % 20 == 0 or (tac.max() > 0 and t < 5):
            rows = []
            d = getattr(sim, "data", None) or sim.sim.data
            for ci in range(d.ncon):
                c = d.contact[ci]
                g1, g2 = geom_names[c.geom1], geom_names[c.geom2]
                if g1 and g2 and ("finger" in g1 or "finger" in g2):
                    other = g2 if "finger" in g1 else g1
                    fg = c.geom1 if "finger" in g1 else c.geom2
                    pos = c.pos.copy()
                    # contact pos in world; convert to tip-local for diagnosis
                    rows.append((other, fg, pos, c.dist))
            print(f"t={t} ncon={sim.data.ncon} tactile_max={tac.max():.4f} "
                  f"ft_norm={np.linalg.norm(reader.wrist_ft(sim)):.3f} finger_contacts={len(rows)}",
                  flush=True)
            # site world positions vs contact world positions
            d = getattr(sim, "data", None) or sim.sim.data
            sid0 = mdl.site_name2id("gripper0_touch_f0_s00")
            sid7 = mdl.site_name2id("gripper0_touch_f0_s07")
            sid14 = mdl.site_name2id("gripper0_touch_f0_s14")
            sid29 = mdl.site_name2id("gripper0_touch_f0_s29")
            s1 = mdl.site_name2id("gripper0_touch_f1_s00")
            print(f"   SITE f0_s00={np.round(d.site_xpos[sid0],4)} "
                  f"f0_s07={np.round(d.site_xpos[sid7],4)} "
                  f"f0_s14={np.round(d.site_xpos[sid14],4)} "
                  f"f0_s29={np.round(d.site_xpos[sid29],4)} "
                  f"f1_s00={np.round(d.site_xpos[s1],4)}", flush=True)
            for other, fg, pos, dist in rows[:3]:
                print(f"   other={other} finger_geom={geom_names[fg]} "
                      f"world_pos={np.round(pos,4)} dist={dist:.4f}", flush=True)
        if check_success(env):
            print("SUCCESS at step", t)
            sys.exit(0)
print("done probing")
