import sys, os
sys.path.insert(0, "/workspace/RLinf")
sys.path.insert(0, "/opt/venv/openpi/libero")
sys.path.insert(0, "/workspace/RLinf/scripts")
os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np
from rlinf.envs.sim.libero.tactile_patch import apply_touch_patch, TactileReader
apply_touch_patch()
from libero.libero import benchmark
from libero_expert import WineRackExpert, ExpertConfig
from collect_wine_rack import make_env
bench = benchmark.get_benchmark_dict()["libero_goal"](task_order_index=0)
bddl = bench.get_task_bddl_file_path(0)
init_states = bench.get_task_init_states(0)
env = make_env(bddl)
expert = WineRackExpert(env, ExpertConfig())
ok, frames = expert.rollout(init_state=init_states[0], verbose=True)
sim = env.sim
bp = sim.data.body_xpos[sim.model.body_name2id("wine_bottle_1_main")]
rp = sim.data.site_xpos[sim.model.site_name2id("wine_rack_1_top_region")]
j1 = sim.model.joint_name2id("finger_joint1")
gq = sim.data.qpos[sim.model.jnt_qposadr[j1]]
print("RESULT:", ok, len(frames))
print("bottle end:", np.round(bp,3), "| slot:", np.round(rp,3), "| horiz gap:", round(float(np.linalg.norm(bp[:2]-rp[:2])),3))
print("finger1 open:", round(float(gq),4), "(0=gripped, 0.04=open)")
