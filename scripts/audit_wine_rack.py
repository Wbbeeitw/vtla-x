"""Quality audit for the wine_rack_tactile LeRobot dataset (read-only).

Seven-layer checks per the approved plan; writes audit_report.md +
audit_detail.json under /data_vtlax/results/audit/.
"""
import json
import os
import sys

sys.path.insert(0, "/workspace/RLinf")
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np
import torch
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

REPO = "wine_rack_tactile"
ROOT = "/data_vtlax/datasets/wine_rack_tactile"  # explicit root = dataset dir itself (no repo_id subdir when root is passed)
OUT_DIR = "/data_vtlax/results/audit"
PROMPT_EXPECT = "put the wine bottle on the rack"
FPS = 20

os.makedirs(OUT_DIR, exist_ok=True)

ds = LeRobotDataset(REPO, root=ROOT)
meta = ds.meta
n_ep = meta.total_episodes
total_frames = meta.total_frames
fps = meta.fps
print(f"dataset: {n_ep} episodes, {total_frames} frames, fps={fps}", flush=True)

report_eps = []
all_tactile_max = []
all_taxel_active = np.zeros(60, dtype=int)
all_len = []
fail_rows = []

# read true episode boundaries from info.json (authoritative)
info = json.load(open(os.path.join(ROOT, "meta", "info.json")))
ep_lengths = {ep["episode_index"]: ep["length"] for ep in info["episodes"]}
print("episodes from info.json:", len(ep_lengths), flush=True)

report_eps = []
for ep_idx in sorted(ep_lengths):
    f0 = start_of[ep_idx]
    length = ep_lengths[ep_idx]
    rec = {"episode_index": int(ep_idx), "frames": int(length)}

    # ---- per-frame read via LeRobotDataset slicing ----
    seg = ds[f0:f0 + length]
    # ---- tactile ----
    rec["tactile_max"] = float(tac.max())
    rec["tactile_mean"] = float(tac.mean())
    active = (tac.max(axis=0) > 0.1)
    rec["active_taxels"] = int(active.sum())
    contact_frames = (tac.max(axis=1) > 0.1)
    rec["contact_frame_frac"] = float(contact_frames.mean())
    rec["grasp_window_max"] = float(tac[: max(1, length // 2)].max())
    all_tactile_max.append(rec["tactile_max"])
    all_taxel_active += active.astype(int)

    # ---- ft ----
    # ft already stacked above
    ft_norm = np.linalg.norm(ft[:, :3], axis=1)
    rec["ft_norm_min"] = float(ft_norm.min())
    rec["ft_norm_max"] = float(ft_norm.max())
    rec["ft_norm_mean"] = float(ft_norm.mean())

    # ---- images (subsample 8 frames) ----

    idxs = np.linspace(0, length - 1, min(8, length)).astype(int)
    stds, diffs, darks = [], [0.0], []
    prev = None
    for i in idxs:
        fr = img[i].astype(np.float32)
        stds.append(float(fr.std()))
        darks.append(float(fr.mean()))
        if prev is not None:
            diffs.append(float(np.abs(fr - prev).mean()))
        prev = fr
    rec["img_std_first"] = stds[0]
    rec["img_min_std"] = min(stds)
    rec["img_frame_diff_min"] = min(diffs[1:]) if len(diffs) > 1 else 0.0
    rec["img_dark_min"] = min(darks)

    # ---- state / action physics ----
    st = np.asarray(st_l, dtype=np.float64).reshape(length, -1)
    act = np.asarray(act_l, dtype=np.float64).reshape(length, -1)
    rec["state_nan"] = bool(np.isnan(st).any() or np.isinf(st).any())
    if length > 1:
        d = np.abs(np.diff(st[:, :3], axis=0)).max()
        rec["max_step_xy jump_pos"] = float(d)
        rec["max_step_pos_delta"] = float(d)
    rec["action_absmax"] = float(np.abs(act).max())
    rec["act_nan"] = bool(np.isnan(act).any())
    g = act[:, -1] if act.shape[1] >= 7 else act[:, -1:]
    rec["gripper_transitions"] = int((np.diff(np.sign(g)) != 0).sum())

    # ---- hard-fail rules ----
    reasons = []
    if rec["state_nan"] or rec["act_nan"]:
        reasons.append("nan")
    if rec["tactile_max"] <= 0:
        reasons.append("tactile_all_zero")
    if length < 50:
        reasons.append("too_short")
    if rec["img_min_std"] < 1.0:
        reasons.append("image_frozen_or_blank")
    if reasons:
        fail_rows.append({"episode": int(ep_idx), "reasons": reasons})
    report_eps.append({"episode_index": int(ep_idx), **rec,
                       "tactile_ok": rec["tactile_max"] >= 0.5,
                       "fail_reasons": reasons})

# ---- overall aggregates ----
lens = [r["frames"] for r in report_eps]
tmax = [r["tactile_max"] for r in report_eps]
atax = [r["active_taxels"] for r in report_eps]
first_eef = []
for r in report_eps:
    pass  # filled from detail below via state recs if needed

n_fail = len(fail_rows)
summary = {
    "episodes_total": len(report_eps),
    "episodes_failed": n_fail,
    "fail_rows": fail_rows,
    "len_mean": float(np.mean(lens)), "len_min": int(min(lens)), "len_max": int(max(lens)),
    "len_std": float(np.std(lens)),
    "tactile_max_overall": float(max(tmax)) if tmax else 0.0,
    "tactile_max_min": float(min(tmax)) if tmax else 0.0,
    "active_taxel_min": int(min(atax)) if atax else 0,
    "dead_taxels": int((all_taxel_active == 0).sum()),
    "img_freeze_eps": sum(1 for r in report_eps if r.get("img_frame_diff_min", 1) == 0),
}

with open(os.path.join(OUT_DIR, "audit_detail.json"), "w") as f:
    json.dump({"episodes": report_eps, "summary": summary}, f, indent=2, default=str)

md = ["# wine_rack_tactile 审计报告", "",
      f"- episodes: {summary['episodes_total']}(失败 {n_fail})",
      f"- 帧数: mean {summary['len_mean']:.0f} / min {summary['len_min']} / max {summary['len_max']}",
      f"- 触觉 max: 总体 {summary['tactile_max_overall']:.2f}N,最弱集 {summary['tactile_max_min']:.2f}N",
      f"- 死 taxel: {summary['dead_taxels']}/60",
      f"- 图像冻结集: {summary['img_freeze_eps']}",
      "", "| 集 | 帧数 | tactile_max | active_taxels | contact% | ft_mean | img_std_min | 失败原因 |",
      "|---|---|---|---|---|---|---|---|"]
for r in report_eps:
    md.append(f"| {r['episode_index']} | {r['frames']} | {r['tactile_max']:.2f} | "
              f"{r['active_taxels']} | {r['contact_frame_frac']:.0%} | "
              f"{r['ft_norm_mean']:.1f} | {r['img_std_first']:.1f} | "
              f"{','.join(r['fail_reasons']) or '-'} |")
with open(os.path.join(OUT_DIR, "audit_report.md"), "w") as f:
    f.write("\n".join(md) + "\n")
print(json.dumps(summary, indent=2), flush=True)
print("REPORT:", os.path.join(OUT_DIR, "audit_report.md"), flush=True)
