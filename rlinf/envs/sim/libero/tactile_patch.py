"""Tactile patch for the LIBERO/robosuite Panda gripper.

Injects 60 touch sites (2 fingers x 30) + MuJoCo <touch> sensors into the
Panda gripper XML so that simulated tactile readings are available in every
env built after the patch is applied. The wrist F/T channel needs no patch:
the stock panda_gripper.xml already carries force_ee / torque_ee sensors at
its ft_frame site.

Usage (before env creation):
    from rlinf.envs.sim.libero.tactile_patch import apply_touch_patch, TactileReader
    apply_touch_patch()                      # idempotent XML edit
    env = make_libero_env(...)               # any env built afterwards
    reader = TactileReader(env.sim)          # resolve sensor addresses once
    tactile60 = reader.tactile(env.sim)      # float32[60] per step
    ft6 = reader.wrist_ft(env.sim)           # float32[6]

The XML edit lands in the robosuite package inside the container image layer;
re-apply after container recreation (idempotent, costs milliseconds).
"""
import os
import re
from pathlib import Path

import numpy as np

MARKER = "<!-- tactile_patch_v1 -->"
N_SITES_PER_FINGER = 30
FINGERS = {"f0": "finger_joint1_tip", "f1": "finger_joint2_tip"}


def _gripper_xml_path() -> Path:
    import robosuite
    p = Path(robosuite.__file__).parent / "models" / "assets" / "grippers" / "panda_gripper.xml"
    if not p.is_file():  # older layouts kept grippers outside assets/
        alt = Path(robosuite.__file__).parent / "models" / "grippers" / "panda_gripper.xml"
        p = alt if alt.is_file() else p
    return p


def _finger_sites(finger_key: str):
    """30 sites over the finger pad box (half extents 0.008 x 0.004 x 0.008,
    pad faces +-y). Sites are slightly oversized boxes so MuJoCo touch
    sensors capture contact points on the pad surface."""
    sites = []
    for i in range(N_SITES_PER_FINGER):
        row, col = divmod(i, 15)  # 2 rows (y) x 15 cols (z)
        y = -0.002 + row * 0.004
        z = -0.007 + col * 0.001
        sites.append(
            f'        <site name="touch_{finger_key}_s{i:02d}" pos="0 {y:.4f} {z:.4f}" '
            f'size="0.0015 0.0015 0.0007" rgba="1 0 1 0.2" type="box" group="1"/>'
        )
    return "\n".join(sites)


def apply_touch_patch(xml_path: Path | None = None) -> Path:
    """Idempotently inject touch sites + sensors into the panda gripper XML."""
    p = Path(xml_path) if xml_path else _gripper_xml_path()
    xml = p.read_text()
    if MARKER in xml:
        return p

    # 60 sites: inserted right after each finger pad tip-body opening tag
    for finger_key, body in FINGERS.items():
        tag = f'<body name="{body}"'
        idx = xml.index(tag)
        line_end = xml.index(">", idx) + 1
        block = f"\n{MARKER}_{finger_key}\n" + _finger_sites(finger_key)
        xml = xml[:line_end] + block + xml[line_end:]
        xml = xml.replace(f"{MARKER}_{finger_key}\n", "", 1)  # keep marker only once

    # 60 touch sensors appended to the <sensor> block
    sensors = "\n".join(
        f'    <touch name="touch_{f}_s{i:02d}" site="touch_{f}_s{i:02d}"/>'
        for f in FINGERS
        for i in range(N_SITES_PER_FINGER)
    )
    xml = xml.replace("</sensor>", f"{sensors}\n</sensor>", 1)
    xml = xml.replace("<sensor>", f"{MARKER}\n    <sensor>", 1)

    # tactile skin surface = high-friction pads (mirrors the real T-Piezo
    # skin material; smooth finger mesh friction 1.0 slips on cylinders)
    xml = xml.replace('friction="1 0.005 0.0001"', 'friction="2.5 0.01 0.0001"')

    p.write_text(xml)
    return p


class TactileReader:
    """Resolve sensor addresses once, then read cheap per step.
    Prefix-tolerant: robosuite prepends robot prefixes (gripper0_...) to
    sensor names at model composition."""

    def __init__(self, sim):
        all_names = [str(n) for n in sim.model.sensor_names]

        def adr_of(name: str) -> int:
            for j, actual in enumerate(all_names):
                if actual == name or actual.endswith(name):
                    return sim.model.sensor_adr[j]
            raise ValueError(f"touch sensor '{name}' not in model")

        names = [f"touch_f{f}_s{i:02d}" for f in (0, 1) for i in range(N_SITES_PER_FINGER)]
        self.tactile_adr = np.asarray([adr_of(n) for n in names])
        self.ft_adr = [adr_of("force_ee"), adr_of("torque_ee")]

    def tactile(self, sim) -> np.ndarray:
        return sim.data.sensordata[self.tactile_adr].astype(np.float32)

    def wrist_ft(self, sim) -> np.ndarray:
        f = sim.data.sensordata[self.ft_adr[0]: self.ft_adr[0] + 3]
        t = sim.data.sensordata[self.ft_adr[1]: self.ft_adr[1] + 3]
        return np.concatenate([f, t]).astype(np.float32)


def apply_ft_patch(xml_path: Path | None = None) -> Path:
    """No-op: the stock panda_gripper.xml already carries force_ee/torque_ee.
    Kept as an explicit call site so the F/T provenance is documented."""
    return Path(xml_path) if xml_path else _gripper_xml_path()
