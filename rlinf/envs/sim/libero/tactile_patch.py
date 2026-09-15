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
    """30 sites per finger on the pad's INNER face (the face that grips).

    Pad collision geom (tip-body frame, from panda_gripper.xml):
      finger1 tip sits at +y; pad volume y∈[-0.009,-0.001] -> the face
      toward the grip center is y=-0.009 (NOT -0.001; that's the outer side)
      finger2 mirrored: inner face y=+0.009
    Sites sit slightly PROUD of the inner face so contact points (which lie
    on/just outside the surface plane) fall inside a site volume.
    Grid: 2 columns (x) x 15 rows (z) per finger."""
    sign = -1.0 if finger_key == "f0" else 1.0
    y_face = sign * 0.0095  # inner face of the pad, 0.5mm proud
    sites = []
    for i in range(N_SITES_PER_FINGER):
        col_x, row_z = divmod(i, 15)  # 2 cols (x) x 15 rows (z)
        x = (-0.004 + col_x * 0.008)
        z = (-0.022 + row_z * 0.001)
        sites.append(
            f'        <site name="touch_{finger_key}_s{i:02d}" '
            f'pos="{x:.4f} {y_face:.4f} {z:.4f}" '
            f'size="0.0035 0.0015 0.0006" rgba="1 0 1 0.2" type="box" group="1"/>'
        )
    return "\n".join(sites)


def _strip_previous(xml: str) -> str:
    """Remove any prior touch-site/sensor injections so we can re-patch
    with fresh geometry (the container XML is patched in place)."""
    lines = [ln for ln in xml.splitlines()
             if "touch_f0_s" not in ln and "touch_f1_s" not in ln
             and MARKER not in ln]
    xml = "\n".join(lines)
    if xml and not xml.endswith("\n"):
        xml += "\n"
    return xml


def apply_touch_patch(xml_path: Path | None = None) -> Path:
    """Inject touch sites + sensors into the panda gripper XML (re-patchable)."""
    p = Path(xml_path) if xml_path else _gripper_xml_path()
    xml = _strip_previous(p.read_text())

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
