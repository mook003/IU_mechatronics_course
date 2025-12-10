#!/usr/bin/env python
"""
UR3 actuator sizing and MJCF model generation for MuJoCo.

This is a single-file version that combines the previous `sizing.py` and
`main.py` and moves motor parameters into CSV files.

Expected project layout:

  model-input/
    universalUR3.xml      # base UR3 MJCF (from MuJoCo "Save XML")
    old_motors.csv        # parameters of the original UR3 motors
    new_motors.csv        # list of candidate harmonic drives per joint

  model-output/
    ... generated MJCF models ...

Run from project root as:

  cd src
  python ur3_actuator_sizing.py
"""

from __future__ import annotations

from dataclasses import dataclass
from copy import deepcopy
from itertools import product
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple
import csv

import numpy as np
from lxml import etree


# =============================================================================
# Data structures
# =============================================================================


@dataclass
class HarmonicDrive:
    """New motor / harmonic drive mounted on the joint."""
    name: str                 # model name
    armature_inertia: float   # reflected inertia on joint axis (GD^2/4), kg·m^2
    mass: float               # mass of actuator, kg
    radius: float             # cylinder radius for housing, m
    length: float             # cylinder length, m
    max_torque: float         # Maximum Torque from catalog, N·m


@dataclass
class MotorShell:
    """Approximate cylindrical shell of the original UR motor."""
    mass: float   # kg
    radius: float # m
    length: float # m


# =============================================================================
# XML helpers
# =============================================================================


def load_xml(path: Path) -> etree._ElementTree:
    """Load MJCF / XML file."""
    return etree.parse(str(path))


def save_xml(tree: etree._ElementTree, path: Path) -> None:
    """Save MJCF / XML file with pretty formatting."""
    tree.write(
        str(path),
        pretty_print=True,
        xml_declaration=True,
        encoding="utf-8",
    )


def find_joints(tree: etree._ElementTree):
    """Return all <joint> elements in the model."""
    root = tree.getroot()
    return root.findall(".//joint")


def map_joint_to_body(tree: etree._ElementTree) -> Dict[str, Tuple[etree._Element, etree._Element]]:
    """
    Build mapping: joint_name -> (joint_element, parent_body_element).

    Assumes that each <joint> is a direct child of the corresponding <body>.
    """
    result: Dict[str, Tuple[etree._Element, etree._Element]] = {}
    for joint in find_joints(tree):
        name = joint.get("name")
        if not name:
            continue
        body = joint.getparent()
        result[name] = (joint, body)
    return result


def get_or_create_actuator_root(root: etree._Element) -> etree._Element:
    """Return <actuator> element, create it if missing."""
    act = root.find("actuator")
    if act is None:
        act = etree.SubElement(root, "actuator")
    return act


# =============================================================================
# Inertial helpers
# =============================================================================


def get_inertial(body_el: etree._Element) -> etree._Element:
    """Return <inertial> element of a body or raise if missing."""
    inertial = body_el.find("inertial")
    if inertial is None:
        raise RuntimeError(f"No <inertial> in body '{body_el.get('name')}'")
    return inertial


def parse_inertial(inertial_el: etree._Element):
    """
    Parse MJCF inertial tag into (mass, pos, diag_inertia).
    pos is a 3D vector, diag_inertia is [ixx, iyy, izz].
    """
    m = float(inertial_el.get("mass"))
    pos_str = inertial_el.get("pos", "0 0 0")
    pos = np.fromstring(pos_str, sep=" ")
    I_diag = np.fromstring(inertial_el.get("diaginertia"), sep=" ")
    return m, pos, I_diag


def cylinder_inertia_diag(m: float, R: float, h: float) -> np.ndarray:
    """
    Diagonal inertia of a solid cylinder in its COM frame (axis along Z).

    Ixx = Iyy = (1/12) m (3R^2 + h^2)
    Izz = (1/2)  m R^2
    """
    i_xy = (1.0 / 12.0) * m * (3.0 * R**2 + h**2)
    i_z = 0.5 * m * R**2
    return np.array([i_xy, i_xy, i_z])


# =============================================================================
# Armature & actuators
# =============================================================================


def set_joint_armature(joint_el: etree._Element, hd: HarmonicDrive) -> None:
    """
    Set MuJoCo 'armature' on the joint.

    hd.armature_inertia is GD^2/4 from the catalog,
    already interpreted as inertia at the joint axis (kg·m^2).
    """
    joint_el.set("armature", f"{hd.armature_inertia:.6e}")


def add_motor_actuator(
    actuator_root: etree._Element,
    joint_name: str,
    hd: HarmonicDrive,
) -> None:
    """
    Add a <general> actuator for a joint, with torque limit from the catalog.

    ctrlrange = ±hd.max_torque.
    """
    motor_name = f"{joint_name}_motor_{hd.name}"
    T_max = hd.max_torque

    motor_el = etree.SubElement(actuator_root, "general")
    motor_el.set("name", motor_name)
    motor_el.set("joint", joint_name)
    motor_el.set("gear", "1.0")  # control units = joint torque
    motor_el.set("ctrlrange", f"{-T_max:.3f} {T_max:.3f}")


# =============================================================================
# Motor replacement (inertial update)
# =============================================================================


def replace_motor_in_inertial_full(
    inertial_el: etree._Element,
    old_motor: MotorShell | None,
    new_motor: HarmonicDrive | None,
    r_m_old: np.ndarray | None = None,  # COM of old motor in body frame
    R_old: np.ndarray | None = None,    # 3x3 rotation from motor principal axes to body frame
    r_m_new: np.ndarray | None = None,  # COM of new motor in body frame
    R_new: np.ndarray | None = None,    # 3x3 rotation for new motor
) -> None:
    """
    Replace old motor by new motor in the link inertial using the full
    parallel axis theorem, including COM shift.

    Steps:
      - start from current (M_tot, r_t, I_tot about r_t),
      - remove old motor → (M_clean, r_clean, I_clean about r_clean),
      - add new motor → (M_final, r_final, I_final about r_final),
      - write back mass, pos=r_final, and diaginertia=diag(I_final).
    """
    I3 = np.eye(3)

    M_tot, r_t, I_diag = parse_inertial(inertial_el)
    r_t = np.asarray(r_t, dtype=float)
    I_tot = np.diag(I_diag)

    # --- remove old motor ----------------------------------------------------
    if old_motor is not None:
        m_old = float(old_motor.mass)

        M_clean = M_tot - m_old
        if M_clean <= 0:
            raise RuntimeError(
                f"Non-positive clean mass after removing motor from body "
                f"{inertial_el.getparent().get('name')}: {M_clean}"
            )

        # COM of old motor
        if r_m_old is None:
            r_m_old = r_t.copy()
        else:
            r_m_old = np.asarray(r_m_old, dtype=float)

        # Orientation of old motor principal axes
        if R_old is None:
            R_old = I3
        else:
            R_old = np.asarray(R_old, dtype=float).reshape(3, 3)

        # Inertia of old motor in its own COM
        I_old_cm_diag = cylinder_inertia_diag(m_old, old_motor.radius, old_motor.length)
        I_old_cm = np.diag(I_old_cm_diag)

        # Rotate into body frame
        I_old_aligned = R_old @ I_old_cm @ R_old.T

        # Parallel axis theorem from r_m_old to r_t
        d_old = r_m_old - r_t
        I_old_about_rt = I_old_aligned + m_old * (
            (np.dot(d_old, d_old)) * I3 - np.outer(d_old, d_old)
        )

        # Remove old motor from total inertia about r_t
        I_clean_about_rt = I_tot - I_old_about_rt

        # New COM (link without motor)
        r_clean = (M_tot * r_t - m_old * r_m_old) / M_clean

        # Shift inertia from r_t to r_clean
        d_shift = r_clean - r_t
        I_clean = I_clean_about_rt - M_clean * (
            (np.dot(d_shift, d_shift)) * I3 - np.outer(d_shift, d_shift)
        )

    else:
        # No old motor: "clean" link is just current inertial
        M_clean = M_tot
        r_clean = r_t.copy()
        I_clean = I_tot

    # --- add new motor -------------------------------------------------------
    if new_motor is not None:
        m_new = float(new_motor.mass)

        if r_m_new is None:
            r_m_new = r_clean.copy()
        else:
            r_m_new = np.asarray(r_m_new, dtype=float)

        if R_new is None:
            R_new = I3
        else:
            R_new = np.asarray(R_new, dtype=float).reshape(3, 3)

        I_new_cm_diag = cylinder_inertia_diag(m_new, new_motor.radius, new_motor.length)
        I_new_cm = np.diag(I_new_cm_diag)

        I_new_aligned = R_new @ I_new_cm @ R_new.T

        d_new = r_m_new - r_clean
        I_new_about_clean = I_new_aligned + m_new * (
            (np.dot(d_new, d_new)) * I3 - np.outer(d_new, d_new)
        )

        M_final = M_clean + m_new
        I_tot_about_clean = I_clean + I_new_about_clean

        r_final = (M_clean * r_clean + m_new * r_m_new) / M_final

        d_shift2 = r_final - r_clean
        I_final = I_tot_about_clean - M_final * (
            (np.dot(d_shift2, d_shift2)) * I3 - np.outer(d_shift2, d_shift2)
        )

    else:
        # No new motor: final = clean
        M_final = M_clean
        r_final = r_clean
        I_final = I_clean

    I_final_diag = np.diag(I_final)

    inertial_el.set("mass", f"{M_final:.6e}")
    inertial_el.set("pos", " ".join(f"{v:.6e}" for v in r_final))
    inertial_el.set("diaginertia", " ".join(f"{v:.6e}" for v in I_final_diag))


def replace_motor_in_inertial_ur3(
    inertial_el: etree._Element,
    old_motor: MotorShell,
    new_motor: HarmonicDrive,
) -> None:
    """
    UR3-specific wrapper around the full replacement:

    Assumptions:
      - old and new motors sit exactly at the link COM (inertial pos),
      - their principal axes are aligned with the link frame.
    Under these assumptions the full formula reduces to simple
    subtraction/addition of diagonal inertias, but keeps the generality.
    """
    _, r_t, _ = parse_inertial(inertial_el)

    replace_motor_in_inertial_full(
        inertial_el=inertial_el,
        old_motor=old_motor,
        new_motor=new_motor,
        r_m_old=np.array(r_t),
        R_old=np.eye(3),
        r_m_new=np.array(r_t),
        R_new=np.eye(3),
    )


# =============================================================================
# High-level generator
# =============================================================================


class UR3ModelGenerator:
    """
    High-level API for generating UR3 MJCF variants.

    It expects:
      - base_mjcf: path to base UR3 MJCF file,
      - joint_order: list of joints in fixed order,
      - old_motors: mapping joint_name -> MotorShell,
      - new_motor_variants: mapping joint_name -> list[HarmonicDrive],
      - output_dir: where to save generated models.
    """

    def __init__(
        self,
        base_mjcf: Path,
        joint_order: Sequence[str],
        old_motors: Mapping[str, MotorShell],
        new_motor_variants: Mapping[str, List[HarmonicDrive]],
        output_dir: Path,
    ) -> None:
        self.base_mjcf = Path(base_mjcf)
        self.joint_order = list(joint_order)
        self.old_motors = dict(old_motors)
        self.new_motor_variants = {j: list(v) for j, v in new_motor_variants.items()}
        self.output_dir = Path(output_dir)

    def build_one_variant(self, filename: str = "ur3_new_replace_motor_single.xml") -> Path:
        """
        Build a single UR3 model with a fixed set of new motors:

          - for each joint, use new_motor_variants[joint][0],
          - set 'armature' on joints,
          - update link inertials (old UR motor → new harmonic drive),
          - create general actuators with proper torque limits.
        """
        tree = load_xml(self.base_mjcf)
        root = tree.getroot()

        joint_map = map_joint_to_body(tree)
        actuator_root = get_or_create_actuator_root(root)

        for joint_name in self.joint_order:
            hd_list = self.new_motor_variants[joint_name]
            new_motor = hd_list[0]
            old_motor = self.old_motors[joint_name]

            joint_el, body_el = joint_map[joint_name]
            inertial_el = get_inertial(body_el)

            set_joint_armature(joint_el, new_motor)
            replace_motor_in_inertial_ur3(inertial_el, old_motor, new_motor)
            add_motor_actuator(actuator_root, joint_name, new_motor)

        self.output_dir.mkdir(parents=True, exist_ok=True)
        out_path = self.output_dir / filename
        save_xml(tree, out_path)
        print(f"[baseline] saved: {out_path}")
        return out_path

    def generate_all_variants(self, prefix: str = "ur3_new_variant_") -> None:
        """
        Generate all combinations of new motors for all joints.

        For each combination:
          - copy the base MJCF tree,
          - for each joint:
              * set armature,
              * replace old UR motor with chosen new motor in link inertial,
              * add general actuator with torque limits,
          - save result as separate XML file in output_dir.
        """
        base_tree = load_xml(self.base_mjcf)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        variants_lists = [self.new_motor_variants[j] for j in self.joint_order]

        for idx, combo in enumerate(product(*variants_lists), start=1):
            tree = deepcopy(base_tree)
            root = tree.getroot()
            joint_map = map_joint_to_body(tree)
            actuator_root = get_or_create_actuator_root(root)

            for joint_name, hd in zip(self.joint_order, combo):
                joint_el, body_el = joint_map[joint_name]
                inertial_el = get_inertial(body_el)

                set_joint_armature(joint_el, hd)
                replace_motor_in_inertial_ur3(inertial_el, self.old_motors[joint_name], hd)
                add_motor_actuator(actuator_root, joint_name, hd)

            suffix = "_".join(hd.name for hd in combo)
            out_path = self.output_dir / f"{prefix}{idx:02d}.xml"
            save_xml(tree, out_path)
            print(f"[{idx:02d}] saved: {out_path} ({suffix})")


# =============================================================================
# CSV-based motor configuration
# =============================================================================


def load_motor_config_from_csv(config_dir: Path):
    """
    Load joint order, old motors and new motor variants from CSV files.

    Files:

      config_dir / "old_motors.csv"
        joint_name,mass_kg,radius_m,length_m

      config_dir / "new_motors.csv"
        joint_name,model_name,armature_inertia_kgm2,mass_kg,radius_m,length_m,max_torque_Nm
    """
    old_csv = config_dir / "old_motors.csv"
    new_csv = config_dir / "new_motors.csv"

    if not old_csv.exists():
        raise FileNotFoundError(f"Old motors CSV not found: {old_csv}")
    if not new_csv.exists():
        raise FileNotFoundError(f"New motors CSV not found: {new_csv}")

    # --- old motors ----------------------------------------------------------
    joint_order: List[str] = []
    old_motors: Dict[str, MotorShell] = {}

    with old_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"joint_name", "mass_kg", "radius_m", "length_m"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(
                f"{old_csv} must contain columns: {', '.join(sorted(required))}. "
                f"Got: {reader.fieldnames!r}"
            )
        for row in reader:
            joint_name = row["joint_name"].strip()
            if not joint_name:
                continue
            if joint_name in old_motors:
                raise ValueError(f"Duplicate joint_name {joint_name!r} in {old_csv}")
            mass = float(row["mass_kg"])
            radius = float(row["radius_m"])
            length = float(row["length_m"])
            old_motors[joint_name] = MotorShell(mass=mass, radius=radius, length=length)
            joint_order.append(joint_name)

    # --- new motors ----------------------------------------------------------
    new_motor_variants: Dict[str, List[HarmonicDrive]] = {j: [] for j in joint_order}

    with new_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {
            "joint_name",
            "model_name",
            "armature_inertia_kgm2",
            "mass_kg",
            "radius_m",
            "length_m",
            "max_torque_Nm",
        }
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(
                f"{new_csv} must contain columns: {', '.join(sorted(required))}. "
                f"Got: {reader.fieldnames!r}"
            )

        for row in reader:
            joint_name = row["joint_name"].strip()
            if not joint_name:
                continue

            if joint_name not in new_motor_variants:
                # allow extra joints not present in old_motors.csv (will be appended)
                new_motor_variants[joint_name] = []
                if joint_name not in joint_order:
                    joint_order.append(joint_name)

            hd = HarmonicDrive(
                name=row["model_name"].strip(),
                armature_inertia=float(row["armature_inertia_kgm2"]),
                mass=float(row["mass_kg"]),
                radius=float(row["radius_m"]),
                length=float(row["length_m"]),
                max_torque=float(row["max_torque_Nm"]),
            )
            new_motor_variants[joint_name].append(hd)

    # basic sanity check
    for j in joint_order:
        if j not in new_motor_variants or not new_motor_variants[j]:
            raise ValueError(f"No new motors configured for joint {j!r} in {new_csv}")

    return joint_order, old_motors, new_motor_variants


# =============================================================================
# CLI
# =============================================================================


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]

    # Base MJCF produced by MuJoCo "File -> Save XML" from UR3 URDF
    base_mjcf = project_root  / "models" / "source" / "description" / "universalUR3.xml"
    output_dir = project_root / "models" / "source" / "my"
    config_dir = project_root / "models" / "motors"

    if not base_mjcf.exists():
        raise FileNotFoundError(f"Base MJCF not found: {base_mjcf}")

    joint_order, old_motors, new_motor_variants = load_motor_config_from_csv(config_dir)

    generator = UR3ModelGenerator(
        base_mjcf=base_mjcf,
        joint_order=joint_order,
        old_motors=old_motors,
        new_motor_variants=new_motor_variants,
        output_dir=output_dir,
    )

    # Single reference model (first motor choice per joint)
    baseline_path = generator.build_one_variant()

    # All combinations of NEW motors per joint
    generator.generate_all_variants()

    print(f"Done. Baseline model: {baseline_path}")


if __name__ == "__main__":
    main()
