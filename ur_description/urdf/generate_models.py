# generate_models.py
from dataclasses import dataclass
from copy import deepcopy
from pathlib import Path
from itertools import product  # <=== добавь в начало файла

import numpy as np
from lxml import etree

from harmonic_drives import HarmonicDrive, joint_variants


# ===== Вспомогательные функции для XML =====

def load_xml(path: str) -> etree._ElementTree:
    return etree.parse(path)


def save_xml(tree: etree._ElementTree, path: str) -> None:
    tree.write(
        path,
        pretty_print=True,
        xml_declaration=True,
        encoding="utf-8",
    )


def find_joints(tree: etree._ElementTree):
    root = tree.getroot()
    return root.findall(".//joint")


def map_joint_to_body(tree: etree._ElementTree):
    res = {}
    for joint in find_joints(tree):
        name = joint.get("name")
        if not name:
            continue
        body = joint.getparent()
        res[name] = (joint, body)
    return res


def get_or_create_actuator_root(root: etree._Element) -> etree._Element:
    act = root.find("actuator")
    if act is None:
        act = etree.SubElement(root, "actuator")
    return act


# ===== Inertial helpers =====

def get_inertial(body_el: etree._Element) -> etree._Element:
    inertial = body_el.find("inertial")
    if inertial is None:
        raise RuntimeError(f"No <inertial> in body '{body_el.get('name')}'")
    return inertial


def parse_inertial(inertial_el: etree._Element):
    m = float(inertial_el.get("mass"))
    pos_str = inertial_el.get("pos", "0 0 0")
    pos = np.fromstring(pos_str, sep=" ")
    I_diag = np.fromstring(inertial_el.get("diaginertia"), sep=" ")
    return m, pos, I_diag


def cylinder_inertia_diag(m: float, R: float, h: float) -> np.ndarray:
    """
    Диагональ тензора инерции цилиндра в его центре масс,
    ось цилиндра вдоль локальной Z.
    """
    i_xy = (1.0 / 12.0) * m * (3 * R**2 + h**2)
    i_z = 0.5 * m * R**2
    return np.array([i_xy, i_xy, i_z])



def set_joint_armature(joint_el: etree._Element, hd: HarmonicDrive) -> None:
    """
    armature в MuJoCo — это добавочная инерция по оси сустава.
    В нашем случае мы храним уже приведённую инерцию актуатора
    на стороне сустава (armature_inertia), поэтому её и пишем.
    """
    joint_el.set("armature", f"{hd.armature_inertia:.6e}")


def update_link_inertia_with_motor_simple(inertial_el: etree._Element,
                                          hd: HarmonicDrive) -> None:
    """
    Простейший вариант: считаем, что мотор сидит в центре масс линка,
    ось мотора = локальная Z. Тогда:
      - центры масс не сдвигаем,
      - просто суммируем диагонали тензоров инерции.
    """
    m_link, pos_link, I_link_diag = parse_inertial(inertial_el)

    m_mot = hd.mass
    I_mot_diag = cylinder_inertia_diag(m_mot, hd.radius, hd.length)

    m_new = m_link + m_mot
    pos_new = pos_link
    I_new_diag = I_link_diag + I_mot_diag

    inertial_el.set("mass", f"{m_new:.6e}")
    inertial_el.set("pos", " ".join(f"{v:.6e}" for v in pos_new))
    inertial_el.set("diaginertia", " ".join(f"{v:.6e}" for v in I_new_diag))


def add_motor_actuator(actuator_root: etree._Element,
                       joint_name: str,
                       hd: HarmonicDrive) -> None:
    """
    Добавляет general-актуатор с лимитом по моменту (ctrlrange = ±MaxTorque).
    """
    motor_name = f"{joint_name}_motor_{hd.name}"
    T_max = hd.max_torque  # N·m

    motor_el = etree.SubElement(actuator_root, "general")
    motor_el.set("name", motor_name)
    motor_el.set("joint", joint_name)
    motor_el.set("gear", "1.0")  # ctrl = момент в суставе
    motor_el.set("ctrlrange", f"{-T_max:.3f} {T_max:.3f}")



def build_one_variant(base_xml_path: str, output_path: str) -> None:
    tree = load_xml(base_xml_path)
    root = tree.getroot()

    joint_map = map_joint_to_body(tree)
    actuator_root = get_or_create_actuator_root(root)

    for joint_name, hd_list in joint_variants.items():
        hd = hd_list[0]

        joint_el, body_el = joint_map[joint_name]
        inertial_el = get_inertial(body_el)

        # 1) инерция мотора в joint → armature
        set_joint_armature(joint_el, hd)
        # 2) масса + инерция линка
        update_link_inertia_with_motor_simple(inertial_el, hd)
        # 3) actuator
        add_motor_actuator(actuator_root, joint_name, hd)

    save_xml(tree, output_path)
    print(f"Saved modified model to: {output_path}")


def generate_all_variants(base_xml_path: str, output_dir: str) -> None:
    base_tree = load_xml(base_xml_path)
    base_root = base_tree.getroot()
    joint_names = [
        "shoulder_pan_joint",
        "shoulder_lift_joint",
        "elbow_joint",
        "wrist_1_joint",
        "wrist_2_joint",
        "wrist_3_joint",
    ]

    variants_lists = [joint_variants[j] for j in joint_names]

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Все декартовы произведения вариантов
    for idx, combo in enumerate(product(*variants_lists)):
        tree = deepcopy(base_tree)
        root = tree.getroot()

        joint_map = map_joint_to_body(tree)
        actuator_root = get_or_create_actuator_root(root)

        name_parts = []

        for joint_name, hd in zip(joint_names, combo):
            joint_el, body_el = joint_map[joint_name]
            inertial_el = get_inertial(body_el)

            # 1) инерция привода → armature
            set_joint_armature(joint_el, hd)
            # 2) масса и тензор инерции звена + корпус мотора-цилиндра
            update_link_inertia_with_motor_simple(inertial_el, hd)
            # 3) actuator general с ограничением по моменту
            add_motor_actuator(actuator_root, joint_name, hd)

            name_parts.append(f"{joint_name}-{hd.name}")

        variant_tag = "__".join(name_parts)
        out_name = f"ur3_sha_sg_101_variant_{idx:02d}.xml"
        out_path = out_dir / out_name

        save_xml(tree, str(out_path))
        print(f"Saved: {out_path}  ({variant_tag})")

if __name__ == "__main__":
    base_xml = "mjmodel.xml"
    out_xml = "ur3_hd_single_variant.xml"
    output_dir = "ur3_hd_variants"
    # build_one_variant(base_xml, out_xml) #если  нужно без перебора оригинально подобранные
    generate_all_variants(base_xml, output_dir)
