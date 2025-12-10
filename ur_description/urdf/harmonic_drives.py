# # harmonic_drives.py
# from dataclasses import dataclass

# @dataclass
# class HarmonicDrive:
#     name: str           # модель актуатора
#     gear_ratio: float   # передаточное число N
#     armature_inertia: float  # эффективная инерция на оси сустава (GD^2/4), kg·m^2
#     mass: float         # масса актуатора, kg
#     radius: float       # радиус цилиндра (для корпуса), m
#     length: float       # длина цилиндра, m
#     max_torque: float   # Max Allowable Torque (берем Maximum Torque из таблицы), N·m


# # --- Конкретные SHA-SG с ratio = 101 ---

# SHA20A_101 = HarmonicDrive(
#     name="SHA20A-101",
#     gear_ratio=101.0,
#     armature_inertia=0.91,   # kg·m^2  (Moment of Inertia GD^2/4, without brake)
#     mass=2.0,                # kg
#     radius=0.047,            # m  (øA/2 ≈ 94/2 мм)
#     length=0.103,            # m  (E ≈ 103 мм)
#     max_torque=107.0,        # N·m (Maximum Torque for ratio 101)
# )

# SHA25A_101 = HarmonicDrive(
#     name="SHA25A-101",
#     gear_ratio=101.0,
#     armature_inertia=2.2,    # kg·m^2
#     mass=5.0,                # kg
#     radius=0.057,            # m (øA/2 ≈ 114/2 мм)
#     length=0.109,            # m (E ≈ 109 мм)
#     max_torque=204.0,        # N·m
# )

# SHA32A_101 = HarmonicDrive(
#     name="SHA32A-101",
#     gear_ratio=101.0,
#     armature_inertia=8.0,    # kg·m^2
#     mass=2.95,               # kg
#     radius=0.073,            # m (øA/2 ≈ 146/2 мм)
#     length=0.125,            # m (E ≈ 125 мм)
#     max_torque=433.0,        # N·m
# )


# # --- Какие актуаторы ставим на какие суставы UR3 ---

# joint_variants = {
#     # плечевые суставы – самые нагруженные → SHA32A
#     "shoulder_pan_joint":  [SHA32A_101],
#     "shoulder_lift_joint": [SHA32A_101],

#     # локоть – средний момент → SHA25A
#     "elbow_joint":         [SHA25A_101],

#     # запястья – малый момент → SHA20A
#     "wrist_1_joint":       [SHA20A_101],
#     "wrist_2_joint":       [SHA20A_101],
#     "wrist_3_joint":       [SHA20A_101],
# }



# harmonic_drives.py
from dataclasses import dataclass

@dataclass
class HarmonicDrive:
    name: str           # модель актуатора
    gear_ratio: float   # передаточное число N
    armature_inertia: float  # инерция на оси сустава (GD^2/4), kg·m^2
    mass: float         # масса актуатора, kg
    radius: float       # радиус цилиндра корпуса, m
    length: float       # длина цилиндра, m
    max_torque: float   # Maximum Torque из каталога, N·m


SHA20A_101 = HarmonicDrive(
    name="SHA20A-101",
    gear_ratio=101.0,
    armature_inertia=0.91,
    mass=2.0,
    radius=0.047,
    length=0.103,
    max_torque=107.0,
)

SHA25A_101 = HarmonicDrive(
    name="SHA25A-101",
    gear_ratio=101.0,
    armature_inertia=2.2,
    mass=5.0,
    radius=0.057,
    length=0.109,
    max_torque=204.0,
)

SHA32A_101 = HarmonicDrive(
    name="SHA32A-101",
    gear_ratio=101.0,
    armature_inertia=8.0,
    mass=2.95,
    radius=0.073,
    length=0.125,
    max_torque=433.0,
)

# На каких суставах какие варианты РЕАЛЬНО допустимы по моменту
joint_variants = {
    # плечи: момент 330 Н·м → только SHA32A-101
    "shoulder_pan_joint":  [SHA32A_101],
    "shoulder_lift_joint": [SHA32A_101],

    # локоть: 150 Н·м → SHA25A и, с запасом, SHA32A
    "elbow_joint":         [SHA25A_101, SHA32A_101],

    # запястья: 54 Н·м → SHA20A и SHA25A
    "wrist_1_joint":       [SHA20A_101, SHA25A_101],
    "wrist_2_joint":       [SHA20A_101, SHA25A_101],
    "wrist_3_joint":       [SHA20A_101, SHA25A_101],
}
