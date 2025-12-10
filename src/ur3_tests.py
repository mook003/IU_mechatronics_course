import argparse
import mujoco
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


# ==========================
# Общие вспомогательные функции
# ==========================

def scalar_quintic(t, T):
    """Скалярная квинтическая траектория 0→1 за время T."""
    tau = t / T
    tau2 = tau * tau
    tau3 = tau2 * tau
    tau4 = tau3 * tau
    tau5 = tau4 * tau

    s = 10.0 * tau3 - 15.0 * tau4 + 6.0 * tau5
    sd = (30.0 * tau2 - 60.0 * tau3 + 30.0 * tau4) / T
    sdd = (60.0 * tau - 180.0 * tau2 + 120.0 * tau3) / (T ** 2)
    return s, sd, sdd


def get_ee_pos(model, data, link_id, q):
    """Позиция схвата для данного q."""
    data.qpos[:] = q
    data.qvel[:] = 0.0
    data.qacc[:] = 0.0
    mujoco.mj_forward(model, data)
    return data.xpos[link_id].copy()

def plot_motor_performance(torques, qd_traj):
    """
    Расчёт худшего момента/скорости и построение
    torque–speed и performance envelope.
    Для каждого сустава рисуется trace (|ω_j(t)|, |τ_j(t)|).
    """
    # torques, qd_traj: (N, nq)
    T_abs = np.abs(torques)
    qd_abs = np.abs(qd_traj)

    # Худший случай по всем суставам и времени
    T_req_dyn = float(T_abs.max())           # [Nm]
    qd_max = float(qd_abs.max())             # [rad/s]
    w_req_rpm = qd_max * 60.0 / (2.0 * np.pi)  # [rpm]

    print(f"Worst-case required torque : {T_req_dyn:.3f} Nm")
    print(f"Worst-case required speed  : {w_req_rpm:.0f} rpm")

    # Простейшая "подборка" мотора (место для реальных данных)
    T_stall = 2.0 * T_req_dyn     # ЗАМЕНИТЬ на табличное
    T_nom = 0.6 * T_stall         # ЗАМЕНИТЬ на табличное
    w_nl_rpm = 1.8 * w_req_rpm    # ЗАМЕНИТЬ на табличное

    T_required = T_req_dyn

    # Идеальная линейная кривая мотора
    w_axis = np.linspace(0.0, w_nl_rpm, 200)
    T_axis = T_stall * (1.0 - w_axis / w_nl_rpm)

    fig2, axes2 = plt.subplots(1, 2, figsize=(12, 4))

    # (a) Torque–Speed curve
    ax_ts = axes2[0]
    ax_ts.plot(w_axis, T_axis, linewidth=2)
    ax_ts.scatter([0.0, w_nl_rpm], [T_stall, 0.0])

    ax_ts.set_xlabel("Speed [rpm]")
    ax_ts.set_ylabel("Torque [Nm]")
    ax_ts.set_title("BLDC Motor Torque–Speed Curve")
    ax_ts.grid(True)

    ax_ts.annotate("Stall torque",
                   xy=(0.0, T_stall),
                   xytext=(0.05 * w_nl_rpm, 1.02 * T_stall),
                   arrowprops=dict(arrowstyle="->"))

    ax_ts.annotate("Maximum velocity",
                   xy=(w_nl_rpm, 0.0),
                   xytext=(0.60 * w_nl_rpm, 0.10 * T_stall),
                   ha="center",
                   arrowprops=dict(arrowstyle="->"))

    ax_ts.scatter([w_req_rpm], [T_required], marker="x", s=60)
    ax_ts.annotate("Required\noperating point",
                   xy=(w_req_rpm, T_required),
                   xytext=(0.5 * w_req_rpm, 0.6 * T_required),
                   arrowprops=dict(arrowstyle="->"),
                   ha="center")

    # (b) Performance envelope + трейсы по суставам
    ax_env = axes2[1]
    ax_env.set_title("BLDC Motor Performance Envelope")
    ax_env.set_xlabel("Speed [rpm]")
    ax_env.set_ylabel("Torque [Nm]")
    ax_env.set_xlim(0.0, w_nl_rpm)
    ax_env.set_ylim(0.0, 1.05 * T_stall)
    ax_env.grid(True)

    # Области номинальной / перегрузочной работы
    ax_env.fill_between(w_axis, 0.0, T_nom,
                        alpha=0.3, hatch='///',
                        label="Nominal region")

    ax_env.fill_between(w_axis, T_nom, T_stall,
                        alpha=0.3, hatch='\\\\\\',
                        label="Overload region")

    ax_env.axhline(T_nom, linestyle="--")
    ax_env.axhline(T_stall, linestyle="--")

    ax_env.annotate("Nominal torque",
                    xy=(w_nl_rpm, T_nom),
                    xytext=(0.55 * w_nl_rpm, 1.05 * T_nom),
                    arrowprops=dict(arrowstyle="->"),
                    ha="left")

    ax_env.annotate("Stall torque",
                    xy=(w_nl_rpm, T_stall),
                    xytext=(0.55 * w_nl_rpm, 0.90 * T_stall),
                    arrowprops=dict(arrowstyle="->"),
                    ha="left")

    # Трейсы для каждого сустава: |ω_j(t)| vs |τ_j(t)|
    N, nq = torques.shape
    for j in range(nq):
        w_joint = qd_abs[:, j] * 60.0 / (2.0 * np.pi)  # [rpm]
        T_joint = T_abs[:, j]                          # [Nm]
        ax_env.plot(w_joint, T_joint, linewidth=1, label=f"Joint {j}")

    # Точка худшего режима
    ax_env.scatter([w_req_rpm], [T_required], marker="x", s=60)
    ax_env.annotate("Required\noperating point",
                    xy=(w_req_rpm, T_required),
                    xytext=(0.45 * w_req_rpm, 0.75 * T_required),
                    arrowprops=dict(arrowstyle="->"),
                    ha="center")

    # Легенда: сустава + области
    ax_env.legend(loc="best", fontsize="small")

    plt.tight_layout()
    plt.show()


# ==========================
# Тест 1: статический (с нагрузкой на ee_link)
# ==========================

def run_static_test(model, data, configs, link_name="ee_link", payload_mass=3.0):
    link_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, link_name)

    all_torques = []

    # Добавляем груз
    model.body_mass[link_id] += payload_mass

    # Новый COM
    current_com = model.body_ipos[link_id].copy()
    payload_pos = np.array([0.0, 0.0, 0.05])
    new_com = (current_com * (model.body_mass[link_id] - payload_mass)
               + payload_pos * payload_mass) / model.body_mass[link_id]
    model.body_ipos[link_id] = new_com

    # Доп. инерция
    r2 = np.sum(payload_pos ** 2)
    model.body_inertia[link_id] += np.array([r2, r2, r2]) * payload_mass  # I = mr^2

    # Статический расчет моментов
    for qpos in configs:
        data.qpos[:] = qpos
        data.qvel[:] = 0.0
        data.qacc[:] = 0.0
        mujoco.mj_inverse(model, data)
        all_torques.append(data.qfrc_inverse.copy())

    all_torques = np.array(all_torques)  # (num_configs, num_joints)
    max_torques = np.max(np.abs(all_torques), axis=0)

    for i, torque in enumerate(max_torques):
        print(f"Joint {i}: Max static torque = {torque:.5f} Nm")

    plt.figure(figsize=(10, 6))
    sns.violinplot(data=all_torques)
    plt.ylabel('Torque [Nm]')
    plt.xlabel('Joint index')
    plt.title('Static torques for multiple configurations (with payload)')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# ==========================
# Тест 2: динамика по декартовой траектории
# ==========================

def ik_position(model, data, target_pos, q_init, link_id,
                tol=1e-4, max_iter=50, step_lim=0.2):
    """Простая позиционная IK по Якобиану."""
    nq = model.nq
    q = q_init.copy()

    for _ in range(max_iter):
        data.qpos[:] = q
        data.qvel[:] = 0.0
        data.qacc[:] = 0.0
        mujoco.mj_forward(model, data)

        p_cur = data.xpos[link_id].copy()
        err = target_pos - p_cur
        if np.linalg.norm(err) < tol:
            break

        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        mujoco.mj_jacBody(model, data, jacp, jacr, link_id)

        J = jacp[:, :nq]  # (3, nq)
        dq = np.linalg.pinv(J) @ err
        dq = np.clip(dq, -step_lim, step_lim)
        q += dq

    return q


def joint_velocity_from_cartesian(model, data, q, link_id, v_des):
    """qdot = J^+ * v_des."""
    nq = model.nq

    data.qpos[:] = q
    data.qvel[:] = 0.0
    data.qacc[:] = 0.0
    mujoco.mj_forward(model, data)

    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    mujoco.mj_jacBody(model, data, jacp, jacr, link_id)

    J = jacp[:, :nq]
    qdot = np.linalg.pinv(J) @ v_des
    return qdot


def cartesian_traj(t, T, p_start, p_goal):
    """Квинтик по прямой в декартовом пространстве."""
    dp = p_goal - p_start
    s, sd, sdd = scalar_quintic(t, T)
    p = p_start + s * dp
    v = sd * dp
    a = sdd * dp
    return p, v, a


def run_dynamic_cartesian_test(model, data, configs, link_name="ee_link", v_max=0.30):
    nq = model.nq
    link_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, link_name)

    q_start = np.array(configs[0], dtype=float)
    q_goal = np.array(configs[1], dtype=float)

    p_start = get_ee_pos(model, data, link_id, q_start)
    p_goal = get_ee_pos(model, data, link_id, q_goal)

    print("Start EE pos:", p_start)
    print("Goal  EE pos:", p_goal)

    dist = np.linalg.norm(p_goal - p_start)
    T = 15.0 / 8.0 * dist / v_max  # рекомендованная длительность

    dt = 0.01
    time = np.arange(0.0, T + dt, dt)
    N = len(time)

    q_traj = np.zeros((N, nq))
    qd_traj = np.zeros((N, nq))
    qdd_traj = np.zeros((N, nq))

    q_curr = q_start.copy()

    # Позиции через IK
    for i, t in enumerate(time):
        p_des, v_des, a_des = cartesian_traj(t, T, p_start, p_goal)
        q_curr = ik_position(model, data, p_des, q_curr, link_id)
        q_traj[i] = q_curr

    # Скорости через Якобиан
    for i, t in enumerate(time):
        _, v_des, _ = cartesian_traj(t, T, p_start, p_goal)
        qd_traj[i] = joint_velocity_from_cartesian(model, data, q_traj[i], link_id, v_des)

    # Ускорения – численная производная
    for j in range(nq):
        qdd_traj[:, j] = np.gradient(qd_traj[:, j], dt, edge_order=2)

    # Моменты
    torques = np.zeros((N, nq))
    for i in range(N):
        data.qpos[:] = q_traj[i]
        data.qvel[:] = qd_traj[i]
        data.qacc[:] = qdd_traj[i]
        mujoco.mj_inverse(model, data)
        torques[i] = data.qfrc_inverse[:nq].copy()

    max_dyn_torques = np.max(np.abs(torques), axis=0)
    for j, tau in enumerate(max_dyn_torques):
        print(f"Joint {j}: Max dynamic torque (Cartesian test) = {tau:.5f} Nm")

    # 1) Моменты во времени
    plt.figure(figsize=(10, 6))
    for j in range(nq):
        plt.plot(time, torques[:, j], label=f'Joint {j}')
    plt.xlabel('Time [s]')
    plt.ylabel('Torque [Nm]')
    plt.title('Dynamic joint torques for Cartesian quintic EE trajectory')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 2) Реальная траектория захвата
    ee_positions = np.zeros((N, 3))
    for i in range(N):
        ee_positions[i] = get_ee_pos(model, data, link_id, q_traj[i])

    fig = plt.figure(figsize=(6, 6))
    ax3d = fig.add_subplot(111, projection='3d')
    ax3d.plot(ee_positions[:, 0], ee_positions[:, 1], ee_positions[:, 2], label='EE path')
    ax3d.scatter(p_start[0], p_start[1], p_start[2], marker='o', label='start')
    ax3d.scatter(p_goal[0], p_goal[1], p_goal[2], marker='^', label='goal')
    ax3d.set_xlabel('X [m]')
    ax3d.set_ylabel('Y [m]')
    ax3d.set_zlabel('Z [m]')
    ax3d.set_title('EE Cartesian trajectory (quintic along straight line)')
    ax3d.legend()
    plt.tight_layout()
    plt.show()

    # 3) Распределение моментов
    plt.figure(figsize=(8, 5))
    sns.violinplot(data=torques)
    plt.xlabel('Joint index')
    plt.ylabel('Torque [Nm]')
    plt.title('Dynamic torques distribution (Cartesian test)')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # 4) Характеристика мотора
    plot_motor_performance(torques, qd_traj)


# ==========================
# Тест 3: динамика в joint space (квинтик между двумя конфигурациями)
# ==========================

def run_dynamic_joint_test(model, data, configs, link_name="ee_link", w_max=1.0):
    nq = model.nq
    link_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, link_name)

    q_start = np.array(configs[0], dtype=float)
    q_goal = np.array(configs[1], dtype=float)

    dq = q_goal - q_start
    dist_joint = np.linalg.norm(dq)
    T = 15.0 / 8.0 * dist_joint / w_max if dist_joint > 1e-6 else 1.0

    dt = 0.01
    time = np.arange(0.0, T + dt, dt)
    N = len(time)

    q_traj = np.zeros((N, nq))
    qd_traj = np.zeros((N, nq))
    qdd_traj = np.zeros((N, nq))

    for i, t in enumerate(time):
        s, sd, sdd = scalar_quintic(t, T)
        q_traj[i] = q_start + s * dq
        qd_traj[i] = sd * dq
        qdd_traj[i] = sdd * dq

    torques = np.zeros((N, nq))
    for i in range(N):
        data.qpos[:] = q_traj[i]
        data.qvel[:] = qd_traj[i]
        data.qacc[:] = qdd_traj[i]
        mujoco.mj_inverse(model, data)
        torques[i] = data.qfrc_inverse[:nq].copy()

    max_dyn_torques = np.max(np.abs(torques), axis=0)
    for j, tau in enumerate(max_dyn_torques):
        print(f"Joint {j}: Max dynamic torque (Joint-space test) = {tau:.5f} Nm")

    # 1) Моменты во времени
    plt.figure(figsize=(10, 6))
    for j in range(nq):
        plt.plot(time, torques[:, j], label=f'Joint {j}')
    plt.xlabel('Time [s]')
    plt.ylabel('Torque [Nm]')
    plt.title('Dynamic joint torques for joint-space quintic trajectory')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 2) ЭЭ траектория (для наглядности)
    ee_positions = np.zeros((N, 3))
    for i in range(N):
        ee_positions[i] = get_ee_pos(model, data, link_id, q_traj[i])

    fig = plt.figure(figsize=(6, 6))
    ax3d = fig.add_subplot(111, projection='3d')
    ax3d.plot(ee_positions[:, 0], ee_positions[:, 1], ee_positions[:, 2], label='EE path')
    ax3d.scatter(ee_positions[0, 0], ee_positions[0, 1], ee_positions[0, 2],
                 marker='o', label='start')
    ax3d.scatter(ee_positions[-1, 0], ee_positions[-1, 1], ee_positions[-1, 2],
                 marker='^', label='goal')
    ax3d.set_xlabel('X [m]')
    ax3d.set_ylabel('Y [m]')
    ax3d.set_zlabel('Z [m]')
    ax3d.set_title('EE trajectory (joint-space quintic)')
    ax3d.legend()
    plt.tight_layout()
    plt.show()

    # 3) Распределение моментов
    plt.figure(figsize=(8, 5))
    sns.violinplot(data=torques)
    plt.xlabel('Joint index')
    plt.ylabel('Torque [Nm]')
    plt.title('Dynamic torques distribution (Joint-space test)')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # 4) Характеристика мотора
    plot_motor_performance(torques, qd_traj)


# ==========================
# main
# ==========================

def main():
    parser = argparse.ArgumentParser(
        description="UR3 tests: static, dynamic in Cartesian space, dynamic in joint space."
    )
    parser.add_argument(
        "--xml",
        type=str,
        default="models/source/description/universalUR3.xml",
        help="Path to MJCF XML model (default: models/source/description/universalUR3.xml)",
    )
    parser.add_argument(
        "--test",
        type=str,
        choices=["static", "cartesian", "joint"],
        required=True,
        help="Which test to run: static | cartesian | joint",
    )
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_path(args.xml)
    data = mujoco.MjData(model)

    # Общие конфигурации
    configs = [
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, -1.57, 0.0, 0.0],
        [3.14, -1.57, 0.0, -1.57, 0.0, 0.0],
        [1.57, -0.78, 0.0, -1.57, -3.14, 0.0],
        [1.57, 1.57, 1.57, 1.57, 1.57, 1.57],
        [-1.57, 0.9, -2.17, -3.9, 3.14, 5.4],
    ]

    if args.test == "static":
        run_static_test(model, data, configs)
    elif args.test == "cartesian":
        run_dynamic_cartesian_test(model, data, configs)
    elif args.test == "joint":
        run_dynamic_joint_test(model, data, configs)
    else:
        raise ValueError(f"Unknown test: {args.test}")


if __name__ == "__main__":
    main()
