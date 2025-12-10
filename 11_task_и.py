import mujoco
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

xml_path = "/content/drive/MyDrive/Mecha/urdf/mjmodel.xml"        # your path
model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

# Joint configurations in generalized coordinates
configs = [
    # write your own positions
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, -1.57, 0.0, 0.0],
    [3.14, -1.57, 0.0, -1.57, 0.0, 0.0],
    [1.57, -0.78, 0.0, -1.57, -3.14, 0.0],
    [1.57, 1.57, 1.57, 1.57, 1.57, 1.57],
    [-1.57, 0.9, -2.17, -3.9, 3.14, 5.4],
]

# Число обобщённых координат
nq = model.nq

link_name = "ee_link"
link_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, link_name)

def get_ee_pos(q):
    data.qpos[:] = q
    data.qvel[:] = 0.0
    data.qacc[:] = 0.0
    mujoco.mj_forward(model, data)
    return data.xpos[link_id].copy()  # (3,)


def scalar_quintic(t, T):
    tau = t / T
    tau2 = tau * tau
    tau3 = tau2 * tau
    tau4 = tau3 * tau
    tau5 = tau4 * tau

    s   = 10.0 * tau3 - 15.0 * tau4 + 6.0 * tau5
    sd  = (30.0 * tau2 - 60.0 * tau3 + 30.0 * tau4) / T
    sdd = (60.0 * tau - 180.0 * tau2 + 120.0 * tau3) / (T**2)
    return s, sd, sdd


def cartesian_traj(t, T, p_start, p_goal):
    dp = p_goal - p_start
    s, sd, sdd = scalar_quintic(t, T)
    p = p_start + s * dp
    v = sd * dp
    a = sdd * dp
    return p, v, a


def ik_position(target_pos, q_init, link_id, tol=1e-4, max_iter=50, step_lim=0.2):
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


def joint_velocity_from_cartesian(q, v_des):

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

q_start = np.array(configs[0], dtype=float)
q_goal  = np.array(configs[1], dtype=float)

p_start = get_ee_pos(q_start)
p_goal  = get_ee_pos(q_goal)

print("Start EE pos:", p_start)
print("Goal  EE pos:", p_goal)

v_max = 0.30  # [м/с]

dist = np.linalg.norm(p_goal - p_start)

# Рекомендованная длительность
T = 15.0 / 8.0 * dist / v_max

dt = 0.01
time = np.arange(0.0, T + dt, dt)
N = len(time)

q_traj   = np.zeros((N, nq))
qd_traj  = np.zeros((N, nq))
qdd_traj = np.zeros((N, nq))

q_curr = q_start.copy()

for i, t in enumerate(time):
    p_des, v_des, a_des = cartesian_traj(t, T, p_start, p_goal)
    q_curr = ik_position(p_des, q_curr, link_id)
    q_traj[i] = q_curr

for i, t in enumerate(time):
    _, v_des, _ = cartesian_traj(t, T, p_start, p_goal)
    qd_traj[i] = joint_velocity_from_cartesian(q_traj[i], v_des)

# 3) Ускорения – численная производная от qdot
for j in range(nq):
    qdd_traj[:, j] = np.gradient(qd_traj[:, j], dt, edge_order=2)

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
    ee_positions[i] = get_ee_pos(q_traj[i])

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


T_req_dyn = float(np.max(np.abs(torques)))   # худший момент из динамики
qd_max    = float(np.max(np.abs(qd_traj)))   # худшая угловая скорость [rad/s]

w_req_rpm = qd_max * 60.0 / (2.0 * np.pi)    # требуемая макс. скорость [rpm]

print(f"Worst-case required torque : {T_req_dyn:.3f} Nm")
print(f"Worst-case required speed  : {w_req_rpm:.0f} rpm")

# Здесь можно подставить реальные параметры из даташита мотора
T_stall  = 2.0 * T_req_dyn     # ЗАМЕНИТЬ
T_nom    = 0.6 * T_stall       # ЗАМЕНИТЬ
w_nl_rpm = 1.8 * w_req_rpm     # ЗАМЕНИТЬ

T_required = T_req_dyn

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

# (b) Performance envelope
ax_env = axes2[1]
ax_env.set_title("BLDC Motor Performance Envelope")
ax_env.set_xlabel("Speed [rpm]")
ax_env.set_ylabel("Torque [Nm]")
ax_env.set_xlim(0.0, w_nl_rpm)
ax_env.set_ylim(0.0, 1.05 * T_stall)
ax_env.grid(True)

ax_env.fill_between(w_axis, 0.0, T_nom,
                    alpha=0.3, hatch='///',
                    edgecolor='green', facecolor='none',
                    label="Nominal region")

ax_env.fill_between(w_axis, T_nom, T_stall,
                    alpha=0.3, hatch='\\\\\\',
                    edgecolor='red', facecolor='none',
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

ax_env.scatter([w_req_rpm], [T_required], marker="x", s=60, color="k")
ax_env.annotate("Required\noperating point",
                xy=(w_req_rpm, T_required),
                xytext=(0.45 * w_req_rpm, 0.75 * T_required),
                arrowprops=dict(arrowstyle="->"),
                ha="center")

plt.tight_layout()
plt.show()
