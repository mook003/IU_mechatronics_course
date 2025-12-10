import mujoco
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from lxml import etree

xml_path = "one.xml"        # your path
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

all_torques = []

""" 
# Test without payload
for qpos in configs:
    # Set config
    data.qpos[:] = qpos
    data.qvel[:] = 0.0
    data.qacc[:] = 0.0
    
    # Get inverse dynamics
    mujoco.mj_inverse(model, data)
    
    all_torques.append(data.qfrc_inverse.copy())
 """

# Test with payload (point mass)
link_name = "ee_link"
link_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, link_name)

# Add mass
payload_mass = 3.0  # kg
model.body_mass[link_id] += payload_mass

# Define COM
current_com = model.body_ipos[link_id].copy()
payload_pos = np.array([0.0, 0.0, 0.05])
new_com = (current_com * (model.body_mass[link_id] - payload_mass) + payload_pos * payload_mass) / model.body_mass[link_id]
model.body_ipos[link_id] = new_com

# Define inertia
r2 = np.sum(payload_pos**2)
model.body_inertia[link_id] += np.array([r2, r2, r2]) * payload_mass    # I = mr**2

for qpos in configs:
    data.qpos[:] = qpos
    data.qvel[:] = 0
    data.qacc[:] = 0
    mujoco.mj_inverse(model, data)
    all_torques.append(data.qfrc_inverse.copy())

all_torques = np.array(all_torques)  # shape: (num_configs, num_joints)

max_torques = np.max(np.abs(all_torques), axis=0)
for i, torque in enumerate(max_torques):
    print(f"Joint {i}: Max torque = {torque:.5f} Nm")

plt.figure(figsize=(10,6))
sns.violinplot(data=all_torques)
plt.ylabel('Torque [Nm]')
plt.xlabel('Joint index')
plt.title('Static torques for multiple configurations')
plt.grid(True)
plt.show()