from pathlib import Path
import mujoco
import mujoco.viewer

here = Path(__file__).resolve().parent
# model_path = here / "mjmodel.xml"  
model_path = here / "ur3_hd_single_variant.xml"  

model = mujoco.MjModel.from_xml_path(str(model_path))
data = mujoco.MjData(model)

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()
