
import pybullet as p
import pybullet_data
import os

p.connect(p.DIRECT)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
# Load the URDF
urdf_path = "pybullet_articubot/assets/ur5_robotiq_85/ur5_robotiq_85.urdf"
pid = p.loadURDF(urdf_path, [0,0,0], useFixedBase=True)

num = p.getNumJoints(pid)
print(f"Num joints: {num}")
for i in range(num):
    info = p.getJointInfo(pid, i)
    # info[1] is name, info[12] is link name
    print(f"ID: {i}, Name: {info[1].decode('utf-8')}, Link: {info[12].decode('utf-8')}")
