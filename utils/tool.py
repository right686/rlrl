import numpy as np
from scipy.spatial.transform import Rotation as R

def quat_to_euler_xyz(quat):
    """MuJoCo 的 [w,x,y,z] → scipy 的 [x,y,z,w] → xyz 欧拉角"""
    quat = np.asarray(quat, dtype=np.float64).reshape(4)
    scipy_quat = np.array([quat[1], quat[2], quat[3], quat[0]])
    return R.from_quat(scipy_quat).as_euler("xyz", degrees=False)
