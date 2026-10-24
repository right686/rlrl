import os
import sys

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces
from scipy.spatial.transform import Rotation as R

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.dynamic_model import UCUDynamics


_TOP_BODY = "top"
_UPPER_SITES = ("con1", "con2", "con3", "con4", "con5", "con6")
_LOWER_BODIES = (
    "legdown00",
    "legdown10",
    "legdown01",
    "legdown11",
    "legdown02",
    "legdown12",
)
_SLIDE_JOINTS = (
    "slide00",
    "slide10",
    "slide01",
    "slide11",
    "slide02",
    "slide12",
)
_POS_SENSOR = "center_pos"
_QUAT_SENSOR = "center_quat"
_LINVEL_SENSOR = "center_linvel"
_ANGVEL_SENSOR = "center_angvel"
_LIN_ACC_SENSOR = "center_linacc"
_ANG_ACC_SENSOR = "center_angacc"

# 可视化窗口标志事件
def write_flag_file(flag_filename="rl_visu_flag"):
    flag_path = os.path.join("/tmp", flag_filename)
    try:
        with open(flag_path, "w", encoding="utf-8") as f:
            f.write("This is a flag file")
        return True
    except OSError:
        return False

def check_flag_file(flag_filename="rl_visu_flag"):
    flag_path = os.path.join("/tmp", flag_filename)
    return os.path.exists(flag_path)


class ParallelEnv(gym.Env):
    metadata = {"render_modes": ["human"]}
    def __init__(
        self,
        visualize: bool = False,
        model_path: str = None,
        test_model_path: str = None,
        direction: str = "z",
        trigonometric_function: str = "sin",
        amplitude: float = 0.05,
        omega: float = np.pi,
        max_episode_time: float = 4.0,
        use_dynamics_feedforward: bool = True,
        use_pd_feedback: bool = True,
        pd_kp: float = 200.0,
        pd_kd: float = 20.0,
        feedforward_scale: float = 1.0,
        energy_cost_weight: float = 1.0,
        residual_limit: float = 1,
        design_parameters=None,
    ):
        super().__init__()
        # 可视化
        self.visualize = visualize
        self.viewer = None


        # 导入数据给mujoco
        self.model_path = model_path
        self.test_model_path = test_model_path
        if self.test_model_path is None:
            raise ValueError("test_model_path is required for the dynamics model")
        self.model = mujoco.MjModel.from_xml_path(self.model_path)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)
        self.dynamics_model = mujoco.MjModel.from_xml_path(self.test_model_path)
        self.dynamics_data = mujoco.MjData(self.dynamics_model)
        mujoco.mj_forward(self.dynamics_model, self.dynamics_data)

        # 获取信息
        self.top_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, _TOP_BODY)
        self.upper_site_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, name) for name in _UPPER_SITES]
        self.lower_body_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name) for name in _LOWER_BODIES]

        self.residual_limit = residual_limit

        # 动作空间
        self.action_space = spaces.Box(
            low=-residual_limit,
            high=residual_limit,
            shape=(self.model.nu,),
            dtype=np.float32,
        )

        # 当前位姿、速度，目标位姿、速度、加速度，腿长，前馈力，时间，设计参数
        self.design_parameters = (
            np.asarray(design_parameters, dtype=np.float32).reshape(-1)
            if design_parameters is not None
            else np.zeros(0, dtype=np.float32)
        )
        obs_size = 3 + 3 + 3 + 3 + 3 + 3 + 3 + 3 + 3 + 3 + 6 + 6 + 1 + self.design_parameters.size
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_size,),
            dtype=np.float32,
        )
        # 随机数生成器
        self.np_random = np.random.default_rng()

        # 目标函数参数
        self.direction = direction
        self.trigonometric_function = trigonometric_function
        self.amplitude = amplitude
        self.omega = omega
        self.max_episode_time = max_episode_time
        self.use_dynamics_feedforward = use_dynamics_feedforward
        self.use_pd_feedback = use_pd_feedback
        self.time = 0.0

        # 数据缓存区
        self.prev_action = np.zeros(self.model.nu, dtype=np.float32)
        self._snapshot = None
        self._snapshot_time = None
        self._cached_observation = None

        self.neutral_height, self.neutral_qpos = self.compute_motion_platform_height(self.model)
        self.dynamics_neutral_height, self.dynamics_neutral_qpos = self.compute_motion_platform_height(self.dynamics_model)
        self.dynamics_data.qpos[:] = self.dynamics_neutral_qpos
        self.dynamics_data.qvel[:] = 0.0
        mujoco.mj_forward(self.dynamics_model, self.dynamics_data)
        self._apply_neutral_state()

        self.initial_top_pos = self.data.sensor(_POS_SENSOR).data.copy()
        self.slide_qpos_adr = [self.model.jnt_qposadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)]for name in _SLIDE_JOINTS]
        self.slide_dof_adr = [self.model.jnt_dofadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)]for name in _SLIDE_JOINTS]
        self.neutral_lengths = self._read_leg_lengths().copy()

        # PD控制器参数
        self.pd_kp = pd_kp
        self.pd_kd = pd_kd
        self.feedforward_scale = feedforward_scale
        self.energy_cost_weight = energy_cost_weight

        # 奖励函数权重
        self.position_cost_weight = 1.0 # 位置误差
        self.posture_cost_weight = 0.5 # 姿势误差
        self.velocity_cost_weight = 0.2 # 速度跟踪误差
        self.acceleration_cost_weight = 0.05 # 加速度跟踪误差
        #self.residual_cost_weight = 0.01 # 
        self.action_rate_cost_weight = 0.05 # 跟踪衔接丝滑

        # 是否使用动力学前馈力
        self.dynamics_available = True
        self.dynamics = None

        if self.use_dynamics_feedforward or self.use_pd_feedback:
            try:
                self.dynamics = UCUDynamics.from_mujoco(
                    self.dynamics_model,
                    self.dynamics_data,
                    top_joint="S",
                )
                self.dynamics_available = True
            except Exception:
                self.dynamics = None
                print(
                    "[ParallelEnv] UCUDynamics initialization failed; "
                    "falling back to zero dynamics feedforward."
                )

        if self.visualize and not check_flag_file():
            write_flag_file()
            try:
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
                self.viewer.cam.distance = 3.0
                self.viewer.cam.azimuth = 0.0
                self.viewer.cam.elevation = -30.0
                self.viewer.cam.lookat = np.array([0.0, 0.0, self.neutral_height])
            except Exception:
                self.viewer = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        if seed is not None:
            self.np_random = np.random.default_rng(seed)

        self._apply_neutral_state()
        self.time = 0.0
        self.prev_action = np.zeros(self.model.nu, dtype=np.float32)
        self.initial_top_pos = self.data.sensor(_POS_SENSOR).data.copy()
        self._snapshot, self._cached_observation = self._make_snapshot()
        self._snapshot_time = self.time
        return self._cached_observation, {}

    def step(self, action):
        # 动作缩放
        action = np.asarray(action, dtype=np.float32).reshape(self.model.nu)
        action = np.clip(action, self.action_space.low, self.action_space.high)

        # 上一时间点的动作
        previous_action = self.prev_action.copy()
        self.prev_action = action.copy()

        snapshot = self._get_snapshot()

        # 动力学前馈 + PD 反馈 + RL 残差
        feedforward = snapshot["feedforward"]
        pd_torque = self._compute_pd_torque_from(
            snapshot["position"],
            snapshot["rotation"],
            snapshot["target_pose"],
            snapshot["target_vel"],
        ) if self.use_pd_feedback else np.zeros(self.model.nu, dtype=np.float32)
        control = feedforward + pd_torque + action
        control = np.clip(
            control,
            self.model.actuator_ctrlrange[:, 0],
            self.model.actuator_ctrlrange[:, 1],
        )
        self.data.ctrl[:] = control
        
        # 执行仿真
        mujoco.mj_step(self.model, self.data)
        self.time += self.model.opt.timestep

        if self.viewer is not None:
            self.viewer.sync()

        self._snapshot, self._cached_observation = self._make_snapshot()
        self._snapshot_time = self.time
        observation = self._cached_observation
        reward, reward_info = self._compute_reward_from_snapshot(
            action, feedforward, previous_action
        )
        terminated = self.time >= self.max_episode_time
        truncated = False
        info = {**reward_info}
        return observation, float(reward), terminated, truncated, info

    def get_observation(self):
        snapshot = self._get_snapshot()
        return self._build_observation(snapshot)

    def _build_observation(self, snapshot):
        current_pos = snapshot["position"]
        current_euler = snapshot["euler"]
        current_linvel = snapshot["linvel"]
        current_angvel = snapshot["angvel"]
        target_pose = snapshot["target_pose"]
        target_vel = snapshot["target_vel"]
        target_acc = snapshot["target_acc"]
        leg_lengths = snapshot["leg_lengths"]
        feedforward_force = snapshot["feedforward"]

        observation = np.concatenate(
            [
                current_pos,
                current_euler,
                current_linvel,
                current_angvel,
                target_pose[:3],
                target_pose[3:],
                target_vel[:3],
                target_vel[3:],
                target_acc[:3],
                target_acc[3:],
                leg_lengths,
                feedforward_force,
                np.array([self.time], dtype=np.float32),
                self.design_parameters,
            ]
        )
        return observation.astype(np.float32)

    def _get_snapshot(self):
        if self._snapshot is None or self._snapshot_time != self.time:
            self._snapshot, self._cached_observation = self._make_snapshot()
            self._snapshot_time = self.time
        return self._snapshot

    def _make_snapshot(self):
        position, euler, rotation, linvel, angvel = self._read_top_state_full()
        target_pose, target_vel, target_acc = self.target_tarject_fun(
            self.direction,
            self.amplitude,
            self.omega,
            self.time,
        )
        feedforward = self._compute_dynamic_force_from(
            position, rotation, linvel, angvel, target_acc
        )
        leg_lengths = self._read_leg_lengths()
        snapshot = {
            "position": position,
            "euler": euler,
            "rotation": rotation,
            "linvel": linvel,
            "angvel": angvel,
            "target_pose": target_pose,
            "target_vel": target_vel,
            "target_acc": target_acc,
            "feedforward": feedforward,
            "leg_lengths": leg_lengths,
        }
        return snapshot, self._build_observation(snapshot)

    def _compute_dynamic_force_from(
        self, position, rotation, linvel, angvel, target_acc
    ):
        if not self.use_dynamics_feedforward or self.dynamics is None:
            return np.zeros(self.model.nu, dtype=np.float32)

        force = self.dynamics.inverse_dynamics(
            position,
            rotation,
            linvel,
            angvel,
            target_acc[:3],
            target_acc[3:],
        )
        return self.feedforward_scale * np.asarray(force, dtype=np.float32).reshape(self.model.nu)

    def _compute_pd_torque_from(self, position, rotation, target_pose, target_vel):
        if self.dynamics is None:
            return np.zeros(self.model.nu, dtype=np.float32)

        target_rotation = R.from_euler("xyz", target_pose[3:], degrees=False).as_matrix()
        desired_points = target_pose[:3, None] + target_rotation @ self.dynamics.p_local
        desired_lengths = np.linalg.norm(desired_points - self.dynamics.b, axis=0)
        neutral_slide_qpos = np.asarray(self.neutral_qpos[self.slide_qpos_adr], dtype=np.float64)
        desired_qpos = neutral_slide_qpos + desired_lengths - self.neutral_lengths

        jacobian = self._force_jacobian(self.dynamics, position, rotation)
        desired_qvel = jacobian @ target_vel

        current_qpos = np.asarray(self.data.qpos[self.slide_qpos_adr])
        current_qvel = np.asarray(self.data.qvel[self.slide_dof_adr])
        position_error = desired_qpos - current_qpos
        velocity_error = desired_qvel - current_qvel
        return self.pd_kp * position_error + self.pd_kd * velocity_error

    def compute_dynamic_force(self):
        return self._get_snapshot()["feedforward"].copy()

    def compute_pd_torque(self):
        snapshot = self._get_snapshot()
        return self._compute_pd_torque_from(
            snapshot["position"],
            snapshot["rotation"],
            snapshot["target_pose"],
            snapshot["target_vel"],
        )

    def compute_length(self, up: np.ndarray, down: np.ndarray) -> np.ndarray:
        up = np.asarray(up, dtype=np.float64)
        down = np.asarray(down, dtype=np.float64)
        if up.ndim == 1:
            up = up.reshape(3, -1)
        if down.ndim == 1:
            down = down.reshape(3, -1)
        return np.linalg.norm(up - down, axis=0)

    def _force_jacobian(self, dynamics, position, rotation):
        p_i = rotation @ dynamics.p_local
        h = position[:, None] + p_i - dynamics.b
        lengths = np.linalg.norm(h, axis=0)
        n = h / lengths
        jacobian = np.zeros((6, 6), dtype=np.float64)
        for i in range(6):
            jacobian[i, :] = np.concatenate((n[:, i], np.cross(p_i[:, i], n[:, i])))
        return jacobian

    def target_tarject_fun(self, direction: str, amplitude: float, omega: float, t: float):
        position = np.array([0.0, 0.0, self.neutral_height], dtype=np.float64)
        linvel = np.zeros(3, dtype=np.float64)
        linacc = np.zeros(3, dtype=np.float64)
        posture = np.zeros(3, dtype=np.float64)
        angvel = np.zeros(3, dtype=np.float64)
        angacc = np.zeros(3, dtype=np.float64)

        if self.trigonometric_function == "cos":
            value = amplitude * np.cos(omega * t)
            derivative = -amplitude * omega * np.sin(omega * t)
            second_derivative = -amplitude * omega**2 * np.cos(omega * t)
        else:
            value = amplitude * np.sin(omega * t)
            derivative = amplitude * omega * np.cos(omega * t)
            second_derivative = -amplitude * omega**2 * np.sin(omega * t)

        if direction == "x":
            position[0] = value
            linvel[0] = derivative
            linacc[0] = second_derivative
        elif direction == "y":
            position[1] = value
            linvel[1] = derivative
            linacc[1] = second_derivative
        elif direction == "z":
            position[2] += value
            linvel[2] = derivative
            linacc[2] = second_derivative
        elif direction == "rx":
            posture[0] = value
            angvel[0] = derivative
            angacc[0] = second_derivative
        elif direction == "ry":
            posture[1] = value
            angvel[1] = derivative
            angacc[1] = second_derivative
        elif direction == "rz":
            posture[2] = value
            angvel[2] = derivative
            angacc[2] = second_derivative
        else:
            raise ValueError(f"Unsupported direction: {direction}")

        return (
            np.concatenate([position, posture]),
            np.concatenate([linvel, angvel]),
            np.concatenate([linacc, angacc]),
        )

    def tran(self, quat):
        return self._quat2euler(quat)

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    def _read_top_state(self):
        position, euler, _, linvel, angvel = self._read_top_state_full()
        return position, euler, linvel, angvel

    def _read_top_state_full(self):
        position = np.asarray(self.data.sensor(_POS_SENSOR).data, dtype=np.float64)
        quat = np.asarray(self.data.sensor(_QUAT_SENSOR).data, dtype=np.float64)
        euler = self._quat2euler(quat)
        rotation = self._quat_to_rotation_matrix(quat)
        linvel = np.asarray(self.data.sensor(_LINVEL_SENSOR).data, dtype=np.float64)
        angvel = np.asarray(self.data.sensor(_ANGVEL_SENSOR).data, dtype=np.float64)
        return position, euler, rotation, linvel, angvel

    def _read_leg_lengths(self):
        upper = np.column_stack(
            [self.data.site_xpos[site_id] for site_id in self.upper_site_ids]
        )
        lower = np.column_stack(
            [self.data.xpos[body_id] for body_id in self.lower_body_ids]
        )
        return self.compute_length(upper, lower)

    def _quat2euler(self, quat: np.ndarray) -> np.ndarray:
        quat = np.asarray(quat, dtype=np.float64).reshape(4)
        scipy_quat = np.roll(quat, shift=-1)
        return R.from_quat(scipy_quat).as_euler("xyz", degrees=False)

    def _quat_to_rotation_matrix(self, quat: np.ndarray) -> np.ndarray:
        quat = np.asarray(quat, dtype=np.float64).reshape(4)
        scipy_quat = np.roll(quat, shift=-1)
        return R.from_quat(scipy_quat).as_matrix()

    def _read_actual_acceleration(self) -> np.ndarray:
        linear_acceleration = self.data.sensor(_LIN_ACC_SENSOR).data.copy()
        angular_acceleration = self.data.sensor(_ANG_ACC_SENSOR).data.copy()
        return np.concatenate([linear_acceleration, angular_acceleration])

    def _compute_reward(self, action: np.ndarray, feedforward: np.ndarray, previous_action: np.ndarray):
        snapshot = self._get_snapshot()
        return self._compute_reward_from_snapshot(action, feedforward, previous_action, snapshot=snapshot)

    def _compute_reward_from_snapshot(
        self,
        action: np.ndarray,
        feedforward: np.ndarray,
        previous_action: np.ndarray,
        snapshot=None,
    ):
        snapshot = snapshot or self._get_snapshot()
        current_pos = snapshot["position"]
        current_euler = snapshot["euler"]
        current_linvel = snapshot["linvel"]
        current_angvel = snapshot["angvel"]
        target_pose = snapshot["target_pose"]
        target_vel = snapshot["target_vel"]
        target_acc = snapshot["target_acc"]

        position_error = float(np.linalg.norm(target_pose[:3] - current_pos))
        posture_error = self._orientation_error(target_pose[3:], current_euler)
        velocity_error = float(np.linalg.norm(target_vel - np.concatenate([current_linvel, current_angvel])))
        actual_acceleration = self._read_actual_acceleration()
        acceleration_error = float(np.linalg.norm(target_acc - actual_acceleration))
        #residual_cost = float(np.mean(np.square(action)))
        # action_rate_cost = float(np.mean(np.square(action - previous_action)))
        #action_rate_cost = 0
        #reward = (
        #    self.position_cost_weight * np.exp(-8.0 * position_error)
            #+ self.posture_cost_weight * np.exp(-4.0 * posture_error)
        #    + self.velocity_cost_weight * np.exp(-2.0 * velocity_error)
        #    + self.acceleration_cost_weight * np.exp(-0.5 * acceleration_error)
            # - self.residual_cost_weight * residual_cost
        #    - self.action_rate_cost_weight * action_rate_cost
        #)

        position_term = self.position_cost_weight * np.exp(-100.0 * position_error**2)
        posture_term = self.posture_cost_weight * np.exp(-100.0 * posture_error**2)
        velocity_term = self.velocity_cost_weight * np.exp(-5.0 * velocity_error**2)
        actuator_force = np.asarray(self.data.actuator_force, dtype=np.float64)
        actuator_velocity = np.asarray(self.data.qvel[self.slide_dof_adr], dtype=np.float64)
        energy_term = float(np.mean(np.abs(actuator_force * actuator_velocity)))
        reward = (
            position_term
            + posture_term
            + velocity_term
            - self.energy_cost_weight * energy_term
        )

        info = {
            "reward": reward,
            "reward_position": position_term,
            "reward_posture": posture_term,
            "reward_velocity": velocity_term,
            "reward_acceleration": 0.0,
            "reward_energy": -self.energy_cost_weight * energy_term,
            # "reward_residual": -self.residual_cost_weight * residual_cost,
            # "reward_action_rate": -self.action_rate_cost_weight * action_rate_cost,
            "position_error": position_error,
            "posture_error": posture_error,
            "velocity_error": velocity_error,
            "acceleration_error": acceleration_error,
            "feedforward_norm": float(np.linalg.norm(feedforward)),
            "dynamics_available": self.dynamics_available,
        }
        return reward, info

    def _orientation_error(self, target_euler: np.ndarray, current_euler: np.ndarray) -> float:
        target_rot = R.from_euler("xyz", target_euler, degrees=False)
        current_rot = R.from_euler("xyz", current_euler, degrees=False)
        return float((target_rot.inv() * current_rot).magnitude())


    def compute_motion_platform_height(self, model):
        # 仿真一个固定的时间步长，获取平台高度
        # 带入一个静止文件，获取平台高度
        data = mujoco.MjData(model)
        for i in range(1000):
            mujoco.mj_step(model, data)   
        mujoco.mj_step(model, data)
        motion_platform_height = float(data.sensor("center_pos").data[2])
        neutral_qpos = data.qpos.copy()
        mujoco.mj_resetData(model, data)
        return motion_platform_height, neutral_qpos

    def _apply_neutral_state(self):
        if self.neutral_qpos is not None and self.neutral_qpos.size == self.model.nq:
            self.data.qpos[:] = self.neutral_qpos
            self.data.qvel[:] = 0.0
        else:
            mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
