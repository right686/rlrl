import sys
import os
import torch
import numpy as np
import random
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from utils.record_data import create_train_reward_path
from envs.ParallelRobotEnv import ParallelEnv
from envs.MultiMorphologyEnv import MultiMorphologyEnv
from stable_baselines3 import PPO
from typing import Optional
from torch import nn
from utils import tool,record_data

# 轨迹参数
DIRECTION = "z"
AMPLITUDE = 0.05
OMEGA = np.pi
DURATION = 4.0
T_THEORY = np.linspace(0.0, DURATION, 4000)


def callback(output_record_path):
    record_callback = record_data.RecordCallback(csv_path=output_record_path, window_size=100)
    return record_callback

def train_new_model(new_model_path, test_model_path, output_policy_path, output_tensorboard_path, output_record_path, generation,
                    n_envs=24, visualize: bool = False, resume_from: Optional[str] = None,
                    total_timesteps: int = 5_000_000,
                    seed: int = 0,
                    design_parameters=None,
                    ):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # 创建子代训练信息文件
    train_information_path = create_train_reward_path(output_record_path)
    # 创建子代的策略保存文件
    policy_path = os.path.join(output_policy_path, str(generation))

    ENV_KWARGS = {'visualize': visualize, 'design_parameters': design_parameters}

    # 加载环境
    env = make_vec_env(
        env_id=lambda: ParallelEnv(** ENV_KWARGS, model_path=new_model_path, test_model_path=test_model_path),
        n_envs=n_envs,
        seed=seed,
        vec_env_cls=SubprocVecEnv,
        vec_env_kwargs={"start_method": "fork"}
    )
    
    model = None
    if resume_from is not None:
        resume_path = resume_from if resume_from.endswith(".zip") else f"{resume_from}.zip"
        if os.path.exists(resume_path):
            print(f"[train_ppo] warm-starting from {resume_path}")
            model = PPO.load(resume_from, env=env)
        else:
            print(f"[train_ppo] resume model not found: {resume_path}, starting from scratch.")
            resume_from = None

    if model is None:
        POLICY_KWARGS = dict(
                activation_fn=nn.ReLU,
                net_arch=dict(pi=[256, 128], vf=[256, 128])
            )
        model = PPO(
                    policy="MlpPolicy",
                    env=env,
                    policy_kwargs=POLICY_KWARGS,
                    verbose=1,
                    n_steps=2048,          
                    batch_size=2048,       
                    n_epochs=10,           
                    gamma=0.99,
                    learning_rate=5e-4,
                    device="cuda" if torch.cuda.is_available() else "cpu",
                    tensorboard_log=output_tensorboard_path
                )

    print(f"并行环境数: {n_envs}")
    call = callback(train_information_path)
    model.learn(
        total_timesteps=total_timesteps,
        progress_bar=False,
        callback=call,
        reset_num_timesteps=True,
    )

    os.makedirs(os.path.dirname(os.path.abspath(policy_path)), exist_ok=True)
    model.save(policy_path)
    env.close()
    print(f"模型已保存至: {policy_path}")
    print(f"训练数据已经保存至: {train_information_path}")
    
    return policy_path


def train_universal_policy(
    model_paths,
    test_model_paths,
    design_parameters,
    output_policy_path,
    output_tensorboard_path,
    total_timesteps: int = 20_000_000,
    seed: int = 0,
    n_envs: int = 16,
):
    if not (len(model_paths) == len(test_model_paths) == len(design_parameters)):
        raise ValueError("model_paths, test_model_paths and design_parameters must have the same length")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    worker_count = min(n_envs, len(model_paths))
    if worker_count == len(model_paths):
        env_fns = [
            lambda i=i: ParallelEnv(
                visualize=False,
                model_path=model_paths[i],
                test_model_path=test_model_paths[i],
                design_parameters=design_parameters[i],
            )
            for i in range(len(model_paths))
        ]
    else:
        design_chunks = np.array_split(np.arange(len(model_paths)), worker_count)
        env_fns = [
            lambda indices=indices: MultiMorphologyEnv(
                model_paths=[model_paths[i] for i in indices],
                test_model_paths=[test_model_paths[i] for i in indices],
                design_parameters=[design_parameters[i] for i in indices],
                visualize=False,
            )
            for indices in design_chunks
        ]
    env = SubprocVecEnv(env_fns, start_method="fork")
    print(f"通用策略并行环境数: {env.num_envs}, 覆盖构型数: {len(model_paths)}")

    policy_kwargs = dict(
        activation_fn=nn.ReLU,
        net_arch=dict(pi=[256, 128], vf=[256, 128]),
    )
    model = PPO(
        policy="MlpPolicy",
        env=env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        n_steps=2048,
        batch_size=2048,
        n_epochs=10,
        gamma=0.99,
        learning_rate=5e-4,
        device="cuda" if torch.cuda.is_available() else "cpu",
        tensorboard_log=output_tensorboard_path,
    )
    model.learn(total_timesteps=total_timesteps, progress_bar=False)

    os.makedirs(os.path.dirname(os.path.abspath(output_policy_path)), exist_ok=True)
    model.save(output_policy_path)
    env.close()

    print(f"通用策略已保存至: {output_policy_path}")
    return output_policy_path


def test_model(new_model_path, test_model_path, output_policy_path, output_record_path, generation, total_episodes=1, seed=0, design_parameters=None):
    np.random.seed(seed)

    env = ParallelEnv(
        model_path=new_model_path,
        test_model_path=test_model_path,
        visualize=False,
        direction=DIRECTION,
        amplitude=AMPLITUDE,
        omega=OMEGA,
        design_parameters=design_parameters,
    )
    model = PPO.load(output_policy_path, env=env)

    # 记录的数据存储区
    # 时间记录器
    time_hist = []
    # 轨迹位置记录器
    pos_hist = []
    euler_hist = []
    # 追踪误差记录器
    residual_hist = []
    # 执行器实际输出力记录器
    actuator_force_hist = []
    # 执行器滑轨位置记录器
    actuator_pos_hist = []
    # 执行器驱动速度记录器
    actutor_vel_hist = []
    # 执行器驱动加速度记录器
    actutor_acc_hist = []

    for ep in range(total_episodes):
            obs, _ = env.reset(seed=seed + ep)
            done = False

            while not done:
                time_hist.append(env.time)

                pos_hist.append(env.data.sensor("center_pos").data.copy())
                euler_hist.append(tool.quat_to_euler_xyz(env.data.sensor("center_quat").data.copy()))
                
                action, _ = model.predict(obs, deterministic=True)
                residual_hist.append(np.asarray(action, dtype=np.float32).copy())
            
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated  
                actuator_force_hist.append(np.asarray(env.data.actuator_force, dtype=np.float32).copy())
                actuator_pos_hist.append(np.asarray(env.data.qpos[env.slide_qpos_adr], dtype=np.float32).copy())
                actutor_vel_hist.append(np.asarray(env.data.qvel[env.slide_dof_adr], dtype=np.float32).copy())
                actutor_acc_hist.append(np.asarray(env.data.qacc[env.slide_dof_adr], dtype=np.float32).copy())
    env.close()

    os.makedirs(output_record_path, exist_ok=True)
    record_path = os.path.join(output_record_path, f"test_generation_{generation}.npz")
    np.savez_compressed(
        record_path,
        time=np.asarray(time_hist, dtype=np.float32),
        pos=np.asarray(pos_hist, dtype=np.float32),
        euler=np.asarray(euler_hist, dtype=np.float32),
        residual=np.asarray(residual_hist, dtype=np.float32),
        actuator_force=np.asarray(actuator_force_hist, dtype=np.float32),
        actuator_pos=np.asarray(actuator_pos_hist, dtype=np.float32),
        actuator_vel=np.asarray(actutor_vel_hist, dtype=np.float32),
        actuator_acc=np.asarray(actutor_acc_hist, dtype=np.float32),
        seed=np.asarray([seed], dtype=np.int64),
    )

    return (
        np.asarray(actuator_force_hist, dtype=np.float32),
        np.asarray(actutor_vel_hist, dtype=np.float32),
    )
