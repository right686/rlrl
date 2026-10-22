import sys
import re
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from envs.ParallelRobotEnv import ParallelEnv
from stable_baselines3 import PPO


task = "zsin_0"
direction = "z"
amplitude = 0.05
omega = np.pi

record_root = project_root / "output" / task / "record"
model_root = project_root / "output" / task / "model"
policy_root = project_root / "output" / task / "policy"
replay_root = project_root / "output" / task / "replay"
figure_root = project_root / "figure" / task / "physical"

axis_names = ("x", "y", "z", "rx", "ry", "rz")
actuator_names = tuple(f"a{i}" for i in range(6))


def numeric_children(path, prefix=None):
    children = []
    if not path.exists():
        return []
    for child in path.iterdir():
        if not child.is_dir():
            continue
        if prefix is not None:
            if not child.name.startswith(prefix):
                continue
            suffix = child.name[len(prefix):]
        else:
            suffix = child.name
        if suffix.isdigit():
            children.append((int(suffix), child))
    return [child for _, child in sorted(children)]


def compute_energy(npz_path):
    with np.load(npz_path) as data:
        force = np.asarray(data["actuator_force"], dtype=np.float64)
        velocity = np.asarray(data["actuator_vel"], dtype=np.float64)
    return float(np.sum(np.abs(force * velocity)) * 0.001)


def select_best_sample(generation_dir):
    candidates = []
    for sample_dir in numeric_children(generation_dir, prefix="sample_"):
        sample = int(sample_dir.name.split("_")[1])
        npz_files = sorted(sample_dir.glob("test_generation_*.npz"))
        if not npz_files:
            continue
        energy = compute_energy(npz_files[0])
        candidates.append((energy, sample, npz_files[0]))
    if not candidates:
        return None
    return min(candidates, key=lambda item: item[0])


def load_trial(npz_path):
    with np.load(npz_path) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def trial_field(data, key):
    if key not in data:
        raise KeyError(f"missing field '{key}' in trial data")
    values = np.asarray(data[key], dtype=np.float64)
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    return values


def wrap_angle(values):
    return (values + np.pi) % (2.0 * np.pi) - np.pi


def prepare_model_path(model_path):
    mesh_dir = (project_root / "xml" / "stl").resolve().as_posix()
    text = model_path.read_text(encoding="utf-8")
    text = re.sub(
        r'(<compiler\b[^>]*\bmeshdir=")[^"]*(")',
        lambda match: f"{match.group(1)}{mesh_dir}{match.group(2)}",
        text,
    )
    temp_path = Path(tempfile.gettempdir()) / f"rlrl_physic_{model_path.parent.name}_{model_path.name}"
    temp_path.write_text(text, encoding="utf-8")
    return temp_path


def test_model_path_for(model_path):
    test_model_path = model_path.parent / f"{model_path.stem}test.xml"
    if not test_model_path.exists():
        raise FileNotFoundError(f"missing test model {test_model_path}")
    return test_model_path


def build_target(model_path, time):
    loadable_model_path = prepare_model_path(model_path)
    loadable_test_model_path = prepare_model_path(test_model_path_for(model_path))
    env = None
    try:
        env = ParallelEnv(
            model_path=str(loadable_model_path),
            test_model_path=str(loadable_test_model_path),
            visualize=False,
            direction=direction,
            amplitude=amplitude,
            omega=omega,
        )
        target = np.asarray(
            [
                env.target_tarject_fun(
                    direction,
                    amplitude,
                    omega,
                    float(t),
                )[0]
                for t in time
            ],
            dtype=np.float64,
        )
    finally:
        if env is not None:
            env.close()
        loadable_model_path.unlink(missing_ok=True)
        loadable_test_model_path.unlink(missing_ok=True)
    return target


def replay_trial(model_path, policy_path, seed, replay_path):
    loadable_model_path = prepare_model_path(model_path)
    loadable_test_model_path = prepare_model_path(test_model_path_for(model_path))
    env = None
    try:
        env = ParallelEnv(
            model_path=str(loadable_model_path),
            test_model_path=str(loadable_test_model_path),
            visualize=False,
            direction=direction,
            amplitude=amplitude,
            omega=omega,
        )
        model = PPO.load(str(policy_path), env=env)
        obs, _ = env.reset(seed=seed)
        done = False

        time_hist = []
        pos_hist = []
        euler_hist = []
        residual_hist = []
        actuator_force_hist = []
        actuator_pos_hist = []
        actuator_vel_hist = []
        actuator_acc_hist = []

        while not done:
            time_hist.append(env.time)
            pos_hist.append(env.data.sensor("center_pos").data.copy())
            euler_hist.append(env._quat2euler(env.data.sensor("center_quat").data.copy()))

            action, _ = model.predict(obs, deterministic=True)
            residual_hist.append(np.asarray(action, dtype=np.float32).copy())
            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            actuator_force_hist.append(np.asarray(env.data.actuator_force, dtype=np.float32).copy())
            actuator_pos_hist.append(np.asarray(env.data.qpos[env.slide_qpos_adr], dtype=np.float32).copy())
            actuator_vel_hist.append(np.asarray(env.data.qvel[env.slide_dof_adr], dtype=np.float32).copy())
            actuator_acc_hist.append(np.asarray(env.data.qacc[env.slide_dof_adr], dtype=np.float32).copy())

        replay_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            replay_path,
            time=np.asarray(time_hist, dtype=np.float32),
            pos=np.asarray(pos_hist, dtype=np.float32),
            euler=np.asarray(euler_hist, dtype=np.float32),
            residual=np.asarray(residual_hist, dtype=np.float32),
            actuator_force=np.asarray(actuator_force_hist, dtype=np.float32),
            actuator_pos=np.asarray(actuator_pos_hist, dtype=np.float32),
            actuator_vel=np.asarray(actuator_vel_hist, dtype=np.float32),
            actuator_acc=np.asarray(actuator_acc_hist, dtype=np.float32),
            seed=np.asarray([seed], dtype=np.int64),
        )
        return load_trial(replay_path)
    finally:
        if env is not None:
            env.close()
        loadable_model_path.unlink(missing_ok=True)
        loadable_test_model_path.unlink(missing_ok=True)


def integrate_velocity(time, velocity):
    dt = np.diff(time)
    position = np.zeros_like(velocity, dtype=np.float64)
    if len(time) > 1:
        increments = 0.5 * (velocity[1:] + velocity[:-1]) * dt[:, None]
        position[1:] = np.cumsum(increments, axis=0)
    return position


def save_figure(fig, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    print(f"saved: {output_path}")


def plot_trajectory(time, actual, target, output_path, title):
    fig, axes = plt.subplots(3, 2, figsize=(13, 9), sharex=True)
    for index, ax in enumerate(axes.ravel()):
        ax.plot(time, actual[:, index], color="tab:blue", linewidth=1.8, label="actual")
        ax.plot(time, target[:, index], color="tab:orange", linestyle="--", linewidth=1.5, label="target")
        ax.set_title(axis_names[index])
        ax.set_ylabel("Position [m]" if index < 3 else "Angle [rad]")
        ax.grid(True, alpha=0.25)
        if index == 0:
            ax.legend()
    for ax in axes[-1]:
        ax.set_xlabel("Time [s]")
    fig.suptitle(title, fontsize=14)
    save_figure(fig, output_path)


def plot_error(time, error, output_path, title):
    fig, axes = plt.subplots(3, 2, figsize=(13, 9), sharex=True)
    for index, ax in enumerate(axes.ravel()):
        ax.plot(time, error[:, index], color="tab:red", linewidth=1.8)
        ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
        ax.set_title(axis_names[index])
        ax.set_ylabel("Error [m]" if index < 3 else "Error [rad]")
        ax.grid(True, alpha=0.25)
    for ax in axes[-1]:
        ax.set_xlabel("Time [s]")
    fig.suptitle(title, fontsize=14)
    save_figure(fig, output_path)


def plot_actuator_series(time, values, output_path, title, ylabel):
    fig, axes = plt.subplots(3, 2, figsize=(13, 9), sharex=True)
    for index, ax in enumerate(axes.ravel()):
        ax.plot(time, values[:, index], linewidth=1.5)
        ax.set_title(actuator_names[index])
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
    for ax in axes[-1]:
        ax.set_xlabel("Time [s]")
    fig.suptitle(title, fontsize=14)
    save_figure(fig, output_path)


def main():
    figure_root.mkdir(parents=True, exist_ok=True)
    for generation_dir in numeric_children(record_root):
        best = select_best_sample(generation_dir)
        if best is None:
            continue

        energy, sample, npz_path = best
        generation = int(generation_dir.name)
        data = load_trial(npz_path)

        model_path = model_root / f"{generation}_{sample}" / f"{generation + 1}.xml"
        if not model_path.exists():
            print(f"skip generation {generation}: missing model {model_path}")
            continue

        if "actuator_pos" not in data:
            replay_path = replay_root / f"generation_{generation:02d}_sample_{sample}.npz"
            try:
                if replay_path.exists():
                    data = load_trial(replay_path)
                else:
                    policy_path = policy_root / f"{generation}_{sample}" / f"{generation}.zip"
                    if not policy_path.exists():
                        raise FileNotFoundError(policy_path)
                    seed = int(np.asarray(data["seed"]).reshape(-1)[0])
                    data = replay_trial(model_path, policy_path, seed, replay_path)
                    print(f"replayed actual qpos: {replay_path}")
            except Exception as exc:
                print(f"replay unavailable for generation {generation}: {exc}")

        time = np.asarray(data["time"], dtype=np.float64)
        position = trial_field(data, "pos")
        euler = trial_field(data, "euler")
        actual = np.concatenate([position, euler], axis=1)
        target = build_target(model_path, time)
        error = target - actual
        error[:, 3:] = wrap_angle(error[:, 3:])

        if "actuator_pos" in data:
            slide_position = trial_field(data, "actuator_pos")
            position_label = "Slide position"
        else:
            slide_position = integrate_velocity(
                time,
                trial_field(data, "actuator_vel"),
            )
            position_label = "Integrated slide position"

        output_dir = figure_root / f"generation_{generation:02d}"
        common_title = (
            f"Generation {generation:02d} | best sample {sample} | "
            f"energy={energy:.6g}"
        )
        plot_trajectory(
            time,
            actual,
            target,
            output_dir / "six_dof_trajectory.png",
            f"{common_title} | trajectory",
        )
        plot_error(
            time,
            error,
            output_dir / "six_dof_tracking_error.png",
            f"{common_title} | tracking error",
        )
        plot_actuator_series(
            time,
            trial_field(data, "actuator_force"),
            output_dir / "actuator_force.png",
            f"{common_title} | actuator force",
            "Force",
        )
        plot_actuator_series(
            time,
            slide_position,
            output_dir / "actuator_slide_position.png",
            f"{common_title} | actuator slide position",
            position_label,
        )
        plot_actuator_series(
            time,
            trial_field(data, "actuator_vel"),
            output_dir / "actuator_velocity.png",
            f"{common_title} | actuator velocity",
            "Velocity",
        )
        plot_actuator_series(
            time,
            trial_field(data, "actuator_acc"),
            output_dir / "actuator_acceleration.png",
            f"{common_title} | actuator acceleration",
            "Acceleration",
        )


if __name__ == "__main__":
    main()
