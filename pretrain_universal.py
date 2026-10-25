import argparse
import os

import numpy as np

from train_inner import train_universal_policy
from utils.design_utils import (
    body_parameter_range,
    denormalize_angle_variables,
    design_from_solution,
)
from utils.generate_new_models import (
    apply_leg_posture,
    compute_leg_posture,
    generate_model,
    generate_test_model,
)


def sample_pretrain_solutions(count, seed):
    pole_r_range, pole_length_range, motion_platform_range, _, _, _, _ = body_parameter_range()
    lows = np.array(
        [
            pole_r_range[0],
            pole_r_range[0],
            pole_length_range[0],
            pole_length_range[0],
            motion_platform_range[0],
            0.0,
            0.0,
            0.0,
            0.0,
        ]
    )
    highs = np.array(
        [
            pole_r_range[1],
            pole_r_range[1],
            pole_length_range[1],
            pole_length_range[1],
            motion_platform_range[1],
            1.0,
            1.0,
            1.0,
            1.0,
        ]
    )
    rng = np.random.default_rng(seed)
    return rng.uniform(lows, highs, size=(count, len(lows)))


def prepare_pretrain_models(pretrain_dir, solutions):
    model_paths = []
    test_model_paths = []
    design_parameters = []

    for index, solution in enumerate(solutions):
        design = design_from_solution(solution)
        model_dir = os.path.join(pretrain_dir, "model", str(index))
        os.makedirs(model_dir, exist_ok=True)

        model_path = generate_model(design, model_dir, index)
        test_model_path = generate_test_model(design, model_dir, index)
        angles = denormalize_angle_variables(solution[5:9])
        posture = compute_leg_posture(test_model_path, angles)
        apply_leg_posture(test_model_path, posture)
        apply_leg_posture(model_path, posture)

        model_paths.append(model_path)
        test_model_paths.append(test_model_path)
        design_parameters.append(np.asarray(design, dtype=np.float32))

    return model_paths, test_model_paths, design_parameters


def main():
    parser = argparse.ArgumentParser(
        description="Train a design-conditioned universal policy across sampled morphologies."
    )
    root_path = os.path.abspath(os.path.dirname(__file__))
    default_output_dir = os.path.join(root_path, "output", "zsin_0", "pretrain")

    parser.add_argument(
        "--designs",
        type=int,
        default=int(os.environ.get("PRETRAIN_DESIGNS", "8")),
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=int(os.environ.get("PRETRAIN_TIMESTEPS", "20_000_000")),
    )
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default=default_output_dir)
    args = parser.parse_args()

    if args.designs <= 0:
        raise ValueError("--designs must be greater than zero")
    if args.workers is None:
        args.workers = min(os.cpu_count() or 16, args.designs)
    if args.workers <= 0:
        raise ValueError("--workers must be greater than zero")

    solutions = sample_pretrain_solutions(args.designs, args.seed)
    model_paths, test_model_paths, design_parameters = prepare_pretrain_models(
        args.output_dir, solutions
    )

    policy_path = os.path.join(args.output_dir, "universal_policy")
    tensorboard_path = os.path.join(args.output_dir, "tensorboard")
    os.makedirs(tensorboard_path, exist_ok=True)

    print(
        f"[pretrain] designs={args.designs}, "
        f"workers={args.workers}, timesteps={args.timesteps}, output={policy_path}"
    )
    train_universal_policy(
        model_paths=model_paths,
        test_model_paths=test_model_paths,
        design_parameters=design_parameters,
        output_policy_path=policy_path,
        output_tensorboard_path=tensorboard_path,
        total_timesteps=args.timesteps,
        seed=args.seed,
        n_envs=args.workers,
    )


if __name__ == "__main__":
    main()
