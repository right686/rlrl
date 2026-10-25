import numpy as np  
import cma
import os
import random
import csv
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

from utils.generate_new_models import (
    apply_leg_posture,
    compute_leg_posture,
    generate_model,
    generate_test_model,
)
from utils.design_utils import (
    body_parameter_range,
    denormalize_angle_variables,
    design_from_solution,
)
from train_inner import train_new_model, test_model


task_object = 'A'

# 总开关
SEED = 42
TOTAL_TIMESTEPS = int(os.environ.get("TOTAL_TIMESTEPS", "50_000_000"))
N_ENVS_PER_SAMPLE = int(os.environ.get("N_ENVS_PER_SAMPLE", "8"))
PARALLEL_SAMPLES = True
MAX_SAMPLE_WORKERS = 1
MAX_GENERATIONS = int(os.environ.get("MAX_GENERATIONS", "2"))


def make_seed(generation, sample_index):
    return SEED + generation * 1000 + sample_index


def evolution_log_path():
    return os.path.join(
        os.path.abspath(os.path.dirname(__file__)),
        "output",
        "zsin_0",
        "evolution_log.csv",
    )


def record_evolution_sample(generation, sample_index, solution, design, objective):
    log_path = evolution_log_path()
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    solution = np.asarray(solution, dtype=np.float64).reshape(-1).tolist()
    design = np.asarray(design, dtype=np.float64).reshape(-1).tolist()
    row = [generation, sample_index, objective] + solution + design
    header = (
        ["generation", "sample_index", "objective"]
        + [f"x{i}" for i in range(len(solution))]
        + [f"design_{i}" for i in range(len(design))]
    )

    write_header = not os.path.exists(log_path) or os.path.getsize(log_path) == 0
    with open(log_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(header)
        writer.writerow(row)

def evaluate_sample(args):
    sample_index, solution, generation, universal_policy_path = args
    design = design_from_solution(solution)
    kuopu_parameter = denormalize_angle_variables(np.asarray(solution[5:9], dtype=np.float64))

    output_model_path, output_policy_path, output_record_path, output_tensorboard_path = save_new_path(
        generation=generation,
        sample_index=sample_index,
    )

    new_model_path = generate_model(design, output_model_path, generation)
    test_model_path = generate_test_model(design, output_model_path, generation)
    posture = compute_leg_posture(test_model_path, kuopu_parameter)
    apply_leg_posture(test_model_path, posture)
    apply_leg_posture(new_model_path, posture)
    seed = make_seed(generation, sample_index)
    control_model_path = train_new_model(
        new_model_path = new_model_path,
        test_model_path = test_model_path,
        output_policy_path = output_policy_path,
        output_tensorboard_path = output_tensorboard_path,
        output_record_path = output_record_path,
        generation = generation,
        n_envs=N_ENVS_PER_SAMPLE,
        total_timesteps=TOTAL_TIMESTEPS,
        seed=seed,
        resume_from=universal_policy_path,
        design_parameters=design,
    )

    force, v = test_model(
        new_model_path=new_model_path,
        test_model_path=test_model_path,
        output_policy_path=control_model_path,
        output_record_path=output_record_path,
        generation=generation,
        seed=seed,
        design_parameters=design,
    )
    objective = float(compute_object_func(task_object, force, v))
    physical_solution = np.asarray(solution, dtype=np.float64).copy()
    physical_solution[5:9] = kuopu_parameter
    record_evolution_sample(generation, sample_index, physical_solution, design, objective)
    return objective

def save_new_path(generation, sample_index):
    root_path = os.path.abspath(os.path.dirname(__file__))
    save_path = os.path.join(root_path, "output")
    new_path = os.path.join(save_path, "zsin_0")

    sample_dir = f"{generation}_{sample_index}"
    output_model_path = os.path.join(new_path, "model", sample_dir)
    output_policy_path = os.path.join(new_path, "policy", sample_dir)
    output_Record_path = os.path.join(new_path, "record", str(generation), f"sample_{sample_index}")
    output_tensorboard_path = os.path.join(new_path, "tensorboard", str(generation), f"sample_{sample_index}")
    output_record_path = output_Record_path

    os.makedirs(output_model_path, exist_ok=True)
    os.makedirs(output_policy_path, exist_ok=True)
    os.makedirs(output_record_path, exist_ok=True)
    os.makedirs(output_tensorboard_path, exist_ok=True)

    return output_model_path, output_policy_path, output_record_path, output_tensorboard_path

def compute_object_func(task: str, force: np.ndarray, actuator_v: np.ndarray, dt: float = 0.001):
    '''
    计算目标函数
    (A:能量损耗,
     B:工作空间大小,
    )
    '''
    force = np.asarray(force, dtype=np.float64)
    actuator_v = np.asarray(actuator_v, dtype=np.float64)

    if force.shape != actuator_v.shape:
        raise ValueError(f"force and actuator_v shape mismatch: {force.shape} vs {actuator_v.shape}")

    if task == 'A':
        power = np.abs(force * actuator_v)
        return float(np.sum(power) * dt)

    raise ValueError(f"Unknown task: {task}")

def initial_alg_parameter():
    '''
    初始化算法参数
    '''
    pop_size = 4
    sigma_fac = 1
    sigma0 = 0.5
    return pop_size, sigma_fac, sigma0

def main_opt():
    '''
    优化入口
    '''
    random.seed(SEED)
    np.random.seed(SEED)

    pole_r_range, pole_length_range, motion_platform_range, polar_range_up, azimuth_range_up, polar_range_down, azimuth_range_down = body_parameter_range()
    pop_size, sigma_fac, sigma0 = initial_alg_parameter()
    x0 = [1, 1, 1, 1, 1, 0.5, 0.5, 0.5, 0.5]
    es = cma.CMAEvolutionStrategy(
        x0,
        sigma0,
        inopts={
            'seed': SEED,
            'popsize': pop_size,
            'CMA_mu': pop_size,
            'CMA_mirrors': 1,
            'bounds': [
                [pole_r_range[0], pole_r_range[0], pole_length_range[0], pole_length_range[0], motion_platform_range[0], 0.0, 0.0, 0.0, 0.0],
                [pole_r_range[1], pole_r_range[1], pole_length_range[1], pole_length_range[1], motion_platform_range[1], 1.0, 1.0, 1.0, 1.0],
            ],
        },
    )

    executor = None
    if PARALLEL_SAMPLES and pop_size > 1:
        executor = ProcessPoolExecutor(
            max_workers=min(pop_size, MAX_SAMPLE_WORKERS),
            mp_context=mp.get_context("spawn"),
        )

    root_path = os.path.abspath(os.path.dirname(__file__))
    default_policy_path = os.path.join(
        root_path, "output", "zsin_0", "pretrain", "universal_policy"
    )
    configured_policy = os.environ.get("UNIVERSAL_POLICY_PATH")
    universal_policy_path = None
    if configured_policy:
        policy_zip = (
            configured_policy
            if configured_policy.endswith(".zip")
            else f"{configured_policy}.zip"
        )
        if os.path.exists(policy_zip):
            universal_policy_path = configured_policy
        else:
            print(f"[search] universal policy not found: {policy_zip}")
    elif os.path.exists(f"{default_policy_path}.zip"):
        universal_policy_path = default_policy_path
    else:
        print("[search] universal policy not found; samples will train from scratch")

    generation = 0
    while not es.stop():
        solutions = es.ask(number=pop_size)
        tasks = [
            (
                sample_index,
                np.asarray(x, dtype=np.float64),
                generation,
                universal_policy_path,
            )
            for sample_index, x in enumerate(solutions)
        ]

        if executor is not None:
            fitnesses = list(executor.map(evaluate_sample, tasks))
        else:
            fitnesses = [evaluate_sample(task) for task in tasks]

        es.tell(solutions, fitnesses)

        generation += 1
        print(f"Gen {generation:4d} | best = {es.result.fbest:.6e}")

        if generation >= MAX_GENERATIONS:
            break

    if executor is not None:
        executor.shutdown()

    print("\n最优解 :", es.result.xbest)
    print("最优值 :", es.result.fbest)
    print("评估次数:", es.result.evaluations)

if __name__ == '__main__':
    main_opt()
