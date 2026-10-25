import gymnasium as gym
import numpy as np

from envs.ParallelRobotEnv import ParallelEnv


class MultiMorphologyEnv(gym.Env):
    """Selects one fixed morphology at each reset from a local design pool."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        model_paths,
        test_model_paths,
        design_parameters,
        **env_kwargs,
    ):
        super().__init__()
        if not (len(model_paths) == len(test_model_paths) == len(design_parameters)):
            raise ValueError(
                "model_paths, test_model_paths and design_parameters must have the same length"
            )
        if len(model_paths) == 0:
            raise ValueError("MultiMorphologyEnv requires at least one morphology")

        self.envs = [
            ParallelEnv(
                model_path=model_paths[index],
                test_model_path=test_model_paths[index],
                design_parameters=design_parameters[index],
                **env_kwargs,
            )
            for index in range(len(model_paths))
        ]
        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space
        self.np_random = np.random.default_rng()
        self.current_index = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        self.current_index = int(self.np_random.integers(0, len(self.envs)))
        observation, _ = self.envs[self.current_index].reset(seed=seed)
        return observation, {}

    def step(self, action):
        if self.current_index is None:
            raise RuntimeError("MultiMorphologyEnv.reset must be called before step")
        return self.envs[self.current_index].step(action)

    def close(self):
        for env in self.envs:
            env.close()
