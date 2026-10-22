from gymnasium.envs.registration import register

register(
    id="Stewt-v0",
    entry_point=".envs:StewtEnv",
    max_episode_steps=1000,
)