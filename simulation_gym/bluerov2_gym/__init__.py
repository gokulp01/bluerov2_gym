from gymnasium.envs.registration import register

register(
    id="BlueRov-v0",
    entry_point="bluerov2_gym.envs:BlueRov",  # Note the updated entry_point
    max_episode_steps=100,
)
