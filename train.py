import gymnasium as gym
from gymnasium.envs.registration import register
from stable_baselines3 import PPO  # You can change this to other algorithms
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

register(
    id="BlueRov-v0",
    entry_point="bluerov_env:BlueRov",
    max_episode_steps=1000,
)


def make_env():
    env = gym.make("BlueRov-v0")
    env = Monitor(env)
    return env


def train_model():
    env = DummyVecEnv([make_env])
    env = VecNormalize(
        env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0,
    )

    model = PPO(
        "MultiInputPolicy",  # Special policy for dict observations
        env,
        verbose=1,
        tensorboard_log="./bluerov_tensorboard/",
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
    )

    model.learn(total_timesteps=1_000_000)

    model.save("bluerov_ppo")

    env.save("bluerov_vec_normalize.pkl")


def evaluate_model():
    env = DummyVecEnv([make_env])
    env = VecNormalize.load("bluerov_vec_normalize.pkl", env)

    env.training = False
    env.norm_reward = False

    model = PPO.load("bluerov_ppo")

    episodes = 10
    for _ in range(episodes):
        obs = env.reset()
        done = False
        total_reward = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            total_reward += reward

            if done:
                print(f"Episode reward: {total_reward}")
                break


if __name__ == "__main__":
    train_model()

    evaluate_model()
