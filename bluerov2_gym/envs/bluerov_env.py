import random
import time

import gymnasium as gym
import meshcat
import meshcat.geometry as g
import meshcat.transformations as tf
import numpy as np
from bluerov_params import *
from gymnasium import spaces
from gymnasium.envs.registration import register
from scipy import stats


class BlueRov(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, render_mode=None):
        super().__init__()

        # Initialize state
        self.state = {
            "x": 0,
            "y": 0,
            "z": 0,
            "theta": 0,
            "vx": 0,
            "vy": 0,
            "vz": 0,
            "omega": 0,
        }

        # Define action and observation spaces
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(4,),  # Changed to 4 to match your step function
            dtype=np.float32,
        )

        # Observation space should match your state dictionary
        self.observation_space = spaces.Dict(
            {
                "x": spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float32),
                "y": spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float32),
                "z": spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float32),
                "theta": spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float32),
                "vx": spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float32),
                "vy": spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float32),
                "vz": spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float32),
                "omega": spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float32),
            }
        )

        self.params = {
            "a_vx": -0.5265,
            "a_vy": -0.5357,
            "a_vz": -1.0653,
            "a_vw": -4.2579,
            "a_vx2": -1.1984,
            "a_vy2": -5.1626,
            "a_vz2": -1.7579e-5,
            "a_vw2": -2.4791e-8,
            "a_vyw": -0.5350,
            "a_vwx": -1.2633,
            "a_vxy": -0.9808,
            "b_vx": 1.2810,
            "b_vy": 0.9512,
            "b_vz": 0.7820,
            "b_vw": 2.6822,
        }

        self.dt = 0.1
        self.disturbance_mean = np.array(
            [-0.01461447, -0.02102184, -0.00115958, 0.05391866]
        )
        self.disturbance_cov = np.array(
            [
                [2.89596342e-2, 5.90296868e-3, -4.22672521e-5, -6.38837738e-3],
                [5.90296868e-3, 2.05937494e-2, 8.59805304e-5, 2.92258483e-3],
                [-4.22672521e-5, 8.59805304e-5, 2.44296056e-3, 1.64117342e-3],
                [-6.38837738e-3, 2.92258483e-3, 1.64117342e-3, 3.71338116e-1],
            ]
        )
        self.disturbance_dist = stats.multivariate_normal(
            mean=self.disturbance_mean, cov=self.disturbance_cov
        )

        self.render_mode = render_mode
        if self.render_mode == "human":
            self.initialize_meshcat()

    def initialize_meshcat(self):
        self.vis = meshcat.Visualizer()
        self.vis.open()

    def reset(self, *, seed=None, options=None):
        # Initialize the RNG if seed is provided
        super().reset(seed=seed)

        # Reset state
        self.state = {
            "x": 0,
            "y": 0,
            "z": 0,
            "theta": 0,
            "vx": 0,
            "vy": 0,
            "vz": 0,
            "omega": 0,
        }

        # Get new disturbances
        self.disturbances = self.disturbance_dist.rvs()

        # Convert state dict to match observation space
        obs = {k: np.array([v], dtype=np.float32) for k, v in self.state.items()}

        return obs, {}

    def step(self, action):
        # Update disturbances
        self.disturbances = self.disturbance_dist.rvs()
        dvx, dvy, dvz, domega = self.disturbances

        # Extract action components
        w_x, w_y, w_z, w_omega = action

        # Update state using your existing physics model
        x, y, z, theta = (
            self.state["x"],
            self.state["y"],
            self.state["z"],
            self.state["theta"],
        )
        vx, vy, vz, omega = (
            self.state["vx"],
            self.state["vy"],
            self.state["vz"],
            self.state["omega"],
        )

        # Position updates
        self.state["x"] += (vx * np.cos(theta) - vy * np.sin(theta)) * self.dt
        self.state["y"] += (vx * np.sin(theta) + vy * np.cos(theta)) * self.dt
        self.state["z"] += vz * self.dt
        self.state["theta"] += omega * self.dt

        # Velocity updates
        self.state["vx"] += (
            self.params["a_vx"] * vx
            + self.params["a_vx2"] * vx * abs(vx)
            + self.params["a_vyw"] * vy * omega
            + self.params["b_vx"] * w_x
            + dvx
        ) * self.dt

        self.state["vy"] += (
            self.params["a_vy"] * vy
            + self.params["a_vy2"] * vy * abs(vy)
            + self.params["a_vwx"] * vx * omega
            + self.params["b_vy"] * w_y
            + dvy
        ) * self.dt

        self.state["vz"] += (
            self.params["a_vz"] * vz
            + self.params["a_vz2"] * vz * abs(vz)
            + self.params["b_vz"] * w_z
            + dvz
        ) * self.dt

        self.state["omega"] += (
            self.params["a_vw"] * omega
            + self.params["a_vw2"] * omega * abs(omega)
            + self.params["a_vxy"] * vx * vy
            + self.params["b_vw"] * w_omega
            + domega
        ) * self.dt

        # Convert state dict to match observation space
        obs = {k: np.array([v], dtype=np.float32) for k, v in self.state.items()}

        # Calculate reward
        reward = self.reward_fn(obs)

        # Determine if episode should end
        terminated = False
        # Example conditions:
        if abs(self.state["z"]) > 10.0:  # Too deep/high
            terminated = True
        if abs(self.state["x"]) > 15.0 or abs(self.state["y"]) > 15.0:  # Out of bounds
            terminated = True

        truncated = False  # Add your truncation conditions

        return obs, reward, terminated, truncated, {}

    def reward_fn(self, obs):
        """Example reward function"""
        # Position error
        position_error = np.sqrt(obs["x"][0] ** 2 + obs["y"][0] ** 2 + obs["z"][0] ** 2)

        # Velocity penalty
        velocity_penalty = np.sqrt(
            obs["vx"][0] ** 2 + obs["vy"][0] ** 2 + obs["vz"][0] ** 2
        )

        # Orientation error
        orientation_error = abs(obs["theta"][0])

        # Combined reward
        reward = -(
            1.0 * position_error  # Weight for position error
            + 0.1 * velocity_penalty  # Weight for velocity
            + 0.5 * orientation_error  # Weight for orientation
        )

        return reward

    def render(self):
        if self.render_mode != "human":
            return
        water_surface = g.Box([30, 30, 0.01])
        water_material = g.MeshPhongMaterial(
            color=0x2389DA, opacity=0.3, transparent=True, side="DoubleSide"
        )
        self.vis["water_surface"].set_object(water_surface, water_material)

        water_volume = g.Box([30, 30, -50])
        water_volume_material = g.MeshPhongMaterial(
            color=0x1A6B9F, opacity=0.2, transparent=True
        )
        water_volume_transform = tf.translation_matrix([0, 0, -5])
        self.vis["water_volume"].set_object(water_volume, water_volume_material)
        self.vis["water_volume"].set_transform(water_volume_transform)

        self.vis["vessel"].set_object(
            g.DaeMeshGeometry.from_file("BlueRov2.dae"),
            g.MeshLambertMaterial(color=0x0000FF, wireframe=False),
        )

        ground = g.Box([30, 30, 0.01])
        ground_material = g.MeshPhongMaterial(color=0x808080, side="DoubleSide")
        ground_transform = tf.translation_matrix([0, 0, -10])
        self.vis["ground"].set_object(ground, ground_material)
        self.vis["ground"].set_transform(ground_transform)

    def step_sim(self):
        if self.render_mode != "human":
            return

        translation = np.array([self.state["x"], self.state["y"], self.state["z"]])
        rotation_matrix = np.array(
            [
                [np.cos(self.state["theta"]), -np.sin(self.state["theta"]), 0],
                [np.sin(self.state["theta"]), np.cos(self.state["theta"]), 0],
                [0, 0, 1],
            ]
        )
        transform_matrix = np.eye(4)
        transform_matrix[:3, :3] = rotation_matrix
        transform_matrix[:3, 3] = translation

        self.vis["vessel"].set_transform(transform_matrix)
