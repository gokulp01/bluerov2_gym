# BlueROV2 Gymnasium Environment

A Gymnasium environment for simulating and training reinforcement learning agents on the BlueROV2 underwater vehicle. This environment provides a realistic simulation of the BlueROV2's dynamics and supports various control tasks.

## 🌊 Features

- **Realistic Physics**: Implements validated hydrodynamic model of the BlueROV2
- **3D Visualization**: Real-time 3D rendering using Meshcat
- **Custom Rewards**: Configurable reward functions for different tasks
- **Disturbance Modeling**: Includes environmental disturbances for realistic underwater conditions
- **Stable-Baselines3 Compatible**: Ready to use with popular RL frameworks
- **Customizable Environment**: Easy to modify for different underwater tasks

## 🛠️ Installation

### Prerequisites
- Python ≥3.10
- [uv](https://github.com/astral-sh/uv) (recommended) or pip

### Using uv (Recommended)
```bash
# Clone the repository
git clone https://github.com/gokulp01/bluerov2_gym.git
cd bluerov2_gym

# Create and activate a virtual environment
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install the package
uv pip install -e .
```

### Using pip
```bash
# Clone the repository
git clone https://github.com/gokulp01/bluerov2_gym.git
cd bluerov2_gym

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install the package
pip install -e .
```

## 🎮 Usage

### Basic Usage
```python
import gymnasium as gym
import bluerov2_gym

# Create the environment
env = gym.make("BlueRov-v0", render_mode="human")

# Reset the environment
observation, info = env.reset()

# Run a simple control loop
while True:
    # Take a random action
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    
    if terminated or truncated:
        observation, info = env.reset()
```

### Training with Stable-Baselines3 (refer to examples/train.py for full code example) 
```python
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# Create and wrap the environment
env = gym.make("BlueRov-v0")
env = DummyVecEnv([lambda: env])
env = VecNormalize(env)

# Initialize the agent
model = PPO("MultiInputPolicy", env, verbose=1)

# Train the agent
model.learn(total_timesteps=1_000_000)

# Save the trained model
model.save("bluerov_ppo")
```

## 🎯 Environment Details

### State Space
The environment uses a Dictionary observation space containing:
- `x, y, z`: Position coordinates
- `theta`: Yaw angle
- `vx, vy, vz`: Linear velocities
- `omega`: Angular velocity

### Action Space
Continuous action space with 4 dimensions:
- Forward/Backward thrust
- Left/Right thrust
- Up/Down thrust
- Yaw rotation

### Reward Function
The default reward function considers:
- Position error from target
- Velocity penalties
- Orientation error
- Custom rewards can be implemented by extending the `Reward` class

## 📊 Examples

The `examples` directory contains several scripts demonstrating different uses:

- `test.py`: Basic environment testing with manual control and evaluation with trained model
- `train.py`: Training script using PPO

### Running Examples
```bash
# Test environment with manual control
python examples/test.py

# Train an agent
python examples/train.py
```

## 🖼️ Visualization

The environment uses Meshcat for 3D visualization. When running with `render_mode="human"`, a web browser window will open automatically showing the simulation. The visualization includes:
- Water surface effects
- Underwater environment
- ROV model
- Ocean floor with decorative elements (I am no good at this) 

## 📚 Project Structure
```
bluerov2_gym/
├── bluerov2_gym/              # Main package directory
│   ├── assets/               # 3D models and resources
│   └── envs/                 # Environment implementation
│       ├── core/            # Core components
│       │   ├── dynamics.py  # Physics simulation
│       │   ├── rewards.py   # Reward functions
│       │   ├── state.py     # State management
│       │   └── visualization/
│       │       └── renderer.py  # 3D visualization
│       └── bluerov_env.py    # Main environment class
├── examples/                  # Example scripts
├── tests/                    # Test cases
└── README.md
```

## 🔧 Configuration

The environment can be configured through various parameters:
- Physics parameters in `dynamics.py`
- Reward weights in `rewards.py`
- Visualization settings in `renderer.py`

## 📝 Citation

If you use this environment in your research, please cite:
```bibtex
@software{bluerov2_gym,
  author = {Gokul Puthumanaillam},
  title = {BlueROV2 Gymnasium Environment},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/gokulp01/bluerov2_gym}
}
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License

## 🙏 Acknowledgements

- BlueRobotics for the BlueROV2 specifications
- OpenAI/Farama Foundation for the Gymnasium framework
- Meshcat for the visualization library

## 📧 Contact

Gokul Puthumanaillam - [@gokulp01](https://github.com/gokulp01) - [gokulp2@illinois.edu]

Project Link: [https://github.com/gokulp01/bluerov2_gym](https://github.com/gokulp01/bluerov2_gym)
