import numpy as np
from scipy import stats


class Dynamics:
    def __init__(self):
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

    def step(self, state, action):
        disturbances = self.disturbance_dist.rvs()
        dvx, dvy, dvz, domega = disturbances

        # Extract action components
        print(action)
        w_x, w_y, w_z, w_omega = action

        # Update state using your existing physics model
        x, y, z, theta = (
            state["x"],
            state["y"],
            state["z"],
            state["theta"],
        )
        vx, vy, vz, omega = (
            state["vx"],
            state["vy"],
            state["vz"],
            state["omega"],
        )

        # Position updates
        state["x"] += (vx * np.cos(theta) - vy * np.sin(theta)) * self.dt
        state["y"] += (vx * np.sin(theta) + vy * np.cos(theta)) * self.dt
        state["z"] += vz * self.dt
        state["theta"] += omega * self.dt

        # Velocity updates
        state["vx"] += (
            self.params["a_vx"] * vx
            + self.params["a_vx2"] * vx * abs(vx)
            + self.params["a_vyw"] * vy * omega
            + self.params["b_vx"] * w_x
            + dvx
        ) * self.dt

        state["vy"] += (
            self.params["a_vy"] * vy
            + self.params["a_vy2"] * vy * abs(vy)
            + self.params["a_vwx"] * vx * omega
            + self.params["b_vy"] * w_y
            + dvy
        ) * self.dt

        state["vz"] += (
            self.params["a_vz"] * vz
            + self.params["a_vz2"] * vz * abs(vz)
            + self.params["b_vz"] * w_z
            + dvz
        ) * self.dt

        state["omega"] += (
            self.params["a_vw"] * omega
            + self.params["a_vw2"] * omega * abs(omega)
            + self.params["a_vxy"] * vx * vy
            + self.params["b_vw"] * w_omega
            + domega
        ) * self.dt

    def reset(self):
        self.disturbances = self.disturbance_dist.rvs()
