import jax.numpy as jnp
from typing import Literal

from hj_reachability import dynamics, sets


class KinematicBicycle(dynamics.ControlAndDisturbanceAffineDynamics):
    """Class that implements kinematic bicycle model."""

    def __init__(
        self,
        control_mode: Literal["min", "max"] = "max",
        disturbance_mode: Literal["min", "max"] = "min",
        params: dict = {},
        control_space=None,
        disturbance_space=None,
    ):
        if control_space == None:
            control_space = sets.Box(lo=jnp.array([0.0, 0.0]), hi=jnp.array([0.0, 0.0]))
        if disturbance_space == None:
            disturbance_space = sets.Box(
                lo=jnp.array([0.0, 0.0]), hi=jnp.array([0.0, 0.0])
            )
        self.params = params
        super().__init__(
            control_mode, disturbance_mode, control_space, disturbance_space
        )

    def open_loop_dynamics(self, state, time):
        """Implements open loop dynamics of a kinematic bicycle model"""
        psi = state[2]
        v = state[3]
        return jnp.array(
            [
                v * jnp.cos(psi),
                v * jnp.sin(psi),
                0.0,
                0.0,
            ]
        )

    def control_jacobian(self, state, time):
        """Calculates control Jacobian of a kinematic bicycle model"""
        L = self.params["L"]  # wheelbase length
        v = state[3]
        return jnp.array(
            [
                [0.0, 0.0],
                [0.0, 0.0],
                [v / L, 0.0],
                [0.0, 1.0],
            ]
        )

    def disturbance_jacobian(self, state, time):
        """Calculates disturbance Jacobian of a kinematic bicycle model"""
        L = self.params["L"]  # wheelbase length
        v = state[3]
        return jnp.array(
            [
                [0.0, 0.0],
                [0.0, 0.0],
                [v / L, 0.0],
                [0.0, 1.0],
            ]
        )
