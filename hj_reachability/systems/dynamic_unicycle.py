import jax.numpy as jnp
from typing import Literal

from hj_reachability import dynamics, sets


class DynamicUnicycle(dynamics.ControlAndDisturbanceAffineDynamics):
    """Class that implements dynamic unicycle model."""

    def __init__(
        self,
        control_mode: Literal["min", "max"] = "max",
        disturbance_mode: Literal["min", "max"] = "min",
        control_space=None,
        disturbance_space=None,
    ):
        if control_space == None:
            control_space = sets.Box(lo=jnp.array([0.0, 0.0]), hi=jnp.array([0.0, 0.0]))
        if disturbance_space == None:
            disturbance_space = sets.Box(
                lo=jnp.array([0.0, 0.0]), hi=jnp.array([0.0, 0.0])
            )
        super().__init__(
            control_mode, disturbance_mode, control_space, disturbance_space
        )

    def open_loop_dynamics(self, state, time):
        """Implements open loop dynamics of a dynamic unicycle model"""
        theta = state[2]
        v = state[3]
        omega = state[4]
        return jnp.array(
            [
                v * jnp.cos(theta),
                v * jnp.sin(theta),
                omega,
                0.0,
                0.0,
            ]
        )

    def control_jacobian(self, state, time):
        """Calculates control Jacobian of a dynamic unicycle model"""
        return jnp.array(
            [
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
            ]
        )

    def disturbance_jacobian(self, state, time):
        """Calculates disturbance Jacobian of a dynamic unicycle model"""
        return jnp.array(
            [
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
            ]
        )
