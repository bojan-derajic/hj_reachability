import jax.numpy as jnp
from typing import Literal

from hj_reachability import dynamics, sets


class VerticalDrone2D(dynamics.ControlAndDisturbanceAffineDynamics):
    """Class that implements 2D vertical drone model."""

    def __init__(
        self,
        control_mode: Literal["min", "max"] = "max",
        disturbance_mode: Literal["min", "max"] = "min",
        control_space=None,
        disturbance_space=None,
        K: float = 1.0,
    ):
        if control_space == None:
            control_space = sets.Box(lo=jnp.array([0.0]), hi=jnp.array([0.0]))
        if disturbance_space == None:
            disturbance_space = sets.Box(lo=jnp.array([0.0]), hi=jnp.array([0.0]))
        super().__init__(
            control_mode, disturbance_mode, control_space, disturbance_space
        )
        self.K = K

    def open_loop_dynamics(self, state, time):
        """Implements open loop dynamics of a 2D vertical drone"""
        return jnp.array([state[1], jnp.full_like(state[1], -9.81)])

    def control_jacobian(self, state, time):
        """Calculates control Jacobian of a 2D vertical drone"""
        return jnp.array([[0.0], [self.K]])

    def disturbance_jacobian(self, state, time):
        """Calculates disturbance Jacobian of a 2D vertical drone"""
        return jnp.array([[0.0], [1.0]])
