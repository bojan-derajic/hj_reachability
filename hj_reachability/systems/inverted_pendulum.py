import jax.numpy as jnp
from typing import Literal

from hj_reachability import dynamics, sets


class InvertedPendulum(dynamics.ControlAndDisturbanceAffineDynamics):
    """Class that implements inverted pendulum model."""

    def __init__(
        self,
        control_mode: Literal["min", "max"] = "max",
        disturbance_mode: Literal["min", "max"] = "min",
        control_space=None,
        disturbance_space=None,
    ):
        if control_space == None:
            control_space = sets.Box(lo=jnp.array([0.0]), hi=jnp.array([0.0]))
        if disturbance_space == None:
            disturbance_space = sets.Box(lo=jnp.array([0.0]), hi=jnp.array([0.0]))
        super().__init__(
            control_mode, disturbance_mode, control_space, disturbance_space
        )

        self.m = 2.0  # mass of the pendulum
        self.l = 1.0  # length of the pendulum
        self.g = 9.81  # acceleration due to gravity

    def open_loop_dynamics(self, state, time):
        """Implements open loop dynamics"""
        f = jnp.array([state[1], (self.g / self.l) * jnp.sin(state[0])])
        print(f.shape)
        return f

    def control_jacobian(self, state, time):
        """Calculates control Jacobian"""
        return jnp.array([[0.0], [1.0 / (self.m * self.l**2)]])

    def disturbance_jacobian(self, state, time):
        """Calculates disturbance Jacobian"""
        return jnp.array([[0.0], [1.0]])
