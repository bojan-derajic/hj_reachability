from typing_extensions import Literal
import jax.numpy as jnp

from hj_reachability import dynamics
from hj_reachability import sets


class UnicycleJoint(dynamics.ControlAndDisturbanceAffineDynamics):
    """Class that implements joint dynamics between an evader and a pursuer modeled as kinematic unicycles."""

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
        return jnp.zeros_like(state)

    def control_jacobian(self, state, time):
        return jnp.array(
            [
                [-jnp.cos(state[2]), 0.0],
                [-jnp.sin(state[2]), 0.0],
                [0.0, 1.0],
                [0.0, 0.0],
            ]
        )

    def disturbance_jacobian(self, state, time):
        return jnp.array(
            [
                [jnp.cos(state[3]), 0.0],
                [jnp.sin(state[3]), 0.0],
                [0.0, 0.0],
                [0.0, 1.0],
            ]
        )
