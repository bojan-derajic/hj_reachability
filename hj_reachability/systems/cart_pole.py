import jax.numpy as jnp
from typing import Literal

from hj_reachability import dynamics, sets


class CartPole(dynamics.ControlAndDisturbanceAffineDynamics):
    """Cart-pole (cart with inverted pendulum) system.

    State: [x, theta, x_dot, theta_dot]
        x         - horizontal cart position
        theta     - pendulum angle (0 = upright, pi = hanging down)
        x_dot     - cart velocity
        theta_dot - pendulum angular velocity

    Control: [f_x] - horizontal force applied to the cart

    Dynamics (theta=0 upright convention, derived via Lagrangian mechanics):
        x_ddot     = [f_x + m_p*sin(theta)*(l*theta_dot^2 - g*cos(theta))] / D
        theta_ddot = [(m_c + m_p)*g*sin(theta) - m_p*l*theta_dot^2*cos(theta)*sin(theta)
                      - f_x*cos(theta)] / (l * D)
    where D = m_c + m_p * sin^2(theta)
    """

    m_c: float  # cart mass [kg]
    m_p: float  # pendulum mass [kg]
    l: float    # pendulum length [m]
    g: float    # gravitational acceleration [m/s^2]

    def __init__(
        self,
        control_mode: Literal["min", "max"] = "max",
        disturbance_mode: Literal["min", "max"] = "min",
        control_space=None,
        disturbance_space=None,
        params: dict = {},
    ):
        if control_space is None:
            control_space = sets.Box(lo=jnp.array([0.0]), hi=jnp.array([0.0]))
        if disturbance_space is None:
            disturbance_space = sets.Box(lo=jnp.array([0.0]), hi=jnp.array([0.0]))
        super().__init__(
            control_mode, disturbance_mode, control_space, disturbance_space
        )
        self.m_c = params.get("m_c", 1.0)
        self.m_p = params.get("m_p", 0.1)
        self.l = params.get("l", 0.5)
        self.g = params.get("g", 9.81)

    def open_loop_dynamics(self, state, time):
        """Open-loop (unforced) dynamics f(x)."""
        _, theta, x_dot, theta_dot = state[0], state[1], state[2], state[3]
        D = self.m_c + self.m_p * jnp.sin(theta) ** 2
        x_ddot = self.m_p * jnp.sin(theta) * (self.l * theta_dot**2 - self.g * jnp.cos(theta)) / D
        theta_ddot = (
            (self.m_c + self.m_p) * self.g * jnp.sin(theta)
            - self.m_p * self.l * theta_dot**2 * jnp.cos(theta) * jnp.sin(theta)
        ) / (self.l * D)
        return jnp.array([x_dot, theta_dot, x_ddot, theta_ddot])

    def control_jacobian(self, state, time):
        """Control Jacobian g(x) such that dynamics = f(x) + g(x)*u."""
        theta = state[1]
        D = self.m_c + self.m_p * jnp.sin(theta) ** 2
        return jnp.array([
            [0.0],
            [0.0],
            [1.0 / D],
            [-jnp.cos(theta) / (self.l * D)],
        ])

    def disturbance_jacobian(self, state, time):
        """Disturbance Jacobian (additive disturbance on cart force)."""
        theta = state[1]
        D = self.m_c + self.m_p * jnp.sin(theta) ** 2
        return jnp.array([
            [0.0],
            [0.0],
            [1.0 / D],
            [-jnp.cos(theta) / (self.l * D)],
        ])
