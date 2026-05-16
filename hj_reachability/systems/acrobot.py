import jax.numpy as jnp
from typing import Literal

from hj_reachability import dynamics, sets


class Acrobot(dynamics.ControlAndDisturbanceAffineDynamics):
    """Acrobot (two-link underactuated pendulum) system.

    State: [theta1, theta2, theta1_dot, theta2_dot]
        theta1     - angle of the first link from the downward vertical
        theta2     - angle of the second link relative to the first link
        theta1_dot - angular velocity of the first link
        theta2_dot - angular velocity of the second link

    Control: [tau2] - torque applied at the second (elbow) joint
             (the first joint is unactuated)

    Dynamics (Spong's formulation):
        M(q) * q_ddot + C(q, q_dot) + G(q) = B * u
    where
        M11 = m1*lc1^2 + m2*(l1^2 + lc2^2 + 2*l1*lc2*cos(theta2)) + I1 + I2
        M12 = m2*(lc2^2 + l1*lc2*cos(theta2)) + I2
        M22 = m2*lc2^2 + I2
        C1  = -m2*l1*lc2*sin(theta2)*(2*theta1_dot*theta2_dot + theta2_dot^2)
        C2  =  m2*l1*lc2*sin(theta2)*theta1_dot^2
        G1  = -(m1*lc1 + m2*l1)*g*sin(theta1) - m2*lc2*g*sin(theta1 + theta2)
        G2  = -m2*lc2*g*sin(theta1 + theta2)
        B   = [0, 1]^T
    """

    m1: float
    m2: float
    l1: float
    l2: float
    lc1: float
    lc2: float
    I1: float
    I2: float
    gravity: float

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
        self.m1 = params.get("m1", 1.0)
        self.m2 = params.get("m2", 1.0)
        self.l1 = params.get("l1", 1.0)
        self.l2 = params.get("l2", 1.0)
        self.lc1 = params.get("lc1", 0.5)
        self.lc2 = params.get("lc2", 0.5)
        self.I1 = params.get("I1", 1.0 / 12.0)
        self.I2 = params.get("I2", 1.0 / 12.0)
        self.gravity = params.get("gravity", 9.81)

    def _mass_matrix(self, theta2):
        """Compute mass matrix elements and determinant."""
        cos_t2 = jnp.cos(theta2)
        M11 = (
            self.m1 * self.lc1**2
            + self.m2 * (self.l1**2 + self.lc2**2 + 2.0 * self.l1 * self.lc2 * cos_t2)
            + self.I1
            + self.I2
        )
        M12 = self.m2 * (self.lc2**2 + self.l1 * self.lc2 * cos_t2) + self.I2
        M22 = self.m2 * self.lc2**2 + self.I2
        det_M = M11 * M22 - M12**2
        return M11, M12, M22, det_M

    def open_loop_dynamics(self, state, time):
        """Open-loop (unforced) dynamics f(x)."""
        theta1 = state[0]
        theta2 = state[1]
        theta1_dot = state[2]
        theta2_dot = state[3]

        M11, M12, M22, det_M = self._mass_matrix(theta2)

        sin_t2 = jnp.sin(theta2)
        C1 = (
            -self.m2
            * self.l1
            * self.lc2
            * sin_t2
            * (2.0 * theta1_dot * theta2_dot + theta2_dot**2)
        )
        C2 = self.m2 * self.l1 * self.lc2 * sin_t2 * theta1_dot**2

        G1 = -(
            (self.m1 * self.lc1 + self.m2 * self.l1) * self.gravity * jnp.sin(theta1)
            + self.m2 * self.lc2 * self.gravity * jnp.sin(theta1 + theta2)
        )
        G2 = -self.m2 * self.lc2 * self.gravity * jnp.sin(theta1 + theta2)

        rhs1 = -(C1 + G1)
        rhs2 = -(C2 + G2)

        theta1_ddot = (M22 * rhs1 - M12 * rhs2) / det_M
        theta2_ddot = (-M12 * rhs1 + M11 * rhs2) / det_M

        return jnp.array([theta1_dot, theta2_dot, theta1_ddot, theta2_ddot])

    def control_jacobian(self, state, time):
        """Control Jacobian g(x) such that dynamics = f(x) + g(x)*u."""
        theta2 = state[1]
        M11, M12, _, det_M = self._mass_matrix(theta2)

        return jnp.array([
            [0.0],
            [0.0],
            [-M12 / det_M],
            [M11 / det_M],
        ])

    def disturbance_jacobian(self, state, time):
        """Disturbance Jacobian (same channel as control by default)."""
        return self.control_jacobian(state, time)
