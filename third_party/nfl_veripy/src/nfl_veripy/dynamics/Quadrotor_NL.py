import numpy as np
import torch
from nfl_veripy.utils.mpc import control_mpc
from scipy.linalg import solve_discrete_are

from .Dynamics import DiscreteTimeDynamics


class Quadrotor_NL(DiscreteTimeDynamics):
    def __init__(self, dt=0.1):
        At = np.eye(6)
        bt = np.array([[0], [0], [0], [0], [0], [1]])*dt
        ct = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]).T

        # u_limits = None
        u_limits = 2 * np.array([[-1, 1]])
        self.vt = 1.
        self.g = 9.81

        super().__init__(At=At, bt=bt, ct=ct, u_limits=u_limits, dt=dt)

        self.cmap_name = "tab10"
        self.r = 1
        self.nn_theta = 0
        self.min_theta = 0

    def dynamics_step(self, xs, us):
        if isinstance(xs, np.ndarray):  # For tracking MC samples

            x_t1 = xs[:, 0] + self.dt*xs[:, 3]
            y_t1 = xs[:, 1] + self.dt*xs[:, 4]
            z_t1 = xs[:, 2] + self.dt*xs[:, 5]
            vx_t1 = xs[:, 3] + self.dt*self.g*np.tan(us[:, 0])
            vy_t1 = xs[:, 4] - self.dt*self.g*np.tan(us[:, 1])
            vz_t1 = xs[:, 5] + self.dt*(us[:, 2] - self.g)

            xs_t1 = np.vstack((x_t1, y_t1, z_t1, vx_t1, vy_t1, vz_t1)).T

        else:  # For solving LP
            xs_t1 = self.At @ xs + self.bt @ us + self.ct
            # if self.x_limits is not None and isinstance(xs, np.ndarray):
            #     for key in self.x_limits:
            #         xs_t1[:, key] = np.minimum(
            #             xs_t1[:, key], self.x_limits[key][1]
            #         )
            #         xs_t1[:, key] = np.maximum(
            #             xs_t1[:, key], self.x_limits[key][0]
            #         )

        return xs_t1

    def dynamics_step_jnp(self, xs, us):
        return (
            jnp.dot(self.At_jnp, xs.T).T
            + jnp.dot(self.bt_jnp, us.T).T
            + self.ct_jnp
        )

    def dynamics_step_torch(self, xs, us):
        return (
            torch.mm(xs, self.At_torch)
            + torch.mm(us, self.bt_torch)
            + self.ct_torch
        )

    def control_mpc(self, x0):
        # LQR-MPC parameters
        if not hasattr(self, "Q"):
            self.Q = np.eye(2)
        if not hasattr(self, "R"):
            self.R = np.eye(2)
        if not hasattr(self, "Pinf"):
            self.Pinf = solve_discrete_are(self.At, self.bt, self.Q, self.R)

        return control_mpc(
            x0,
            self.At,
            self.bt,
            self.ct,
            self.Q,
            self.R,
            self.Pinf,
            self.u_limits[:, 0],
            self.u_limits[:, 1],
            n_mpc=10,
            debug=False,
        )

    # Control function for if model gives [v, w] command inputs
    def control_nn_vw(self, x, model):
        if x.ndim == 1:
            batch_x = np.expand_dims(x, axis=0)
        else:
            batch_x = x
        us = model.forward(torch.Tensor(batch_x)).data.numpy()
        if not hasattr(self, "theta") or len(self.theta) != len(us):
            self.theta = np.zeros(len(us))

        R = np.array(
            [
                [
                    [np.cos(theta), -self.r * np.sin(theta)],
                    [np.sin(theta), self.r * np.cos(theta)],
                ]
                for theta in self.theta
            ]
        )
        # import pdb; pdb.set_trace()
        us_transformed = np.array(
            [R[i][:, 0] * us[i][0] for i in range(len(us))]
        )

        # print("theta: {}".format(self.theta[0]))
        # print("transformed u: {}".format(us_transformed[0]))
        # print("x-direction: {}".format(R[0][:,0]))
        self.theta = self.theta + self.dt * us[:, 1]
        return us_transformed

    # def control_nn(self, x, model):
    #     if x.ndim == 1:
    #         batch_x = np.expand_dims(x, axis=0)
    #     else:
    #         batch_x = x
    #     us = model.forward(torch.Tensor(batch_x)).data.numpy()
    #     if not hasattr(self, "theta") or len(self.theta) != len(us):
    #         self.theta = np.zeros(len(us))

    #     R = np.array(
    #         [
    #             [
    #                 [np.cos(theta), -self.r * np.sin(theta)],
    #                 [np.sin(theta), self.r * np.cos(theta)],
    #             ]
    #             for theta in self.theta
    #         ]
    #     )
    #     # import pdb; pdb.set_trace()
    #     us_transformed = np.array(
    #         [R[i][:, 0] * us[i][0] for i in range(len(us))]
    #     )

    #     # print("theta: {}".format(self.theta[0]))
    #     # print("transformed u: {}".format(us_transformed[0]))
    #     # print("x-direction: {}".format(R[0][:,0]))
    #     self.theta = self.theta + self.dt * us[:, 1]
    #     return us_transformed