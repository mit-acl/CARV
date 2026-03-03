import numpy as np

class SafetyFilter:
    """
    Implements the "1-step nominal + k-step safe backup control" safety filter.
    Nominal control is retreived from NN controller
    """

    def __init__(self, dynamics, estimator, obstacles, u_safe, controller,
                 horizon: int = 8):
        self.dynamics   = dynamics
        self.estimator  = estimator
        self.obstacles  = obstacles
        self.u_safe     = u_safe
        self.controller = controller
        self.horizon    = horizon
        self.At = np.array(dynamics.At)
        self.bt = np.array(dynamics.bt)
        self.ct = np.array(dynamics.ct).flatten()

    def get_nominal_action(self) -> np.ndarray:
        """Apply NN controller on current Kalman estimated state."""
        u_nn = self.dynamics.control_nn(
            self.estimator.x.reshape(1, -1),
            self.controller.cpu()
        )
        return np.atleast_1d(u_nn).flatten()

    def step_bounds(self, bounds: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Given current bounds and control, compute next step bounds."""
        A_pos = np.maximum(self.At, 0.0)
        A_neg = np.minimum(self.At, 0.0)
        u_flat = np.atleast_1d(u).flatten()
        ctrl  = (self.bt @ u_flat + self.ct).flatten()
        low_b, high_b = bounds[:, 0], bounds[:, 1]
        new_lb = A_pos @ low_b + A_neg @ high_b + ctrl
        new_ub = A_pos @ high_b + A_neg @ low_b + ctrl
        return np.stack([new_lb, new_ub], axis=1)


    def collides(self, bounds: np.ndarray) -> bool:
        for obs in self.obstacles:
            if np.all(bounds[:, 1] >= obs[:, 0]) and np.all(bounds[:, 0] <= obs[:, 1]):
                return True
        return False


    def filter(self, current_bounds: np.ndarray) -> dict:
        """
        Check if nominal action is safe and return chosen action.
        """

        u_nom = self.get_nominal_action()
        u_safe = self.u_safe
        trajectory = [current_bounds]

        # Step 1: nominal
        bounds = self.step_bounds(current_bounds, u_nom)
        trajectory.append(bounds)
        if self.collides(bounds):
            return dict(action=u_safe, u_nominal=u_nom, intervened=True,
                        reason='Nominal step itself collides. Should apply safe backup control',
                        collision_at=1, trajectory=trajectory)

        # Steps 2..H+1: safe backup
        for step in range(1, self.horizon + 1):
            bounds = self.step_bounds(bounds, u_safe)
            trajectory.append(bounds)
            if self.collides(bounds):
                return dict(action=u_safe, u_nominal=u_nom, intervened=True,
                            reason=f'Backup collides at step {step + 1}',
                            collision_at=step + 1, trajectory=trajectory)

        return dict(action=u_nom, u_nominal=u_nom, intervened=False,
                    reason='Safe to apply nominal control',
                    collision_at=None, trajectory=trajectory)

    def is_safe(self, current_bounds: np.ndarray) -> bool:
        return not self.filter(current_bounds)['intervened']


def make_safety_filter(tester, obstacles, u_max: float = 1.0, horizon: int = 8):
    """
    Build a SafetyFilter
    Return None if Unicycle dynamics - havent set that up yet
    """

    dynamics = tester.analyzer.cl_system.dynamics
    if type(dynamics).__name__ != 'DoubleIntegrator':
        return None

    return SafetyFilter(
        dynamics=tester.analyzer.cl_system.dynamics,
        estimator=tester.estimator,
        obstacles=obstacles,
        u_safe=np.array([u_max]),
        controller=tester.analyzer.cl_system.controller,
        horizon=horizon,
    )
