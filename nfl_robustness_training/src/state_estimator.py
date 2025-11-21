import numpy as np
from typing import Tuple, Callable
from filterpy.kalman import KalmanFilter
import torch


class StateEstimator:
    """
    Abstract base class for state estimation.
    Maintains uncertainty bounds around true state.
    """
    def __init__(self, initial_state: np.ndarray, initial_bounds: np.ndarray):
        self.state = initial_state.copy()
        self.bounds = initial_bounds.copy()
    
    def predict(self, dynamics_fn: Callable, control_input: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict next state and uncertainty bounds.
        
        Args:
            dynamics_fn: Function that takes (state, control) and returns next state
            control_input: Control input for this step
            
        Returns:
            next_state: Predicted state
            next_bounds: Uncertainty bounds [lower, upper] for each dimension
        """
        raise NotImplementedError
    
    def update(self, measurement: np.ndarray):
        """
        Update estimate with measurement (if available).
        
        Args:
            measurement: Observed state measurement
        """
        raise NotImplementedError
    
    def reset(self, state: np.ndarray, bounds: np.ndarray):
        """
        Reset estimator to a specific state and bounds.
        
        Args:
            state: New state vector
            bounds: New bounds array
        """
        raise NotImplementedError

class LinearKalmanEstimator(StateEstimator):
    """
    Standard Kalman Filter for linear dynamics with known A, B matrices.
    Perfect for your DoubleIntegrator system!
    """
    
    def __init__(self, initial_state: np.ndarray, initial_bounds: np.ndarray,
                 A: np.ndarray, B: np.ndarray,
                 process_noise_std: float = 0.01,
                 measurement_noise_std: float = 0.05):
        """
        Args:
            initial_state: Initial state estimate
            initial_bounds: Initial uncertainty bounds
            A: State transition matrix (from dynamics.At)
            B: Control matrix (from dynamics.bt)
            process_noise_std: Process noise standard deviation
            measurement_noise_std: Measurement noise standard deviation
        """
        super().__init__(initial_state, initial_bounds)
        
        n_states = len(initial_state)
        n_controls = B.shape[1]
        
        # Store system matrices
        self.A = A  # (n_states x n_states)
        self.B = B  # (n_states x n_controls)
        
        # State and covariance
        self.x = initial_state.copy()
        initial_std = (initial_bounds[:, 1] - initial_bounds[:, 0]) / 6.0
        self.P = np.diag(initial_std ** 2)
        
        # Process noise covariance: Q
        self.Q = np.eye(n_states) * (process_noise_std ** 2)
        
        # Measurement noise covariance: R
        self.R = np.eye(n_states) * (measurement_noise_std ** 2)
        
        # Measurement matrix (observe full state)
        self.H = np.eye(n_states)
    
    def predict(self, dynamics_fn: Callable, control_input: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Kalman Filter prediction step.
        
        Note: We ignore dynamics_fn and use the linear model directly!
        """
        # Convert control to numpy if needed
        if isinstance(control_input, torch.Tensor):
            u = control_input.cpu().numpy().flatten()
        else:
            u = control_input.flatten()

        u.reshape((1,-1))

        print(f'DEBUG: u is {u.shape} \n B is {self.B.shape}')
        
        # Ensure u is a column vector for matrix multiplication
        u = u.reshape(-1, 1)  # Shape: (n_controls, 1)
        
        # Ensure x is a column vector
        x = self.x.reshape(-1, 1)  # Shape: (n_states, 1)
        
        # Predict state: x = A*x + B*u
        x_pred = self.A @ x + self.B @ u
        self.x = x_pred.flatten()  # Convert back to 1D array
        
        # Predict covariance: P = A*P*A' + Q
        self.P = self.A @ self.P @ self.A.T + self.Q
        
        # Update bounds
        self.state = self.x.copy()
        self.bounds = self._covariance_to_bounds()
        
        return self.state.copy(), self.bounds.copy()
    
    def update(self, measurement: np.ndarray):
        """
        Kalman Filter update step.
        Fuses noisy measurement with prediction.
        """
        z = measurement.flatten()
        
        # Innovation (residual): y = z - H*x
        y = z - self.H @ self.x
        
        # Innovation covariance: S = H*P*H' + R
        S = self.H @ self.P @ self.H.T + self.R
        
        # Kalman gain: K = P*H' * inv(S)
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        # Update state: x = x + K*y
        self.x = self.x + K @ y
        
        # Update covariance: P = (I - K*H)*P
        I = np.eye(len(self.x))
        self.P = (I - K @ self.H) @ self.P
        
        # Update bounds
        self.state = self.x.copy()
        self.bounds = self._covariance_to_bounds()
    
    def reset(self, state: np.ndarray, bounds: np.ndarray):
        """Reset estimator to new state and bounds."""
        self.state = state.copy()
        self.bounds = bounds.copy()
        self.x = state.copy()
        
        # Reconstruct covariance from bounds
        std_devs = (bounds[:, 1] - bounds[:, 0]) / 6.0
        self.P = np.diag(std_devs ** 2)
    
    def _covariance_to_bounds(self, n_sigma: float = 3.0):
        """Convert covariance to bounds (3-sigma)."""
        std_devs = np.sqrt(np.diag(self.P))
        lower = self.x - n_sigma * std_devs
        upper = self.x + n_sigma * std_devs
        return np.stack([lower, upper], axis=1)