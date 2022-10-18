import numpy as np

class Marker:
    # Measurements are of landmarks in 2D and have a position as well as tag id.
    def __init__(self, position, tag, covariance = (0.1*np.eye(2))):
        self.position = position
        self.tag = tag
        self.covariance = covariance

class Drive:
    # Measurement of the robot wheel velocities
    def __init__(self, left_speed, right_speed, dt, left_cov = 1, right_cov = 1):
        self.left_speed = left_speed
        self.right_speed = right_speed
        self.dt = dt

        if left_speed == 0 and right_speed == 0: #stopped:
            cov = 0
        elif left_speed == right_speed: # Lower covariance for driving straight
            cov = 1
        else: # higher covariance since turning is less consistent
            cov = 2

        self.left_cov = cov
        self.right_cov = cov
        # self.left_cov = left_cov
        # self.right_cov = right_cov