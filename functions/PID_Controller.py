class PIDController:
    def __init__(self, kp, ki, kd, dt, windup_guard=20.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.dt = dt
        self.integral = 0.0
        self.prev_error = 0.0
        self.windup_guard = windup_guard

    def update(self, error):
        self.integral += error * self.dt
        # Clamp the integral term to prevent windup.
        if self.integral > self.windup_guard:
            self.integral = self.windup_guard
        elif self.integral < -self.windup_guard:
            self.integral = -self.windup_guard
        derivative = (error - self.prev_error) / self.dt
        self.prev_error = error
        return self.kp * error + self.ki * self.integral + self.kd * derivative

    def get_parameters(self):
        return {
            "kp": self.kp,
            "ki": self.ki,
            "kd": self.kd,
            "windup_guard": self.windup_guard
        }
