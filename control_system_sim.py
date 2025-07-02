import numpy as np
# ────────────────────────────────────────────────────────────────
#  Spring–Mass–Damper (discrete-time, constant-acceleration)
# ────────────────────────────────────────────────────────────────
class SMD_System_Sim:
    """
    Second-order spring–mass–damper model

        m x¨ + c ẋ + k x = F

    solved in fixed time steps under the assumption of **constant
    acceleration** during each step (small Δt ⇒ good accuracy).
    """

    def __init__(self, mass: float, k: float, c: float):
        self.mass = mass          # kg (or any consistent unit)
        self.k = k                # N/m Spring constant
        self.c = c                # N·s/m Damping constant Damping is assumed to depend linearly on velocity

    # ----- kinematic equations in discrete form --------------------
    def get_acceleration(self, force: float, v0: float, x0: float) -> float:
        """a = (F − k x − c v) / m"""
        return (force - self.k * x0 - self.c * v0) / self.mass

    def get_velocity(self, a: float, v0: float, dt: float) -> float:
        """v(t+dt) = v0 + a Δt"""
        return v0 + a * dt

    def get_position(self, a: float, v0: float, x0: float, dt: float) -> float:
        """x(t+dt) = x0 + v0 Δt + ½ a Δt²"""
        return x0 + v0 * dt + 0.5 * a * dt**2


# ────────────────────────────────────────────────────────────────
#                        PID Controller
# ────────────────────────────────────────────────────────────────
class PID_Controller:
    """
    Discrete-time PID controller with:

    * fixed sample period (sample_rate)
    * anti-wind-up via integral clamping
    * derivative on error
    """
    #Initialize:
    def __init__(
        self,
        kp: float = 1.0,
        ki: float = 0.0,
        kd: float = 0.0,
        max_signal: float = 1000.0,
        sample_rate: float = 0.1,
    ):
        # PID gains
        self.kp, self.ki, self.kd = kp, ki, kd

        # runtime state
        self.target = 0.0
        self.signal = 0.0
        self.accumulator = 0.0
        self.last_error = 0.0

        # limits / timing
        self.max_signal = max_signal
        self.sample_rate = sample_rate

    # ----- public API ------------------------------------------
    def set_new_target(self, target: float) -> None:
        """Start a new command ⇒ reset integral & derivative state"""
        self.target = target
        self.accumulator = 0.0
        self.last_error = 0.0

    # core PID calculation
    def _compute_signal(self, feedback: float, dt: float) -> None:
        error = self.target - feedback

        # P
        p_term = self.kp * error

        # I – integrate with respect to time
        self.accumulator += error * dt
        # anti-wind-up
        if self.ki:
            i_limit = self.max_signal / self.ki
            self.accumulator = np.clip(self.accumulator, -i_limit, i_limit)
        i_term = self.ki * self.accumulator

        # D
        d_error = (error - self.last_error) / dt
        d_term = self.kd * d_error
        self.last_error = error

        # total and clamp
        self.signal = p_term + i_term + d_term
        self.signal = np.clip(self.signal, -self.max_signal, self.max_signal)

    # -----------------------------------------------------------
    def run_simulation(
        self,
        system: SMD_System_Sim,
        duration: float,
        commands: list[list[float]],
        time_step: float = 0.01,
    ):
        """
        Simulate closed-loop response of *system* for *duration* [s].

        Parameters
        ----------
        system    : SMD_System_Sim
        duration  : total simulation time [s]
        commands  : list of [t, target] pairs (step sequence) 
        t         : defines at which time the target is fed to the controller
        time_step : integration step size; must divide sample_rate

        Returns (numpy arrays)
        -------
        t  : time vector
        u  : controller output (signal)
        x  : position
        r  : target history (command profile)
        """
        # ---- precompute timeline --------------------------------
        t = np.arange(0.0, duration + time_step, time_step)
        n = len(t)

        # ---- buffers --------------------------------------------
        x = np.zeros(n)
        v = np.zeros(n)
        u = np.zeros(n)
        r = np.zeros(n)

        # ---- command handling ------------------------------------
        cmd_idx = 0
        if commands and commands[0][0] == 0:
            self.set_new_target(commands[0][1])
            cmd_idx += 1
        r[0] = self.target

        # initial output / plant acceleration
        self._compute_signal(x[0], dt=self.sample_rate)
        u[0] = self.signal
        a_prev = system.get_acceleration(u[0], v[0], x[0])

        # ---- sampling bookkeeping -------------------------------
        sample_interval = max(1, int(round(self.sample_rate / time_step)))

        # ---- main loop ------------------------------------------
        for i in range(1, n):
            # process step commands
            if cmd_idx < len(commands) and t[i] >= commands[cmd_idx][0] - 1e-12:
                self.set_new_target(commands[cmd_idx][1])
                cmd_idx += 1
            r[i] = self.target

            # propagate plant state (uses acceleration from *previous* step)
            x[i] = system.get_position(a_prev, v[i - 1], x[i - 1], time_step)
            v[i] = system.get_velocity(a_prev, v[i - 1], time_step)

            # controller update at fixed sampling period
            if i % sample_interval == 0:
                self._compute_signal(x[i], dt=self.sample_rate)

            u[i] = self.signal
            a_prev = system.get_acceleration(u[i], v[i], x[i])

        return t, u, x, r

    # -----------------------------------------------------------
    @staticmethod 
    def ise(x: np.ndarray, r: np.ndarray) -> float:
        """Integral of squared error – quick performance metric"""
        return np.sum((x - r) ** 2)
