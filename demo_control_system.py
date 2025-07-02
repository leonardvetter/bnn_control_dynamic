
import numpy as np
import matplotlib.pyplot as plt
from control_system_sim import SMD_System_Sim
from control_system_sim import PID_Controller

# ────────────────────────────────────────────────────────────────
#                       Quick demo / sanity check
# ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # plant: 1 kg mass, k=1 N/m, c=0.1 N·s/m
    system = SMD_System_Sim(mass=1.0, k=1.0, c=0.1)

    # controller gains chosen by hand
    pid = PID_Controller(kp=10.0, ki=2.0, kd=1.0, max_signal=100.0)

    # simulate a single 1 m step at t=0
    t, u, x, r = pid.run_simulation(system, duration=100.0, commands= [[0,50], [20,49], [40,25],[60,40],[80,15]])

    print(f"Final position = {x[-1]:.3f} m, error = {x[-1] - r[-1]:+.3e} m")

    # quick plot
    plt.figure()
    plt.plot(t, x, label="position x(t)")
    plt.plot(t, r, "--", label="target r(t)")
    plt.xlabel("time [s]")
    plt.legend()
    plt.tight_layout()
    plt.title("Simulated PID controller of the spring-mass-damper system")
    plt.show()