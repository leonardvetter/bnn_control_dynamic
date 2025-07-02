# bnn_control_dynamic
The goal is to use a Bayesian Neuronal Network (BNN) to learn the dynamic behavior of a simple control loop (spring, mass and dampener) together with a PID controller.

control_system_sim.py defines the class SMD_system_Sim that simulates the physical system and the class PID_Controller.
To see a system of the symulation, download the files and execute demo_control_system.py.

bnn_trained_on_control_system_02.py trains a simple BNN to predict the system behaviour.
$x_t$ Position at time t
$u_t$ Control input at time t (Force, applied by PID controller)
The goal of the BNN is to predict the next position $x_{t+1}$, given the current position and control input $(x_t,u_t)$.

We train the BNN by generating an artificial Dataset $D$, using control_system_sim


