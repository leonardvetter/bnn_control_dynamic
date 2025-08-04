# bnn_control_dynamic
The goal is to use a Bayesian Neuronal Network (BNN) to learn the dynamic behavior of a simple control loop (spring, mass and dampener) together with a PID controller.


control_system_sim.py defines the class SMD_system_Sim that simulates the physical system and the class PID_Controller.
To see a simulation of the system, download the files and execute demo_control_system.py.


bnn_trained_on_control_system_02.py trains a simple BNN to predict the system behaviour.


$s_t$ State at time t


$u_t$ Control input at time t (Force, applied by PID controller)


The goal of the BNN is to predict the next position $x_{t+1}$, given the current position, velocity, acceleration and control input $(x_t,u_t)$.
EDIT: It was observed that the BNN was uncertain at times with no control input. The hypothesis is that it has no knowledge about the history of the system.
Therefore, the jupyter notebook we improved the accuracy by including v_t and a_t for the predicion, i.e. we use $s_t = (x_t, v_t, a_t)$ and $u_t$ for predicting $x_{t+1}$.
They are approximated by v_t = (x_t-x_t-1)/delta_t and a_t = (x_t-x_t-2)/delta_t^2. We could probably also just use $(x_t,x_{t-1},x_{t-2},u_t)$ as input.



We train the BNN by generating an artificial Dataset $D = \{((s_t,u_t),x_{t+1})  \}$, using the control system simulation.
The BNN also gives the uncertainty of the prediction.


