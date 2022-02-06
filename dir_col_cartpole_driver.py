import time
import gym
from gym.wrappers import Monitor

import casadi
from casadi import *

import numpy as np
import matplotlib.pyplot as plt

from continuous_cartpole import ContinuousCartPoleEnv
from plot_utils import *

from dir_col import *
from cartpole_dynamics import *

""" Cartpend Dynamics """
## Cartpend params
m = 0.1     # m_p: mass of pendulum
M = 1.0     # m_c: mass of cart
L = 0.5
g = 9.81
d = 1.0

s = 1 # pendulum up (s=1)

""" MPC params """
## Reference point
x_ref = np.array([0., 0., 0., 0.])

## MPC params
N = 100
DT = 0.02

nx = x_ref.shape[0]
nu = 1

## Weights
Q = np.eye(nx) * 0.01
R = 0.001


""" Utilities """
def map(x, in_min, in_max, out_min, out_max):
    return float((x-in_min) * (out_max-out_min) / (in_max-in_min) + out_min)


""" Goood ol main """
def main():
    """ Init environment """    
    env = ContinuousCartPoleEnv()
    
    done = False
    obs = env.reset(state=[0, 0, np.pi, 0])
    # obs = env.reset()   # Randomly initializes in the vicinity of equilibrium point

    """ Run the controller """
    x_guess = None  # Initialize x_guess
    while True:
        start_time = time.time()
    
        ### MPC solve 
        # x_sol, u_sol = trap_dir_col(cartpend_nonlinear_func_vec_, obs, x_guess=x_guess, N=60)
        x_sol, u_sol = hermite_simp_dir_col(cartpend_nonlinear_func_vec_, obs, x_ref, x_guess=x_guess, N=55, dt=DT)

        controller_out = u_sol[0]
        controller_out = map(controller_out, -30, 30, -1, 1)
    
        action = np.clip(controller_out, env.action_space.low, env.action_space.high)

        # Step through environment
        obs, rew, done, info = env.step(action)

        env.render()

        ### Store the guess for next iteration
        # x_guess = x_sol
        
        ### Debugging
        # print(f"Current state: {obs}")
        # print(f"Control input: {controller_out}")
        # print(f"Action (constrained): {action}")
        # print("----------------")


        ### Debug control output
        # print(f"x_sol: ", x_sol)
        # plot_control_output(x_sol, u_sol, x_ref)
        # plt.show(block=False)
        
        # # Step into next iteration
        # input("[+] Step...")
        # plt.close('all')


        print(f"[+] Solve time: {time.time() - start_time}")

if __name__ == "__main__":
    main()  



