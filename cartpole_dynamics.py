"""
Cartpole Dynamics script

All cartpole dynamics related things will belong here.
"""
import numpy as np 
from casadi import *

""" Linearization of Cartpole dynamics """
### Cartpend dynamics
## Cartpend params
m = 0.1     # m_p: mass of pendulum
M = 1.0     # m_c: mass of cart
L = 0.5
g = 9.81
d = 1.0

s = 0 # pendulum up (s=1)

## Nonlinear dynamics
def cartpend_nonlinear_func(x,u):
    ### Returns nonlinear model of cartpend f(x,u)
    # x: state [x, x_dot, theta, theta_dot].T == [x0, x1, x2, x3]
    # u: control input (force)
    theta_ddot = (g*np.sin(x[2]) + np.cos(x[2])*((-u - (m*L*x[3]**2*np.sin(x[2]))) / (m + M))) / \
            (L * ((4/3) - ((m*np.cos(x[2])**2) / (m + M))))

    x_ddot = (u + (m*L*(x[3]**2*np.sin(x[2]) - theta_ddot*np.cos(x[2])))) / (m + M)

    x_state = np.concatenate((np.array([x[1]]), 
                            np.array(x_ddot), 
                            np.array([x[3]]), 
                            np.array(theta_ddot)), 
                            axis=0)
    return x_state

## Linearized dynamics
A = np.array([[0., 1., 0., 0.],
            [0., 0., 0., 0.],
            [0., 0., 0., 1.],
            [0., 0., g / (L*((4/3) - (m / (m+M)))), 0.]])

B = np.array([0., 1. / (m + M), 0., -1 / (m + M)])
B = B.reshape(-1,1)






""" Nonlinear dynamics """
def cartpend_nonlinear_func_vec_(x,u):
    ### Returns nonlinear model of cartpend f(x,u)
    # x: state [x, x_dot, theta, theta_dot].T == [x0, x1, x2, x3]
    # u: control input (force)
    theta_ddot = (g*sin(x[2,:]) + cos(x[2,:])*((-u - (m*L*x[3,:]**2*sin(x[2,:]))) / (m + M))) / \
            (L * ((4/3) - ((m*cos(x[2,:])**2) / (m + M))))

    x_ddot = (u + (m*L*(x[3,:]**2*sin(x[2,:]) - theta_ddot*cos(x[2,:])))) / (m + M)

    return vertcat(x[1,:], x_ddot, x[3,:], theta_ddot)

