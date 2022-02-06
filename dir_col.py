import numpy as np
from casadi import *


def trap_dir_col(f_, x0, xf, N=100, nx=4, nu=1, dt=0.02):
    """
    Traoezoidal direct collocation

    f_: dynamics of the system as a function of f(x_k,u_k). Expects a CASADI function
    x0: initial conidition
    xf: final desired state
    x_guess: initial guess if provided otherwise linearly interpolated traj would be use as guess
    nx: state dim
    nu: control dim
    dt: delta time between collocation points
    """
    # Reshaping
    x0 = x0.reshape(-1,1)
    xf = xf.reshape(-1,1)
    
    """ Optimization process """
    opti = casadi.Opti()

    X = opti.variable(nx,N)     # state trajectory
    U = opti.variable(nu,N)   # input trajectory
    
    
    """ Cost and collocation constraints """
    h_k = dt

    ### Vectorized version
    # Collocation constraints
    f_ks = f_(X[:,:-1], U[:,:-1])
    f_k_1s = f_(X[:,1:], U[:,1:])
    
    opti.subject_to(X[:,1:] - X[:,:-1] == 0.5 * h_k * (f_ks + f_k_1s))

    # Compute cost
    cost = 0.5 * sum2(U[:,1:]**2 + U[:,:-1]**2) * h_k


    # # Compute final cost
    # cost += cost_final(X[:,-1], U[:,-1], x_ref)

    ## Minimize cost function
    opti.minimize(cost)

    """ Additional constraints """
    # Input constraint
    opti.subject_to(-30<=U)
    opti.subject_to(U<=30)

    # State constraint
    opti.subject_to(X[:,0] == x0)
    opti.subject_to(X[:,-1] == xf)

    opti.subject_to(-2 <= X[0,:])
    opti.subject_to(X[0,:] <= 2)

    """ Initial guess """
    ## Linear trajectory guess
    x_guess = np.hstack((np.linspace(x0[0], xf[0], N).reshape(-1,1),
                np.zeros(N).reshape(-1,1),
                np.linspace(x0[2], xf[2], N).reshape(-1,1),
                np.zeros(N).reshape(-1,1),
            ))
    x_guess = x_guess.T

    # Set initial guess
    opti.set_initial(X, x_guess)


    """ Solve! """
    opts = {'ipopt.print_level':0, 'print_time':0}
    opti.solver('ipopt', opts)      # Set numerical backend
    sol = opti.solve();   # actual solv

    x_sol = sol.value(X)
    u_sol = sol.value(U) 

    return x_sol, u_sol



""" Hermite-Simpson Collocation """
def hermite_simp_dir_col(f_, x0, xf, x_guess=None, N=100, nx=4, nu=1, dt=0.02):
    """
    Hermite-Simpson Direct Collocation 

    f_: dynamics of the system as a function of f(x_k,u_k). Expects a CASADI function 
    x0: initial conidition
    xf: final desired state
    x_guess: initial guess if provided otherwise linearly interpolated traj would be use as guess
    nx: state dim
    nu: control dim
    dt: delta time between collocation points
    """
    # Reshaping
    x0 = x0.reshape(-1,1)
    xf = xf.reshape(-1,1)
    
    """ Optimization process """
    opti = casadi.Opti()

    X = opti.variable(nx, N+(N-1))     # state trajectory
    U = opti.variable(nu, N+(N-1))   # input trajectory
        
    """ Cost and collocation constraints """
    h_k = dt
 
    ### Vectorized version
    # Collocation constraints
    x_k = X[:, :-1:2]
    u_k = U[:, :-1:2]

    x_k_half = X[:, 1:-1:2]
    u_k_half = U[:, 1:-1:2]

    x_k_1 = X[:, 2::2]
    u_k_1 = U[:, 2::2]

    f_k = f_(x_k, u_k)
    f_k_half = f_(x_k_half, u_k_half)
    f_k_1 = f_(x_k_1, u_k_1)

    opti.subject_to(x_k_1 - x_k == (h_k/6.) * (f_k + 4*f_k_half + f_k_1))
    opti.subject_to(x_k_half == 0.5*(x_k + x_k_1) + (h_k/8.)*(f_k - f_k_1))

    # Cost 
    cost = (h_k/6.) * sum2(u_k**2 + 4*u_k_half**2 + u_k_1**2)

    ## Minimize cost function
    opti.minimize(cost)

    """ Additional constraints """
    # Input constraint
    opti.subject_to(-30<=U)
    opti.subject_to(U<=30)

    # State constraint
    opti.subject_to(X[:,0] == x0)
    opti.subject_to(X[:,-1] == xf)

    opti.subject_to(-2.5 <= X[0,:])
    opti.subject_to(X[0,:] <= 2.5)

    """ Initial guess """
    if x_guess is None:
        ## Linear trajectory guess
        x_guess = np.hstack((np.linspace(x0[0], xf[0], N + (N-1)).reshape(-1,1),
                    np.zeros(N + (N-1)).reshape(-1,1),
                    np.linspace(x0[2], xf[2], N + (N-1)).reshape(-1,1),
                    np.zeros(N + (N-1)).reshape(-1,1),
                ))
        x_guess = x_guess.T

        # Set initial guess
        opti.set_initial(X, x_guess)
    else:
        # Set initial guess with guess provided
        opti.set_initial(X, x_guess)

    """ Solve! """
    opts = {'ipopt.print_level':0, 'print_time':0}
    opti.solver('ipopt', opts)      # Set numerical backend
    sol = opti.solve();   # actual sol

    x_sol = sol.value(X)
    u_sol = sol.value(U) 

    return x_sol, u_sol


