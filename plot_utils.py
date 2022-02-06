import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm

def plot_control_output(x_ctrl_out, u_ctrl_out, x_ref):
    # TODO: incorporate control output plots
    fig, axs = plt.subplots(x_ref.shape[0], 1)
    color = cm.rainbow(np.linspace(0, 1, x_ref.shape[0]))

    nx, N = x_ctrl_out.shape     # Shape (nx, N)
    for i in range(x_ref.shape[0]):
        axs[i].plot(np.full(N, x_ref[i]), '--', c=color[i])
        axs[i].plot(x_ctrl_out[i,:], '-', c=color[i])  

    # Plot control output
    # TODO: incorporate into axs plot
    fig2, axs2 = plt.subplots(1)
    axs2.plot(u_ctrl_out[:], 'ro-')

    #plt.show()
