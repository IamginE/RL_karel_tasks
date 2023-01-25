import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple

def plot(yvals:list, ylim:Tuple[int, int], x_ticks:list, names:list, out:str, title:str, xaxis:str, 
    yaxis:str, linspace_mul=1, start_origin=False):
    r""""Plots multiple line plots for the specified yvals in one plot to the output file 'out'."""
    fig, axs = plt.subplots(1, 1, figsize=(10, 8), squeeze=False)
    
    if start_origin:
        x = linspace_mul*np.linspace(0, len(yvals[0])-1, num = len(yvals[0]))
    else:
        x = linspace_mul*np.linspace(1, len(yvals[0]), num = len(yvals[0]))
    for k in range(len(yvals)):
        label = names[k]
        axs[0,0].plot(x, yvals[k], label=label)
        
    axs[0,0].legend(loc="best")
    axs[0,0].set_xticks(x_ticks)
    axs[0,0].set_title(title)
    axs[0,0].set_xlabel(xaxis)
    axs[0,0].set_ylabel(yaxis)
    axs[0,0].set_ylim(ylim)
    plt.tight_layout()
    plt.savefig(out)