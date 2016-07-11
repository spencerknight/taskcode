'''
Plotting tools for gps/accel data visualizations.
'''

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def errorband(x,y,yerr,ax=None, **kwargs):
    '''
    Plot x vs y with a shaded band to represent yerr.

    Kwargs:
    ax    -- matplotlib.Axes instance to draw on

    Remaining args are passed to call to matplotlib.pyplot.plot
    '''
    if ax is None:
        ax = plt.gca()
    line = ax.plot(x,y, **kwargs)[0]
    band = ax.fill_between(x, y-yerr, y+yerr, facecolor=line.get_color(), alpha=0.5)


def main():
    # Run a few examples
    x = np.linspace(0,4*np.pi,30)
    y = np.sin(x) + 0.05*np.random.randn(30)
    yerr = 0.25*np.random.rand(30) + 0.1
    
    fig, ax = plt.subplots()
    errorband(x,y,yerr,ax=ax, label='Example')
    plt.legend()
    plt.show()
