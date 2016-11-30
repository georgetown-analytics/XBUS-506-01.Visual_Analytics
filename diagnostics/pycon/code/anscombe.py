# Writing the Anscomb visualization code

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Create the Anscombe arrays
i   = np.array([
    [10.0, 8.0, 13.0, 9.0, 11.0, 14.0, 6.0, 4.0, 12.0, 7.0, 5.0],
    [8.04, 6.95, 7.58, 8.81, 8.33, 9.96, 7.24, 4.26, 10.84, 4.82, 5.68]
])

ii  = np.array([
    [10.0, 8.0, 13.0, 9.0, 11.0, 14.0, 6.0, 4.0, 12.0, 7.0, 5.0],
    [9.14, 8.14, 8.74, 8.77, 9.26, 8.10, 6.13, 3.10, 9.13, 7.26, 4.74]
])

iii = np.array([
    [10.0, 8.0, 13.0, 9.0, 11.0, 14.0, 6.0, 4.0, 12.0, 7.0, 5.0],
    [7.46, 6.77, 12.74, 7.11, 7.81, 8.84, 6.08, 5.39, 8.15, 6.42, 5.73]
])

iv  = np.array([
    [8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 19.0, 8.0, 8.0, 8.0],
    [6.58, 5.76, 7.71, 8.84, 8.47, 7.04, 5.25, 12.50, 5.56, 7.91, 6.89]
])


def make_plot(a, b, c, d):
    """
    Creates a grid plot of four datasets such that (a, b) and (c,d) share
    the same Y axis and (a,c) and (b,d) share the same X axis.
    """
    fig, ((axa, axb), (axc, axd)) =  plt.subplots(2, 2, sharex='col', sharey='row')
    for arr, ax in ((a, axa), (b, axb), (c, axc), (d, axd)):
        x = arr[0]
        y = arr[1]

        ax.scatter(x, y, c='#2B94E9')
        m,b = np.polyfit(x, y, 1)
        X = np.linspace(ax.get_xlim()[0], ax.get_xlim()[1], 100)
        ax.plot(X, m*X+b, '-', c='#666666')
    plt.savefig("/Users/pepper/Desktop/Projects/DDL/pycon2016/vizMLslides/images/anscombe.png",transparent=True)

make_plot(i, ii, iii, iv)
