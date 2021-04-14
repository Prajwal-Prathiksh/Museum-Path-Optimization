###########################################################################
# In main directory
# Usage: '$ python code/branch_and_bound/post_processing.py'
###########################################################################
# Imports
###########################################################################
import __init__
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


###########################################################################
# Code
###########################################################################

results = np.load(
    "output/branch_and_bound/BnB/BnB_results.npz", allow_pickle=True
)

tour = results["tour"]
full_tour = results["full_tour"]

# coordi = np.random.rand(len(tour), 2) * 10
coordi = np.array(
    [
        [0, 248],
        [290, 248],
        [365, 248],
        [290, 184],
        [365, 184],
        [440, 248],
        [440, 184],
        [504, 184],
        [624, 184],
        [624, 64],
        [504, 64],
        [440, 64],
        [440, 0],
        [365, 64],
        [290, 64],
        [365, 0],
        [290, 0],
        [0, 0],
    ]
)

# coordi[1]

fig, ax = plt.subplots()

xpt, ypt = [], []
for edge in full_tour[-1]:
    i, j = edge
    xpt.append([coordi[i][0], coordi[j][0]])
    ypt.append([coordi[i][1], coordi[j][1]])
plt.plot(xpt, ypt, "or")

(tour_plot,) = plt.plot([], [], "-ob")


def init():
    # ax.set_xlim(-10, 10)
    # ax.set_ylim(-10, 10)
    return (tour_plot,)


def next_tour(tour_num):
    xdata, ydata = [], []
    for edge in full_tour[tour_num]:
        i, j = edge
        xdata.append([coordi[i][0], coordi[j][0]])
        ydata.append([coordi[i][1], coordi[j][1]])
    tour_plot.set_data(xdata, ydata)
    return (tour_plot,)


tour_ani = animation.FuncAnimation(
    fig, next_tour, len(full_tour), interval=1000, init_func=init, blit=True
)

writer = animation.FFMpegWriter(fps=4)
tour_ani.save("output/branch_and_bound/BnB/tour_ani_new.mp4", writer=writer)

# plt.plot(coordi[1][0],coordi[1][1])
plt.show()
