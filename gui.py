''' gui.py
Displays training information in real time.
'''
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='whitegrid')
import numpy as np

# need at least 2 points to form a line plot,
# otherwise use scatter plot
size = 2

# create vectors to store plot values
x_vec = np.zeros(size)
y_vec = np.zeros(size)
x_cnt = 0
prev_rand_val = 0

# setup figure
plt.figure()
plt.title('Cycle-of-Learning')
plt.xlabel('Time Step')
plt.ylabel('Reward')

for _ in range(200):
    # load data value to plot
    rand_val = np.random.randn(1)*0.5 + np.log(x_cnt+1)

    # append to the end of array to be plotted
    y_vec[-1] = rand_val
    x_vec[-1] = x_cnt

    # plot
    if size == 1:
        plt.scatter(x_vec, y_vec, color='tab:blue', alpha=0.5)
    else: 
        plt.plot(x_vec, y_vec, color='tab:blue', marker='o', linestyle='--', alpha=0.5)

    # append new data to previous to make a continuous plot
    y_vec = np.append(y_vec[1:],0.0)
    x_vec = np.append(x_vec[1:],0.0)

    # draw lines and update counters
    plt.pause(0.01)
    x_cnt += 1

plt.show()