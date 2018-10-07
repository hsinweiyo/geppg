import numpy as np
import matplotlib.pyplot as plt
import csv
x = []
y = []
with open('reacher_goal_act.csv', 'r') as f:
    plots = list(csv.reader(f, delimiter=' '))
    #print(plots)
    for row in plots:
        print(row)
        x.append(float(row[0]))
        y.append(float(row[3]))

plt.plot(x ,y)
plt.xlabel('Timesteps')
plt.ylabel('l2norm')
plt.title('Learning Curve')
plt.legend()
plt.savefig('reacher_goal_per.png')
plt.show()
