import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib

# algorithm for growth
def growing(array, iteration):
	for i in range(len(array)):
		array[i].append([array[i][len(array[i])-1][0] + np.random.uniform(-1, 1), array[i][len(array[i])-1][1] + np.random.uniform(-1, 1), array[i][len(array[i])-1][2] + np.random.uniform(-0.01, 0.2), iteration])
	return array
	

# dimensions of the touchpad
width_1 = 100
width_2 = 100
thickness = 30

# number of channels
n_channels = 10

# number of iterations
n_iterations = 100

# array of channels
array_of_channels = []

for i in range(n_channels):
	array_of_channels.append([[100*np.random.rand(), 100*np.random.rand(), 0, 0]])

for i in range(n_iterations):
	growing(array_of_channels, i)

# flattening
flat_list = [item for sublist in array_of_channels for item in sublist]

# converting to numpy array
array_of_channels = np.array(flat_list)

# extracting x, y and z values for plotting
x_values = array_of_channels[:, 0]
y_values = array_of_channels[:, 1]
z_values = array_of_channels[:, 2]
c_values = array_of_channels[:, 3]

print(c_values)

# colours
cmap = matplotlib.cm.get_cmap('viridis')
normalize = matplotlib.colors.Normalize(vmin=0, vmax=n_iterations)
colors = [cmap(normalize(value)) for value in c_values]

# plotting
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim3d(0, 100)
ax.set_ylim3d(0,100)
ax.set_zlim3d(0,100)
ax.scatter(x_values, y_values, z_values,  marker='o', s=2, color=colors)

plt.show()