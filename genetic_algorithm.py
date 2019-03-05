# This file includes the algorithm for two-dimensional optimisation of the channels of a soft touchpad. 

import cv2 
import numpy as np

# Hyperparameters
height = 50 # height of the touchpad
width  = 50 # width of the touchpad
maxt_iterations = 100000 # maximum number of iterations
max_individuals = 100 # number of individuals

# Fitness weights, min = 0, max = 1
w_unique  = 0.25 # uniqueness
w_cover   = 0.25 # coverage

touch_kernel = [3, 3] # effect of touch

# Blueprint of individuals
class Individual:
	def __init__(self):
		self.fitness = 0
		# Sequence representation as an OpenCV image object, BGR represent the three different layers 
		self.genome = np.random.choice([0, 255], size=(height, width, 3)).astype('uint8')

	def update_genome(self, new_genome):
		self.genome = new_genome

def connected_components(layer): 
	eq_list = [[],[]] # equivalency list
	components = np.array
	n_components = 0 # components found
			

def fitness_calculation(genome):
	# The fitness has four weighted parts: uniqueness, coverage, number of channels and unconnected pixels
	s_unique  = 0 # uniqueness
	s_cover   = 0 # coverage
	s_channel = 0 # number of channels
	s_alone   = 0 # unconnected channels

	# Unconnected channels: channels that are not connected to the edges receive penalty


	return w_unique * s_unique + w_cover * s_cover + w_channel * s_channel + w_alone * s_alone

# Creating instances
individuals = [Individual() for i in range(max_individuals)]

print(fitness_calculation(individuals[0].genome))

# Resizing image to show
image_to_show = cv2.resize(individuals[0].genome, (0,0), fx = 10, fy = 10)
cv2.imshow('Touchpad channels', image_to_show)
cv2.waitKey(0)
cv2.destroyAllWindows()