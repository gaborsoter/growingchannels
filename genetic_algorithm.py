# This file includes the algorithm for two-dimensional optimisation of the channels of a soft touchpad. 
import cv2 
import numpy as np
import random
import matplotlib
import matplotlib.pyplot as plt
from time import time
import math
import sys

# np.random.seed(3)
np.random.seed()

# Hyperparameters
height = 50 # height of the touchpad
width  = 50 # width of the touchpad
max_iterations = 1000 # maximum number of iterations
max_individuals = 10 # number of individuals, need to be even
length = 10
weight_c = 0.5
weight_u = 0.5
rulearray = ['A', 'B', '+', '-', '[', ']']

def gkern(l=5, sig=1.):
	##############################################################################
	## creates gaussian kernel with side length l and a sigma of sig
	##############################################################################
    ax = np.arange(-l // 2 + 1., l // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)

    kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(sig))

    return kernel #/ np.sum(kernel)

# Blueprint of individuals
class GeneticAlgorithm:
	def __init__(self):
		pass
	
	def probabilities(self):
		fitsum  = 0

		for i in range(population.shape[0]):
			fitsum = fitsum + population[i].fitness

		for i in range(population.shape[0]):
			population[i].probability = population[i].fitness / fitsum		

		return

	
	def rouletteWheel(self):
		newpopulation = np.array([np.copy(population[i].genome) for i in range(population.shape[0])])
		noi = 0

		while noi < (population.shape[0]):
			rp = np.random.rand() / population.shape[0] / 10

			for i in range(population.shape[0]):
				if noi < (population.shape[0]) and rp <= population[i].probability:
					newpopulation[noi] = population[i].genome
					noi += 1

		for i in range(population.shape[0]):
			population[i].genome = np.copy(newpopulation[i])

		int_list = np.linspace(0, population.shape[0] - 1, population.shape[0], dtype = 'uint8')
		newlist = np.copy(int_list)

		for i in range(population.shape[0]):
			newlist[np.random.randint(low = 0, high = population.shape[0])] = int_list[(np.random.randint(low = 0, high = population.shape[0]))]

		for i in range(population.shape[0]):
			newpopulation[i] = np.copy(population[i].genome)

		for i in range(population.shape[0]):
			newpopulation[i] = np.copy(population[newlist[i]].genome)

		for i in range(population.shape[0]):
			population[i].genome  = np.copy(newpopulation[i])

		return
	
	
	def crossover(self):
		split = np.zeros(int(population.shape[0] / 2), dtype = 'uint8')

		for i in range(int(population.shape[0] / 2)):
			split[i] = np.random.randint(low = 0, high = population.shape[0] - 1)

		for i in range(int(population.shape[0] / 2)):
			crossoverrate = np.random.choice([0, 1], 1, p = [0.3, 0.7])

			if crossoverrate[0] == 1:
				pop1 = np.copy(population[i * 2].genome)
				pop2 = np.copy(population[i * 2 - 1].genome)
				for k in range(split[i] + 1, 8):
					population[i * 2].genome[k] = pop2[k]
					population[i * 2 - 1].genome[k] = pop1[k]

		return

	def mutation(self):
		for i in range(population.shape[0]):
			mutation = np.random.choice(2, 20, p = [0.99, 0.01])
			# for j in range((population[0].genome).shape[0]):
			for j in range(len(population[0].genome)):
				if mutation[j] == 1:
					population[i].genome[j] = np.random.randint(low = 0, high = 6)

	def fitnessfunction(self, covergence, uniqueness):
		fitnessarray = np.zeros(population.shape[0], dtype = 'uint8')

		fitness = weight_c * covergence + weight_u * uniqueness  # max fitness = 1, min fitness = 0

		# print "Fitness:", fitness

		# for k in range(population.shape[0]):
		# 	for i in range(19):
		# 		for j in range(i+1, 20):
		# 			if population[k].genome[i] != population[k].genome[j] and population[k].genome[i] != (population[k].genome[j] + (j-i)) and population[k].genome[i] != (population[k].genome[j] - (j - i)):
		# 				fitnessarray[k] += 1		

		# for i in range(population.shape[0]):
		# 	population[i].fitness = fitnessarray[i]	

		return fitness

	def perf_meas_u(self, image):

		kernel_size = 3 # 3 x 3
		fitness_u = 0
		# kernel = [[1, 3, 1],
		# 		  [3, 5, 3],
		# 		  [1, 3, 1]]
		kernel = gkern(kernel_size)
		fitness_u = sum(sum(cv2.filter2D(image, cv2.CV_64F, kernel)))

		return fitness_u / (sum(sum(kernel)) * 996004.0)

	def perf_meas_c(self, image):

		kernel_size = 3 # 3 x 3
		fitness_c = 0
		for i in range(image.shape[0] - (kernel_size - kernel_size % 2)):
			for j in range(image.shape[0] - (kernel_size - kernel_size % 2)):	
				subimage = image[i:i+kernel_size][j:j+kernel_size]
				if cv2.countNonZero(subimage) >= 1:
					fitness_c += 1
		return fitness_c / 996004.0 # ((image.shape[0] - (kernel_size - kernel_size % 2)) **2)

	def l_system(self, iter_lsystem):

		char_array = np.array([list(map(lambda x: rulearray[x], population[i].genome)) for i in range(max_individuals)])
		# char_array = list(map(lambda x: rulearray[x], population[i].genome))
		sep = ''
		strings_to_draw = []#np.array(["" for i in range(max_individuals)])
		rules = np.array([sep.join(char_array[i]) for i in range(max_individuals)])
		for genes in range(max_individuals):
			old_string = 'B'
			for j in range(iter_lsystem):
				new_string = []
				for k in range(len(old_string)):
					if old_string[k] == 'B':
						new_string.append(rules[i][0:18])
					elif old_string[k] == 'A':
						new_string.append(rules[i][18:20])
					else:
						new_string.append(old_string[k])
				old_string = sep.join(new_string)
			strings_to_draw.append(old_string)

		return rules, strings_to_draw

	def drawing(self, final_string):
		length = 10.0
		img = np.zeros((1000,1000), np.uint8)
		position = (500, 500) # (cols, rows): (0,0) is at top-left
		heading = math.radians(90) # init heading going directly down
		turn = math.radians(25)
		stack = []
		err = 0

		try:
			for item in final_string:
				if item == 'A' or item == 'B':
					x_new = int(position[0]+length*math.cos(heading))
					y_new = int(position[1]+length*math.sin(heading))
					new_position = ( x_new, y_new )
					cv2.line(img,position,new_position,1,1)
					# cv2.line(img,position,new_position,255,1)
					position = new_position
				elif item == '+':
					heading = heading + turn
				elif item == '-':
					heading = heading - turn
				elif item == '[':
					stack.append((position, heading))
				elif item == ']':
					position, heading = stack.pop()
				else:
					print 'Non'
		except Exception as e:
			# print "Genome not executable"
			err = 1

		######################################################################################
		## TO VISUALISE, CHANGE CV2.LINE(IMG,POS,NEWPOS,1,1) T0 CV2.LINE(IMG,POS,NEWPOS,255,1)
		######################################################################################
		# cv2.imshow('Channels', img)
		# cv2.waitKey(5)
		return img, err


class Individual_Lsystem():
	def __init__(self):
		self.fitness = 0
		self.genome = np.random.randint(low = 0, high = 6, size=(20))
		# self.genome = [0, 3, 4, 4, 1, 5, 2, 1, 5, 2, 0, 4, 2, 0, 1, 5, 3, 1, 0, 0] # Existing L-systems rules: A-[[B]+B]+A[+AB]-BAA
		self.probability = 0
	
# Creating objects
population = np.array([Individual_Lsystem() for i in range(max_individuals)])
# print population[0].genome

fitnessarray = []
previous_string = ''
iteration = 0
geneticAlgorithm = GeneticAlgorithm()

while iteration < max_iterations:
	
	# print '-- Iteration',iteration,'--'
	
	genes, strings_to_draw = geneticAlgorithm.l_system(4)

	for item in range(max_individuals):
		img, err = geneticAlgorithm.drawing(strings_to_draw[item])
		if strings_to_draw[item] != previous_string and err == 0:
			# print 'Rules:', genes[item]
			covergence = geneticAlgorithm.perf_meas_c(img)
			uniqueness = geneticAlgorithm.perf_meas_u(img)
			fitness = geneticAlgorithm.fitnessfunction(covergence, uniqueness)
		else:
			fitness = 0.0
		fitnessarray.append([genes[item], fitness])
		print 'iteration', iteration, 'fitnessarray', fitnessarray[item]
	sys.exit()
	geneticAlgorithm.crossover()
	geneticAlgorithm.mutation()
	previous_string = strings_to_draw

	iteration += 1

# cv2.imshow('Channels', geneticAlgorithm.l_system())
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# # Fitness array
# fitness_array = np.array([population[i].fitness for i in range(population.shape[0])])

# iteration = 0


# while iteration < max_iterations:
# 	geneticAlgorithm.fitnessfunction()
# 	geneticAlgorithm.probabilities()
# 	geneticAlgorithm.rouletteWheel()
# 	geneticAlgorithm.crossover()
# 	geneticAlgorithm.mutation()

# 	fitness_array = np.array([population[i].fitness for i in range(population.shape[0])])

# 	avg = np.mean(fitness_array)
# 	maximum = np.max(fitness_array)

# 	if (28 in fitness_array) == True:
# 		print(population[fitness_array.index(28)].genome)
# 		break

# 	print('Iteration: ',iteration, 'Avg fitness:', avg, 'Max fitness: ', maximum)

# 	iteration += 1

# plt.show()