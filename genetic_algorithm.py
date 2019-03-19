# This file includes the algorithm for two-dimensional optimisation of the channels of a soft touchpad. 
import cv2 
import numpy as np
import random
import matplotlib
import matplotlib.pyplot as plt
import time

np.random.seed(3)

# Hyperparameters
height = 50 # height of the touchpad
width  = 50 # width of the touchpad
max_iterations = 100000 # maximum number of iterations
max_individuals = 1 # number of individuals, need to be even
length = 10
iter_lsystem = 2
rulearray = ['A', 'B', '+', '-', ']', '[']


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
			mutation = np.random.choice(2, 8, p = [0.99, 0.01])
			for j in range((population[0].genome).shape[0]):
				if mutation[j] == 1:
					population[i].genome[j] = np.random.randint(low = 0, high = 7)

	def fitnessfunction(self):
		fitnessarray = np.zeros(population.shape[0], dtype = 'uint8')

		for k in range(population.shape[0]):
			for i in range(7):
				for j in range(i+1, 8):
					if population[k].genome[i] != population[k].genome[j] and population[k].genome[i] != (population[k].genome[j] + (j-i)) and population[k].genome[i] != (population[k].genome[j] - (j - i)):
						fitnessarray[k] += 1		

		for i in range(population.shape[0]):
			population[i].fitness = fitnessarray[i]	

		return

	def l_system(self):
		char_array = list(map(lambda x: rulearray[x], population[i].genome))
		sep = ''
		rule = sep.join(char_array)
		old_string = 'A'
		for j in range(iter_lsystem):
			new_string = []
			for k in range(len(old_string)):
				if old_string[k] == 'A':
					new_string.append(rule[0:18])
				elif old_string[k] == 'B':
					new_string.append(rule[18:20])
				else:
					new_string.append(old_string[k])
		
			old_string = sep.join(new_string)

		return old_string

	def drawing(self, final_string):
		image = np.zeros(1000, 1000)
		position = [0, 500]

		for item in final_string:
			if item == 'A':
				


class Individual_Lsystem():
	def __init__(self):
		self.fitness = 0
		self.genome = np.random.randint(low = 0, high = 6, size=(20))
		self.probability = 0
	
# Creating objects
population = np.array([Individual_Lsystem() for i in range(max_individuals)])


geneticAlgorithm = GeneticAlgorithm()

print(geneticAlgorithm.l_system())

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