# This file includes the algorithm for two-dimensional optimisation of the channels of a soft touchpad. 
import cv2 
import numpy as np
import random
import matplotlib
import matplotlib.pyplot as plt
from time import time
import math
import sys
from sklearn import preprocessing

# np.random.seed(3)
np.random.seed()

# Hyperparameters
height = 50 # height of the touchpad
width  = 50 # width of the touchpad
max_iterations = 100 # maximum number of iterations
max_individuals = 100 # number of individuals, need to be even
length = 10
weight_c = 0.5
weight_u = 0.5
rulearray = ['A', 'B', '+', '-', '[', ']']
kernel_size = 101
division = (1000 - kernel_size)**2

def gkern(l=5, sig=1.):
	'''
	creates gaussian kernel with side length l and a sigma of sig
	'''
	ax = np.arange(-l // 2 + 1., l // 2 + 1.)
	xx, yy = np.meshgrid(ax, ax)

	kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(sig))

	return kernel #/ np.sum(kernel)

class GeneticAlgorithm:
	def __init__(self):
		self.fitness = 0
		self.genome = np.random.randint(low = 0, high = 6, size=(20))
		self.angle = np.random.randint(low = 0, high = 90, size=(1))
		# self.rules = np.random.randint(low = 0, high = 6, size=(20))
		# self.angle = np.random.randint(low = 0, high = 90, size=(1))
		# self.genome = np.concatenate((self.rules, self.angle), axis=None)
		# self.genome = [0, 3, 4, 4, 1, 5, 2, 1, 5, 2, 0, 4, 2, 0, 1, 5, 3, 1, 0, 0] # Existing L-systems rules: A-[[B]+B]+A[+AB]-BAA
		self.probability = 0
	
	def probabilities(self):
		fitsum  = 0

		for i in range(population.shape[0]):
			fitsum = fitsum + population[i].fitness

		for i in range(population.shape[0]):
				try:
					population[i].probability = population[i].fitness / fitsum
				except ZeroDivisionError:
					population[i].probability = 0.0
		return
	
	def rouletteWheel(self):

		newpopulation = np.array([np.copy(population[i].genome) for i in range(population.shape[0])])
		newangles = np.array([np.copy(population[i].angle) for i in range(population.shape[0])])
		noi = 0

		while noi < (population.shape[0]):
			# rp = np.random.rand() / population.shape[0] / 10
			rp = np.random.rand() / population.shape[0]
			print 'rp',rp

			for i in range(population.shape[0]):
				if noi < (population.shape[0]) and rp <= population[i].probability:
					newpopulation[noi] = population[i].genome
					newangles[noi] = population[i].angle
					noi += 1

		for i in range(population.shape[0]):
			population[i].genome = np.copy(newpopulation[i])
			population[i].angle = np.copy(newangles[i])

		int_list = np.linspace(0, population.shape[0] - 1, population.shape[0], dtype = 'uint8')
		newlist = np.copy(int_list)

		for i in range(population.shape[0]):
			newlist[np.random.randint(low = 0, high = population.shape[0])] = int_list[(np.random.randint(low = 0, high = population.shape[0]))]

		for i in range(population.shape[0]):
			newpopulation[i] = np.copy(population[i].genome)
			newangles[i] = np.copy(population[i].angle)

		for i in range(population.shape[0]):
			newpopulation[i] = np.copy(population[newlist[i]].genome)
			newangles[i] = np.copy(population[newlist[i]].angle)

		for i in range(population.shape[0]):
			population[i].genome  = np.copy(newpopulation[i])
			population[i].angle = np.copy(newangles[i])

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
			for j in range(len(population[0].genome)):
				if mutation[j] == 1:
					population[i].genome[j] = np.random.randint(low = 0, high = 6)
		return

	def fitnessfunction(self, covergences, uniquenesses):

		for i in range(max_individuals):
			population[i].fitness = weight_c * covergences[i] + weight_u * uniquenesses[i]  # max fitness = 1, min fitness = 0

		return

	def perf_meas_u(self, images):
		
		# kernel_size =51 # 3 x 3
		kernel = gkern(kernel_size)
		fitness_u = np.zeros(population.shape[0])
		index = 0
		
		for image in images:
			fitness_u[index] = sum(sum(cv2.filter2D(image, cv2.CV_64F, kernel)))
			index += 1
		# print fitness_u
		# fitness_u = np.array(preprocessing.normalize([fitness_u[i]]) for i in range(max_individuals))
		# fitness_u = preprocessing.normalize([fitness_u])
		# print fitness_u
		# sys.exit()
		

		# fitness_u = [fitness_u[i]/sum(fitness_u) for i in range(max_individuals)]
		return fitness_u / (sum(sum(kernel)) * division)

	def perf_meas_c(self, images):

		# kernel_size = 51 # 3 x 3
		fitness_c = np.zeros(population.shape[0])
		index = 0

		for image in images:
			for i in range(image.shape[0] - (kernel_size - kernel_size % 2)):
				for j in range(image.shape[0] - (kernel_size - kernel_size % 2)):	
					subimage = image[i:i+kernel_size][j:j+kernel_size]
					if cv2.countNonZero(subimage) >= 1:
						fitness_c[index] += 1
			index += 1
		# fitness_c = preprocessing.normalize([fitness_c])


		# fitness_c = [fitness_c[i]/sum(fitness_c) for i in range(max_individuals)]
		return fitness_c / division # ((image.shape[0] - (kernel_size - kernel_size % 2)) **2)

class L_Systems():
	def __init__(self):
		pass

	def expand_rules(self, iter_lsystem):

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
						new_string.append(rules[genes][0:18])
					elif old_string[k] == 'A':
						new_string.append(rules[genes][18:20])
					else:
						new_string.append(old_string[k])
				old_string = sep.join(new_string)
			strings_to_draw.append(old_string)
		return rules, strings_to_draw

	def drawing(self, expanded_strings):
		length = 10.0
		imgs =[]
		imgs_show = []
		errs = []

		for i in range(max_individuals):
			img = np.zeros((1000,1000), np.uint8)
			img_show = np.zeros((1000,1000), np.uint8)
			position = (500, 500) # (cols, rows): (0,0) is at top-left
			heading = math.radians(90) # init heading going directly down
			turn = math.radians(population[i].angle)
			stack = []
			err = 0

			try:
				for char in expanded_strings[i]:
					if char == 'A' or char == 'B':
						x_new = int(position[0]+length*math.cos(heading))
						y_new = int(position[1]+length*math.sin(heading))
						new_position = ( x_new, y_new )
						cv2.line(img,position,new_position,1,1)
						cv2.line(img_show,position,new_position,255,1)
						position = new_position
					elif char == '+':
						heading = heading + turn
					elif char == '-':
						heading = heading - turn
					elif char == '[':
						stack.append((position, heading))
					elif char == ']':
						position, heading = stack.pop()
					else:
						print 'Non'
			except Exception as e:
				# print "Genome not executable"
				err = 1

			imgs.append(img)
			imgs_show.append(img_show)
			errs.append(err)
		
		return imgs, imgs_show, errs

#####################################################################################################
## MAIN
#####################################################################################################
population = np.array([GeneticAlgorithm() for i in range(max_individuals)])

iteration = 0
geneticAlgorithm = GeneticAlgorithm()
lSystem = L_Systems()

globalmaxfitness = 0.0
globalmaxfitnessangle = 0.0
globalmaxfitnessrule = ''

while iteration < max_iterations:
	# print 'globalmaxfitnessrule', globalmaxfitnessrule, 'globalmaxfitness', globalmaxfitness, iteration
	rules, expanded_rules = lSystem.expand_rules(4)
	imgs,imgs_show, errs = lSystem.drawing(expanded_rules)
	uniqueness = geneticAlgorithm.perf_meas_u(imgs)
	covergence = geneticAlgorithm.perf_meas_c(imgs)
	geneticAlgorithm.fitnessfunction(covergence, uniqueness)

	'''
	show images and print genomes and fitnesses
	'''
	for item in range(max_individuals):
		cv2.imshow('Channels', imgs_show[item])
		cv2.waitKey(250)
		print 'iteration', iteration, 'gene', item+1, '/', max_individuals, 'rules', rules[item], 'angle', population[item].angle, 'fitness', population[item].fitness #, 'err', errs[item]
		if population[item].fitness > globalmaxfitness:
			globalmaxfitness = population[item].fitness
			globalmaxfitnessangle = population[item].angle
			globalmaxfitnessrule = rules[item]
	print '###################################################################'
	print 'globalmaxfitness: rule', globalmaxfitnessrule, 'angle', globalmaxfitnessangle, 'fitness', globalmaxfitness
	print '###################################################################'
	'''
	print average and max fitness
	'''
	fitnessarray = np.array([population[i].fitness for i in range(population.shape[0])])
	print 'Iteration: ',iteration, 'Avg fitness:', np.mean(fitnessarray), 'Max fitness: ', np.max(fitnessarray)

	geneticAlgorithm.probabilities()
	geneticAlgorithm.rouletteWheel()
	geneticAlgorithm.crossover()
	geneticAlgorithm.mutation()

	iteration += 1


cv2.destroyAllWindows()