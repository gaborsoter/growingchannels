import cv2 
import numpy as np
import math
import sys
import numpy as np


def gkern(l=5, sig=1.):
    """
    creates gaussian kernel with side length l and a sigma of sig
    """

    ax = np.arange(-l // 2 + 1., l // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)
    print xx,yy

    kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(sig))
    print sum(sum(kernel))* 996004.0

    return kernel #/ np.sum(kernel)

print gkern(3)

sys.exit()


# final_string = 'A+A[-A+]+A'

rulearray = ['A', 'B', '+', '-', '[', ']']
iter_lsystem = 4

#################################################################
## mutation of existing L-systems rules
#################################################################
# char_array = A-[[B]+B]+A[+AB]-BAA
# gene = [0, 3, 4, 4, 1, 5, 2, 1, 5, 2, 0, 4, 2, 0, 1, 5, 3, 1, 0, 0]
#       A  -  [  [  B  ]  +  B  ]  +  B  [  +  A  B  ]  -  B  +  A
gene = [0, 3, 4, 4, 1, 5, 2, 1, 5, 2, 1, 4, 2, 0, 1, 5, 3, 1, 2, 0]
char_array = list(map(lambda x: rulearray[x], gene))
#################################################################
# char_array = list(map(lambda x: rulearray[x], population[i].genome))

count_bracket_open = 0
count_bracket_close = 0

sep = ''
rule = sep.join(char_array)
# rule = '[][A--B]A+B+[]][B[[+'
old_string = 'B'
for j in range(iter_lsystem):
	print old_string
	new_string = []
	for k in range(len(old_string)):
		if old_string[k] == 'B':
			new_string.append(rule[0:18])
		elif old_string[k] == 'A':
			new_string.append(rule[18:20])
		else:
			new_string.append(old_string[k])

	old_string = sep.join(new_string)

for item in old_string:
	if item == ']':
		count_bracket_close += 1
	if item == '[':
		count_bracket_open += 1
	if count_bracket_close > count_bracket_open:
		# if at any point POP outnumber APPEND then POP will error
		print(old_string)
		print count_bracket_open, count_bracket_close
		print 'ERROR: CLOSE BRACKET (POP) OUTNUMBER OPEN BRACKETS (APPEND)'
		sys.exit()

# final_string = 'BBBBBBBB-[[BBBB-[[BB-[[B-[[A]+A]+B[+BA]-A]+B-[[A]+A]+B[+BA]-A]+BB[+BBB-[[A]+A]+B[+BA]-A]-B-[[A]+A]+B[+BA]-A]+BB-[[B-[[A]+A]+B[+BA]-A]+B-[[A]+A]+B[+BA]-A]+BB[+BBB-[[A]+A]+B[+BA]-A]-B-[[A]+A]+B[+BA]-A]+BBBB[+BBBBBB-[[B-[[A]+A]+B[+BA]-A]+B-[[A]+A]+B[+BA]-A]+BB[+BBB-[[A]+A]+B[+BA]-A]-B-[[A]+A]+B[+BA]-A]-BB-[[B-[[A]+A]+B[+BA]-A]+B-[[A]+A]+B[+BA]-A]+BB[+BBB-[[A]+A]+B[+BA]-A]-B-[[A]+A]+B[+BA]-A]+BBBB-[[BB-[[B-[[A]+A]+B[+BA]-A]+B-[[A]+A]+B[+BA]-A]+BB[+BBB-[[A]+A]+B[+BA]-A]-B-[[A]+A]+B[+BA]-A]+BB-[[B-[[A]+A]+B[+BA]-A]+B-[[A]+A]+B[+BA]-A]+BB[+BBB-[[A]+A]+B[+BA]-A]-B-[[A]+A]+B[+BA]-A]+BBBB[+BBBBBB-[[B-[[A]+A]+B[+BA]-A]+B-[[A]+A]+B[+BA]-A]+BB[+BBB-[[A]+A]+B[+BA]-A]-B-[[A]+A]+B[+BA]-A]-BB-[[B-[[A]+A]+B[+BA]-A]+B-[[A]+A]+B[+BA]-A]+BB[+BBB-[[A]+A]+B[+BA]-A]-B-[[A]+A]+B[+BA]-A]+BBBBBBBB[+BBBBBBBBBBBB-[[BB-[[B-[[A]+A]+B[+BA]-A]+B-[[A]+A]+B[+BA]-A]+BB[+BBB-[[A]+A]+B[+BA]-A]-B-[[A]+A]+B[+BA]-A]+BB-[[B-[[A]+A]+B[+BA]-A]+B-[[A]+A]+B[+BA]-A]+BB[+BBB-[[A]+A]+B[+BA]-A]-B-[[A]+A]+B[+BA]-A]+BBBB[+BBBBBB-[[B-[[A]+A]+B[+BA]-A]+B-[[A]+A]+B[+BA]-A]+BB[+BBB-[[A]+A]+B[+BA]-A]-B-[[A]+A]+B[+BA]-A]-BB-[[B-[[A]+A]+B[+BA]-A]+B-[[A]+A]+B[+BA]-A]+BB[+BBB-[[A]+A]+B[+BA]-A]-B-[[A]+A]+B[+BA]-A]-BBBB-[[BB-[[B-[[A]+A]+B[+BA]-A]+B-[[A]+A]+B[+BA]-A]+BB[+BBB-[[A]+A]+B[+BA]-A]-B-[[A]+A]+B[+BA]-A]+BB-[[B-[[A]+A]+B[+BA]-A]+B-[[A]+A]+B[+BA]-A]+BB[+BBB-[[A]+A]+B[+BA]-A]-B-[[A]+A]+B[+BA]-A]+BBBB[+BBBBBB-[[B-[[A]+A]+B[+BA]-A]+B-[[A]+A]+B[+BA]-A]+BB[+BBB-[[A]+A]+B[+BA]-A]-B-[[A]+A]+B[+BA]-A]-BB-[[B-[[A]+A]+B[+BA]-A]+B-[[A]+A]+B[+BA]-A]+BB[+BBB-[[A]+A]+B[+BA]-A]-B-[[A]+A]+B[+BA]-A'
# final_string = 'BBBBBBBBBBBBBBBB-[[BBBBBBBB-[[BBBB-[[BB-[[B-[[A]+A]+B[+BA]-A]+B-[[A]+A]+B[+BA]-A]+BB[+BBB-[[A]+A]+B[+BA]-A]-B-[[A]+A]+B[+BA]-A]+BB-[[B-[[A]+A]+B[+BA]-A]+B-[[A]+A]+B[+BA]-A]+BB[+BBB-[[A]+A]+B[+BA]-A]-B-[[A]+A]+B[+BA]-A]+BBBB[+BBBBBB-[[B-[[A]+A]+B[+BA]-A]+B-[[A]+A]+B[+BA]-A]+BB[+BBB-[[A]+A]+B[+BA]-A]-B-[[A]+A]+B[+BA]-A]-BB-[[B-[[A]+A]+B[+BA]-A]+B-[[A]+A]+B[+BA]-A]+BB[+BBB-[[A]+A]+B[+BA]-A]-B-[[A]+A]+B[+BA]-A]+BBBB-[[BB-[[B-[[A]+A]+B[+BA]-A]+B-[[A]+A]+B[+BA]-A]+BB[+BBB-[[A]+A]+B[+BA]-A]-B-[[A]+A]+B[+BA]-A]+BB-[[B-[[A]+A]+B[+BA]-A]+B-[[A]+A]+B[+BA]-A]+BB[+BBB-[[A]+A]+B[+BA]-A]-B-[[A]+A]+B[+BA]-A]+BBBB[+BBBBBB-[[B-[[A]+A]+B[+BA]-A]+B-[[A]+A]+B[+BA]-A]+BB[+BBB-[[A]+A]+B[+BA]-A]-B-[[A]+A]+B[+BA]-A]-BB-[[B-[[A]+A]+B[+BA]-A]+B-[[A]+A]+B[+BA]-A]+BB[+BBB-[[A]+A]+B[+BA]-A]-B-[[A]+A]+B[+BA]-A]+BBBBBBBB[+BBBBBBBBBBBB-[[BB-[[B-[[A]+A]+B[+BA]-A]+B-[[A]+A]+B[+BA]-A]+BB[+BBB-[[A]+A]+B[+BA]-A]-B-[[A]+A]+B[+BA]-A]+BB-[[B-[[A]+A]+B[+BA]-A]+B-[[A]+A]+B[+BA]-A]+BB[+BBB-[[A]+A]+B[+BA]-A]-B-[[A]+A]+B[+BA]-A]+BBBB[+BBBBBB-[[B-[[A]+A]+B[+BA]-A]+B-[[A]+A]+B[+BA]-A]+BB[+BBB-[[A]+A]+B[+BA]-A]-B-[[A]+A]+B[+BA]-A]-BB-[[B-[[A]+A]+B[+BA]-A]+B-[[A]+A]+B[+BA]-A]+BB[+BBB-[[A]+A]+B[+BA]-A]-B-[[A]+A]+B[+BA]-A]-BBBB-[[BB-[[B-[[A]+A]+B[+BA]-A]+B-[[A]+A]+B[+BA]-A]+BB[+BBB-[[A]+A]+B[+BA]-A]-B-[[A]+A]+B[+BA]-A]+BB-[[B-[[A]+A]+B[+BA]-A]+B-[[A]+A]+B[+BA]-A]+BB[+BBB-[[A]+A]+B[+BA]-A]-B-[[A]+A]+B[+BA]-A]+BBBB[+BBBBBB-[[B-[[A]+A]+B[+BA]-A]+B-[[A]+A]+B[+BA]-A]+BB[+BBB-[[A]+A]+B[+BA]-A]-B-[[A]+A]+B[+BA]-A]-BB-[[B-[[A]+A]+B[+BA]-A]+B-[[A]+A]+B[+BA]-A]+BB[+BBB-[[A]+A]+B[+BA]-A]-B-[[A]+A]+B[+BA]-A]+BBBBBBBB-[[BBBB-[[BB-[[B-[[A]+A]+B[+BA]-A]+B-[[A]+A]+B[+BA]-A]+BB[+BBB-[[A]+A]+B[+BA]-A]-B-[[A]+A]+B[+BA]-A]+BB-[[B-[[A]+A]+B[+BA]-A]+B-[[A]+A]+B[+BA]-A]+BB[+BBB-[[A]+A]+B[+BA]-A]-B-[[A]+A]+B[+BA]-A]+BBBB[+BBBBBB-[[B-[[A]+A]+B[+BA]-A]+B-[[A]+A]+B[+BA]-A]+BB[+BBB-[[A]+A]+B[+BA]-A]-B-[[A]+A]+B[+BA]-A]-BB-[[B-[[A]+A]+B[+BA]-A]+B-[[A]+A]+B[+BA]-A]+BB[+BBB-[[A]+A]+B[+BA]-A]-B-[[A]+A]+B[+BA]-A]+BBBB-[[BB-[[B-[[A]+A]+B[+BA]-A]+B-[[A]+A]+B[+BA]-A]+BB[+BBB-[[A]+A]+B[+BA]-A]-B-[[A]+A]+B[+BA]-A]+BB-[[B-[[A]+A]+B[+BA]-A]+B-[[A]+A]+B[+BA]-A]+BB[+BBB-[[A]+A]+B[+BA]-A]-B-[[A]+A]+B[+BA]-A]+BBBB[+BBBBBB-[[B-[[A]+A]+B[+BA]-A]+B-[[A]+A]+B[+BA]-A]+BB[+BBB-[[A]+A]+B[+BA]-A]-B-[[A]+A]+B[+BA]-A]-BB-[[B-[[A]+A]+B[+BA]-A]+B-[[A]+A]+B[+BA]-A]+BB[+BBB-[[A]+A]+B[+BA]-A]-B-[[A]+A]+B[+BA]-A]+BBBBBBBB[+BBBBBBBBBBBB-[[BB-[[B-[[A]+A]+B[+BA]-A]+B-[[A]+A]+B[+BA]-A]+BB[+BBB-[[A]+A]+B[+BA]-A]-B-[[A]+A]+B[+BA]-A]+BB-[[B-[[A]+A]+B[+BA]-A]+B-[[A]+A]+B[+BA]-A]+BB[+BBB-[[A]+A]+B[+BA]-A]-B-[[A]+A]+B[+BA]-A]+BBBB[+BBBBBB-[[B-[[A]+A]+B[+BA]-A]+B-[[A]+A]+B[+BA]-A]+BB[+BBB-[[A]+A]+B[+BA]-A]-B-[[A]+A]+B[+BA]-A]-BB-[[B-[[A]+A]+B[+BA]-A]+B-[[A]+A]+B[+BA]-A]+BB[+BBB-[[A]+A]+B[+BA]-A]-B-[[A]+A]+B[+BA]-A]-BBBB-[[BB-[[B-[[A]+A]+B[+BA]-A]+B-[[A]+A]+B[+BA]-A]+BB[+BBB-[[A]+A]+B[+BA]-A]-B-[[A]+A]+B[+BA]-A]+BB-[[B-[[A]+A]+B[+BA]-A]+B-[[A]+A]+B[+BA]-A]+BB[+BBB-[[A]+A]+B[+BA]-A]-B-[[A]+A]+B[+BA]-A]+BBBB[+BBBBBB-[[B-[[A]+A]+B[+BA]-A]+B-[[A]+A]+B[+BA]-A]+BB[+BBB-[[A]+A]+B[+BA]-A]-B-[[A]+A]+B[+BA]-A]-BB-[[B-[[A]+A]+B[+BA]-A]+B-[[A]+A]+B[+BA]-A]+BB[+BBB-[[A]+A]+B[+BA]-A]-B-[[A]+A]+B[+BA]-A]+BBBBBBBBBBBBBBBB[+BBBBBBBBBBBBBBBBBBBBBBBB-[[BBBB-[[BB-[[B-[[A]+A]+B[+BA]-A]+B-[[A]+A]+B[+BA]-A]+BB[+BBB-[[A]+A]+B[+BA]-A]-B-[[A]+A]+B[+BA]-A]+BB-[[B-[[A]+A]+B[+BA]-A]+B-[[A]+A]+B[+BA]-A]+BB[+BBB-[[A]+A]+B[+BA]-A]-B-[[A]+A]+B[+BA]-A]+BBBB[+BBBBBB-[[B-[[A]+A]+B[+BA]-A]+B-[[A]+A]+B[+BA]-A]+BB[+BBB-[[A]+A]+B[+BA]-A]-B-[[A]+A]+B[+BA]-A]-BB-[[B-[[A]+A]+B[+BA]-A]+B-[[A]+A]+B[+BA]-A]+BB[+BBB-[[A]+A]+B[+BA]-A]-B-[[A]+A]+B[+BA]-A]+BBBB-[[BB-[[B-[[A]+A]+B[+BA]-A]+B-[[A]+A]+B[+BA]-A]+BB[+BBB-[[A]+A]+B[+BA]-A]-B-[[A]+A]+B[+BA]-A]+BB-[[B-[[A]+A]+B[+BA]-A]+B-[[A]+A]+B[+BA]-A]+BB[+BBB-[[A]+A]+B[+BA]-A]-B-[[A]+A]+B[+BA]-A]+BBBB[+BBBBBB-[[B-[[A]+A]+B[+BA]-A]+B-[[A]+A]+B[+BA]-A]+BB[+BBB-[[A]+A]+B[+BA]-A]-B-[[A]+A]+B[+BA]-A]-BB-[[B-[[A]+A]+B[+BA]-A]+B-[[A]+A]+B[+BA]-A]+BB[+BBB-[[A]+A]+B[+BA]-A]-B-[[A]+A]+B[+BA]-A]+BBBBBBBB[+BBBBBBBBBBBB-[[BB-[[B-[[A]+A]+B[+BA]-A]+B-[[A]+A]+B[+BA]-A]+BB[+BBB-[[A]+A]+B[+BA]-A]-B-[[A]+A]+B[+BA]-A]+BB-[[B-[[A]+A]+B[+BA]-A]+B-[[A]+A]+B[+BA]-A]+BB[+BBB-[[A]+A]+B[+BA]-A]-B-[[A]+A]+B[+BA]-A]+BBBB[+BBBBBB-[[B-[[A]+A]+B[+BA]-A]+B-[[A]+A]+B[+BA]-A]+BB[+BBB-[[A]+A]+B[+BA]-A]-B-[[A]+A]+B[+BA]-A]-BB-[[B-[[A]+A]+B[+BA]-A]+B-[[A]+A]+B[+BA]-A]+BB[+BBB-[[A]+A]+B[+BA]-A]-B-[[A]+A]+B[+BA]-A]-BBBB-[[BB-[[B-[[A]+A]+B[+BA]-A]+B-[[A]+A]+B[+BA]-A]+BB[+BBB-[[A]+A]+B[+BA]-A]-B-[[A]+A]+B[+BA]-A]+BB-[[B-[[A]+A]+B[+BA]-A]+B-[[A]+A]+B[+BA]-A]+BB[+BBB-[[A]+A]+B[+BA]-A]-B-[[A]+A]+B[+BA]-A]+BBBB[+BBBBBB-[[B-[[A]+A]+B[+BA]-A]+B-[[A]+A]+B[+BA]-A]+BB[+BBB-[[A]+A]+B[+BA]-A]-B-[[A]+A]+B[+BA]-A]-BB-[[B-[[A]+A]+B[+BA]-A]+B-[[A]+A]+B[+BA]-A]+BB[+BBB-[[A]+A]+B[+BA]-A]-B-[[A]+A]+B[+BA]-A]-BBBBBBBB-[[BBBB-[[BB-[[B-[[A]+A]+B[+BA]-A]+B-[[A]+A]+B[+BA]-A]+BB[+BBB-[[A]+A]+B[+BA]-A]-B-[[A]+A]+B[+BA]-A]+BB-[[B-[[A]+A]+B[+BA]-A]+B-[[A]+A]+B[+BA]-A]+BB[+BBB-[[A]+A]+B[+BA]-A]-B-[[A]+A]+B[+BA]-A]+BBBB[+BBBBBB-[[B-[[A]+A]+B[+BA]-A]+B-[[A]+A]+B[+BA]-A]+BB[+BBB-[[A]+A]+B[+BA]-A]-B-[[A]+A]+B[+BA]-A]-BB-[[B-[[A]+A]+B[+BA]-A]+B-[[A]+A]+B[+BA]-A]+BB[+BBB-[[A]+A]+B[+BA]-A]-B-[[A]+A]+B[+BA]-A]+BBBB-[[BB-[[B-[[A]+A]+B[+BA]-A]+B-[[A]+A]+B[+BA]-A]+BB[+BBB-[[A]+A]+B[+BA]-A]-B-[[A]+A]+B[+BA]-A]+BB-[[B-[[A]+A]+B[+BA]-A]+B-[[A]+A]+B[+BA]-A]+BB[+BBB-[[A]+A]+B[+BA]-A]-B-[[A]+A]+B[+BA]-A]+BBBB[+BBBBBB-[[B-[[A]+A]+B[+BA]-A]+B-[[A]+A]+B[+BA]-A]+BB[+BBB-[[A]+A]+B[+BA]-A]-B-[[A]+A]+B[+BA]-A]-BB-[[B-[[A]+A]+B[+BA]-A]+B-[[A]+A]+B[+BA]-A]+BB[+BBB-[[A]+A]+B[+BA]-A]-B-[[A]+A]+B[+BA]-A]+BBBBBBBB[+BBBBBBBBBBBB-[[BB-[[B-[[A]+A]+B[+BA]-A]+B-[[A]+A]+B[+BA]-A]+BB[+BBB-[[A]+A]+B[+BA]-A]-B-[[A]+A]+B[+BA]-A]+BB-[[B-[[A]+A]+B[+BA]-A]+B-[[A]+A]+B[+BA]-A]+BB[+BBB-[[A]+A]+B[+BA]-A]-B-[[A]+A]+B[+BA]-A]+BBBB[+BBBBBB-[[B-[[A]+A]+B[+BA]-A]+B-[[A]+A]+B[+BA]-A]+BB[+BBB-[[A]+A]+B[+BA]-A]-B-[[A]+A]+B[+BA]-A]-BB-[[B-[[A]+A]+B[+BA]-A]+B-[[A]+A]+B[+BA]-A]+BB[+BBB-[[A]+A]+B[+BA]-A]-B-[[A]+A]+B[+BA]-A]-BBBB-[[BB-[[B-[[A]+A]+B[+BA]-A]+B-[[A]+A]+B[+BA]-A]+BB[+BBB-[[A]+A]+B[+BA]-A]-B-[[A]+A]+B[+BA]-A]+BB-[[B-[[A]+A]+B[+BA]-A]+B-[[A]+A]+B[+BA]-A]+BB[+BBB-[[A]+A]+B[+BA]-A]-B-[[A]+A]+B[+BA]-A]+BBBB[+BBBBBB-[[B-[[A]+A]+B[+BA]-A]+B-[[A]+A]+B[+BA]-A]+BB[+BBB-[[A]+A]+B[+BA]-A]-B-[[A]+A]+B[+BA]-A]-BB-[[B-[[A]+A]+B[+BA]-A]+B-[[A]+A]+B[+BA]-A]+BB[+BBB-[[A]+A]+B[+BA]-A]-B-[[A]+A]+B[+BA]-A'
final_string = old_string
length = 10.0
img = np.zeros((1000,1000), np.uint8)
position = (500, 500) # (cols, rows): (0,0) is at top-left
heading = math.radians(90) # init heading going directly down
turn = math.radians(25)
stack = []

for item in final_string:
	if item == 'A' or item == 'B':
		x_new = int(position[0]+length*math.cos(heading))
		y_new = int(position[1]+length*math.sin(heading))
		new_position = ( x_new, y_new )
		cv2.line(img,position,new_position,255,1)
		position = new_position
		# print '[ FRWD ] ', position
	elif item == '+':
		heading = heading + turn
		# print '[ RGHT ] ', math.degrees(turn_right)
	elif item == '-':
		heading = heading - turn
		# print '[ LEFT ] ', math.degrees(turn_left)
	elif item == '[':
		stack.append((position, heading))
		# print '[ APPEND ]', stack
	elif item == ']':
		position, heading = stack.pop() #len(stack)-1
		# print '[ POP  ] ', position, heading
	else:
		print '[ NOP  ] ', codebit

cv2.imshow('Channels', img)
cv2.waitKey(0)