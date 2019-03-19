import cv2 
import numpy as np
import math

# length = 10.0
# img = np.zeros((1000,1000), np.uint8)
# line_thickness = 1

# angle = 90
# angle_rad = math.radians(angle)
# position = (500, 0)
# x_new = int(position[0]+length*math.cos(angle_rad))
# y_new = int(position[1]+length*math.sin(angle_rad))
# new_position = ( x_new, y_new )
# cv2.line(img,position,new_position,255,line_thickness)


# angle = angle+25
# angle_rad = math.radians(angle)
# position = new_position
# x_new = int(position[0]+length*math.cos(angle_rad))
# y_new = int(position[1]+length*math.sin(angle_rad))
# new_position = ( x_new, y_new )
# cv2.line(img,position,new_position,255,line_thickness)

# angle = angle-50
# angle_rad = math.radians(angle)
# position = new_position
# x_new = int(position[0]+length*math.cos(angle_rad))
# y_new = int(position[1]+length*math.sin(angle_rad))
# new_position = ( x_new, y_new )
# cv2.line(img,position,new_position,255,line_thickness)

# cv2.imshow('Channels', img)
# cv2.waitKey(0)

final_string = 'A+A[]-A++A'

length = 10.0
img = np.zeros((1000,1000), np.uint8)
position = (500, 0) # (cols, rows): (0,0) is at top-left
heading = math.radians(90) # init heading going directly down
turn_left = math.radians(25)
turn_right = math.radians(-25)
stack = []
for item in final_string:
	if item == 'A':
		x_new = int(position[0]+length*math.cos(heading))
		y_new = int(position[1]+length*math.sin(heading))
		new_position = ( x_new, y_new )
		cv2.line(img,position,new_position,255,1)
		position = new_position
		print '[ FRWD ] ', length
	elif item == '+':
		heading = heading + turn_right
		print '[ RGHT ] ', math.degrees(turn_right)
	elif item == '-':
		heading = heading + turn_left
		print '[ LEFT ] ', math.degrees(turn_left)
	elif item == '[':
		stack.append((position, heading))
		print '[ APPEND ]', stack
	elif item == ']':
		position = stack[len(stack)-1][0]
		heading =  stack[len(stack)-1][1]
		print '[ POP  ] ', (position, heading)
		stack.pop(len(stack)-1)
		print '[ POP  ] ', stack

cv2.imshow('Channels', img)
cv2.waitKey(0)
