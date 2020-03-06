import cv2
import numpy as np
import sys
import operator
import math
import time
from matplotlib import pyplot as plt

# accumulator = {tuple([0,0,0]): 0}

def HoughXfm_Circle(x_points, y_points):
	x_center_points = np.arange(min(x_points), max(x_points), 1)
	y_center_points = np.arange(min(y_points), max(y_points), 1)

	accumulator = np.zeros((len(x_center_points), len(y_center_points), len(x_center_points)))
	for x in range(len(x_points)):
		for y in range(len(y_points)):
			for xc in range(len(x_center_points)):
				for yc in range(len(y_center_points)):
					radius_sqrd = (x_points[x] - x_center_points[xc])**2 + (y_points[y] - y_center_points[yc])**2
					accumulator[x_center_points[xc], y_center_points[yc], radius_sqrd] = accumulator[x_center_points[xc], y_center_points[yc], radius_sqrd] + 1 
					# key = tuple([x_center_points[xc], y_center_points[yc], radius_sqrd])
					# if key not in accumulator.keys():
					# 	accumulator[key] = 1
					# else:
					# 	accumulator[key] = accumulator[key] + 1
	print("Max Votes: ", numpy.where(accumulator == numpy.amax(accumulator)))
	# max_votes = max(accumulator.items(), key=operator.itemgetter(1))[0]
	# values = accumulator.values()
	# keys = accumulator.keys()
	# max_val = max(values)
	# for key in keys:
	# 	if accumulator[key] == values:
	# 		print(max_votes, ", ", accumulator[max_votes])

def test():
	radius = 5.0
	xc = 10.0
	yc = 10.0
	x_points = []
	y_points = []
	theta = np.arange(0, 2*np.pi, np.pi/64)
	for t in range(len(theta)):
		x = xc + (radius * np.cos(theta[t]))
		y = yc + (radius * np.sin(theta[t]))
		x_points.append(x)
		y_points.append(y)

	tic = time.perf_counter()
	HoughXfm_Circle(x_points, y_points)
	toc = time.perf_counter()
	print(f"Computation time {toc - tic:0.4f}")

if __name__ == '__main__':
	img = cv2.imread('circlesAndLines.jpg', 0)
	edges = cv2.Canny(img, 50, 150, apertureSize=3)
	x_list = []
	y_list = []
	for i in range(np.size(edges, 0)):
		for j in range(np.size(edges, 1)):
			if edges[i,j] != 0.0:
				x_list.append(i)
				y_list.append(j)

	# print(len(x_list[1::1000]))
	x_list = x_list[1::1000]
	y_list = y_list[1::1000]
	# HoughXfm_Circle(x_list, y_list)
	test()
	plt.subplot(121),plt.imshow(img, cmap='gray')
	plt.title('Original Image'), plt.xticks([]), plt.yticks([])
	plt.subplot(122),plt.imshow(edges,cmap = 'gray')
	plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
	plt.show()