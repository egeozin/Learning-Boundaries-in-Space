from GridCell import GridCell
import math
import numpy as np
def getFeatures (grid_list, bound_list,hd):
	#NORMALIZE DATA AND RETURN FULL FEATURE MATRIX
	hd= hd + np.pi
	hd_x = np.cos(hd)
	hd_y = np.sin(hd)
	bound_list.pop()
	grid_activation=np.zeros(len(grid_list))
	bound_activation=np.zeros(len(bound_list))
	head_activation = np.zeros(len(bound_list))
	
	print(grid_activation.shape)
	print(bound_activation.shape)

	for idx,g in enumerate(grid_list):
		grid_activation[idx]=g.activation+0.5
	
	for idx,b in enumerate(bound_list):
		bound_activation[idx]=4.0 / b**2


	bound_activation = np.minimum(1, bound_activation)

	for idx,h in enumerate (head_activation):
		shd_x = np.cos(idx * 2*np.pi/ len(bound_list))
		shd_y = np.sin(idx * 2*np.pi/ len(bound_list))
		dist = np. dot([hd_x,hd_y],[shd_x,shd_y])
		dist = (dist+1.0)/2
		head_activation[idx]=dist
	
	featureRep=np.concatenate((grid_activation, bound_activation, head_activation))
	return grid_activation


def generateGrids(numGrids):
	Grids=[]
	f=0.3
	pD=0.0
	for i in range(numGrids):
		Grids.append(GridCell(pD,pD,f))
		f  += (0.7/numGrids)
		pD += np.pi/4;
	return Grids

def updateGrids(Grids, x ,y):
	for g in Grids:
		g.update(x,y)