#!/usr/bin/env python
import copy
import random
import math
import random
import sys
def dis(vect1, vect2):
	res = 0;
	for i in range(len(vect1)):
		vectsum = 0;
		for j in range(len(vect1[0])):
			vectsum += (vect1[i][j] - vect2[i][j])**2;
		res += math.sqrt(vectsum);
	return res;

def randomcenter(dataset, k):
	print("dataset sample number for clustering is %s and k is: %s"%(len(dataset), k))
	centers = [];
	s = set();
	while(len(s)<k):
		m = int(float(random.random())*len(dataset) )
		s.add(m);
	for s0 in s:
		centers.append(dataset[s0]);
	print("randomly seleted k centers")
	return centers;

def average(vects):
	if(len(vects)==0):
		return 0;
	t = [0, 0, 0, 0 ,0];
	res = []
	for i in range(len(vects[0])):
		res.append(copy.deepcopy(t));
	for j in range(len(vects[0])):
		for k in range(len(vects[0][0])):
			for i in range(len(vects)):
				res[j][k] += vects[i][j][k];
			res[j][k] = res[j][k]/len(vects);
	return res;

def renewcenters(centers, assignment, k, dataset):
	for i in range(k):
		temp = [];
		for j in range(len(assignment)):
			if(assignment[j] == i):
				temp.append(dataset[j]);
		newtemp = average(temp);
		if(newtemp != 0):
			centers[i] = newtemp;
			

def reallocate(assignment, k, dataset, centers):
	flag = False;
	for i in range(len(dataset)):
		mindist = sys.maxint 
		lastassign = 0;
		original  = assignment[i];
		for j in range(k):
			dist = dis(dataset[i], centers[j]);
			if(dist<mindist):
				mindist = dist;			
				lastassign = j;
				assignment[i] = j;
		if(lastassign != original):
			flag = True;	
	return flag;
				 
				
def kmeans(dataset, k):
	if(len(dataset)<k):
		return [0];
	assign = True;
	print("start to select randomly k centers")
	centers = randomcenter(dataset, k);
	assignment = [];
	print("start to initalize clustering...")
	for i in range(len(dataset)):
		assignment.append(0);
	assign = reallocate(assignment, k, dataset, centers);
	print("initalizatino is completed!");
	while(assign):
		assign = False;
		print("renewing centers times...")
		renewcenters(centers, assignment, k, dataset);
		print("reassign dots...")
		assign = reallocate(assignment, k, dataset, centers);
	return assignment;

if __name__ == "__main__":
	dataset = [];
	a = [];
	a.append([1,2,3,4,8]);
	a.append([3,4,2,0,2]);
	dataset.append(a)
	b = [];
	b.append([2,3,2,7,1]);
	b.append([3,4,2,4,2]);
	dataset.append(b)
	assign = kmeans(dataset, 2);
	
