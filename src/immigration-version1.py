#-*-coding:utf-8-*-
#!/usr/bin/env python
import sys
import math
import copy
sys.path.append("../")
import os
import random
import numpy as np
import argparse
import util.svmpreprocess as svm
import util.Kmeans as kmeans
from operator import attrgetter

PHY_MEM = 140000;#number is temporary; and resources occupied is measured directly by simple minus
CPU_CAPs = 2000;
IO_CAPs = 20000; 
NET_CAPs =20000;
INITAL_MAX_POPS = 100;
MUTATION_RATE = 0.01;
MAX_POPS = 100;

MEASEQ = [PHY_MEM, CPU_CAPs, IO_CAPs, NET_CAPs];

class Utiltools:
	#shuffl one sequence one time;		
	@classmethod
	def do_one_rpermutation(cls, seq):
		pos1 = int(random.random()*len(seq));
		pos2 = int(random.random()*len(seq));
		t = seq[pos1];
		seq[pos1] = seq[pos2];
		seq[pos2] = t;

	#check if a virtual machine can be contained in a physical machine
	@classmethod	
	def ifCanContain(cls, vmachine, pmachine, time):
		matched = True;
		if(float(vmachine.res_request[time]["PHY_MEM"])>float(pmachine.phy_mem)):
			matched = False;
		if(float(vmachine.res_request[time]["CPU_CAPs"])>float(pmachine.cpu_caps)):
			matched = False;
		if(float(vmachine.res_request[time]["IO_CAPs"])>float(pmachine.io_caps)):
			matched = False;
		if(float(vmachine.res_request[time]["NET_CAPs"])>float(pmachine.net_caps)):
			matched = False;
		return matched;

	@classmethod
	def generate(cls, items, bins):
		solution_vector = [];
		for i in range(len(items)):
			solution_vector.append(int(random.random()*len(bins)));
		solution = Solution(solution_vector, 0);
		solution.update(0);
		return solution;

	#generate one first heuristic algorithm solution
	@classmethod
	def firstfit(cls, itemseq, binseq, time):
		for x in xrange(len(binseq)):
			binseq[x].restore();
		solution_vector = [];
		for i in range(len(itemseq)):
			solution_vector.append(0);
		for i in range(len(itemseq)):
			for j in range(len(binseq)):
				if(cls.ifCanContain(itemseq[i], binseq[j], time)):
					solution_vector[i] = j; 
					net = -float(itemseq[i].res_request[time]["NET_CAPs"])
					cpu = -float(itemseq[i].res_request[time]["CPU_CAPs"])
					io = -float(itemseq[i].res_request[time]["IO_CAPs"])
					mem = -float(itemseq[i].res_request[time]["PHY_MEM"])
					binseq[j].update(io, mem, cpu, net);
					break;
		solution = Solution(solution_vector,0);
		solution.update(0);
		return solution;

	#generate one best heuristic algorithm solution
	@classmethod
	def bestfit(cls, itemseq, binseq, time):
		sorted(binseq, key = attrgetter("phy_mem", "io_caps", "cpu_caps", "net_caps"));
		return cls.firstfit(itemseq, binseq, time);

	@classmethod
	def generateRadomTestData(cls, vmn):
		pm = open("../data/pm.txt", "w")
		for i in range(len(MEASEQ)):
			pm.write(str(MEASEQ[i]*((random.random()+1)/2))+"\n")	
		pm.close();
		for i in range(int(vmn)):
			vm = open("../data/vm"+str(i)+".txt", "w");
			vm.write("mem cpu io net\n");
			vm.write(str(MEASEQ[0]*random.random()/2)+" "+str(MEASEQ[1]*random.random()/2)+" "+str(MEASEQ[2]*random.random()/2)+" "+str(MEASEQ[3]*random.random()/2));

class DataReader:
	def __init__(self):
		self.dict_resource = None;
	@classmethod
	def readForOneVirtual(cls, fileName, no):
		count = 0;
		res_request = [];	
		f = open(fileName, "r");
		count = 0;
		for line in f.readlines():
			sample = dict();
			if(count>0):
				sample["PHY_MEM"] = line.split()[0];	
				sample["CPU_CAPs"] = line.split()[1];	
				sample["IO_CAPs"] = line.split()[2];	
				sample["NET_CAPs"] = line.split()[3];
				res_request.append(sample);	
			count+=1;
		virtualMachine = VirtualMachine(res_request, no);
		return virtualMachine;

	@classmethod			
	def readPhysicalParameter(cls, fileName):
		f = open(fileName, "r");
		dc = dict();
		data = f.readlines()
		dc["PHY_MEM"] =  data[0];
		dc["CPU_CAPs"] =  data[1];
		dc["IO_CAPs"] =  data[2];
		dc["NET_CAPs"] =  data[3];
		f.close();
		return dc;

class BinItemManager:
	bins = [];
	items = [];
	@classmethod
	def initalizeItems(cls, numsOfVm):
		for i in range(int(numsOfVm)):
			BinItemManager.items.append(DataReader.readForOneVirtual("../data/vm"+str(i)+".txt", i))
				
	@classmethod
	def initalizeBins(cls, numsOfBins):
		for i in range(int(numsOfBins)):
			BinItemManager.bins.append(PhysicalMachine(DataReader.readPhysicalParameter("../data/pm.txt"), i))




class Solution:
	def __init__(self, solution_vector, rate_value):
		self.solution_vector = solution_vector;	
		self.rate_value = rate_value; 

	#recalculate the fittness of this solution;
	def update(self, time):
		self.rate_value = self.calrate_value(time);

	#calculate the fitness of this solution;!!! the equation is still not complete now
	def calrate_value(self, time):
		rate_value = 0.0;
		for i in range(len(BinItemManager.bins)):
			phy_mem = 0;
			cpu_caps = 0;
			io_caps = 0;
			net_caps = 0;
			pval = 1;
			for n in range(len(self.solution_vector)):
				if(self.solution_vector[n] == i):
					phy_mem = float(BinItemManager.items[n].res_request[time]["PHY_MEM"]);
					cpu_caps = float(BinItemManager.items[n].res_request[time]["CPU_CAPs"]);
					io_caps = float(BinItemManager.items[n].res_request[time]["IO_CAPs"]);
					net_caps = float(BinItemManager.items[n].res_request[time]["NET_CAPs"])
			pval = pval*float(phy_mem/float(BinItemManager.bins[i].phy_mem))*float(cpu_caps/float(BinItemManager.bins[i].cpu_caps))*float(io_caps/float(BinItemManager.bins[i].io_caps))*float(net_caps/float(BinItemManager.bins[i].net_caps));
			rate_value += pval;	
		return float(rate_value/self.numsOfitemed());
	def numsOfitemed(self):
		t = set();	
		for i in range(len(self.solution_vector)):
			t.add(self.solution_vector[i]);
		return len(t);
		
class VirtualMachine:
	def __init__(self, res_request = None, no = None, class_vector = []): 
		self.class_vect = class_vector;
		self.res_request = res_request;
		self.id = no;
		
class PhysicalMachine:
	def __init__(self, resource_dict = None, no = None):
		self.RESOURCE = resource_dict;
		self.isoccupied = False;
		if self.RESOURCE:
			self.io_caps = float(self.RESOURCE["IO_CAPs"]);
			self.phy_mem = float(self.RESOURCE["PHY_MEM"]);
			self.cpu_caps = float(self.RESOURCE["CPU_CAPs"]);
			self.net_caps = float(self.RESOURCE["NET_CAPs"]);
		self.id = no;
	def update(self, IO_diff, PHY_diff, CPU_diff, NET_diff):
		self.io_caps = self.io_caps + IO_diff;
		self.phy_mem = self.phy_mem + PHY_diff;
		self.cpu_caps = self.cpu_caps + CPU_diff;
		self.net_caps = self.net_caps + NET_diff;

	def restore(self):
		self.io_caps =float(self.RESOURCE["IO_CAPs"]); 
		self.phy_mem =float(self.RESOURCE["PHY_MEM"]); 
		self.cpu_caps =float(self.RESOURCE["CPU_CAPs"]); 
		self.net_caps =float(self.RESOURCE["NET_CAPs"]); 
		

class GeneticAlgorithm:
	def __init__(self, items = None, bins = None):
		self.items = items;
		self.bins = bins;
		self.initalPops = None;		
		self.cur_pops = None;
		self.next_candidates = [];
		self.cur_groups = [];
		self.cur_groups.append([])
		self.cur_groups.append([])
		self.temp_pops = [];

	def ifsolfeasible(self, sol):
		for c in xrange(len(BinItemManager.bins)):
			BinItemManager.bins[c].restore();

		for i in xrange(len(sol.solution_vector)):
			iodiff = -float(BinItemManager.items[i].res_request[0]["IO_CAPs"]);
			memdiff = -float(BinItemManager.items[i].res_request[0]["PHY_MEM"]);
			cpudiff = -float(BinItemManager.items[i].res_request[0]["CPU_CAPs"]);
			netdiff = -float(BinItemManager.items[i].res_request[0]["NET_CAPs"]);
			BinItemManager.bins[sol.solution_vector[i]].update(iodiff, memdiff, cpudiff, netdiff);
			if(BinItemManager.bins[sol.solution_vector[i]].io_caps<0):
				return False;
			if(BinItemManager.bins[sol.solution_vector[i]].io_caps<0):
				return False;
			if(BinItemManager.bins[sol.solution_vector[i]].io_caps<0):
				return False;
			if(BinItemManager.bins[sol.solution_vector[i]].io_caps<0):
				return False;
		return True;

	def solutionfilter(self, osols):
		sols = [];
		for i in xrange(len(osols)):
			if(self.ifsolfeasible(osols[i])):
				sols.append(osols[i]);
			for c in xrange(len(BinItemManager.bins)):
				BinItemManager.bins[c].restore();
		return sols;	

				
	def report(self):
		self.temp_pops = self.solutionfilter(self.temp_pops);
		self.cur_pops = self.solutionfilter(self.cur_pops);
		sorted(self.temp_pops,key = attrgetter("rate_value"));
		print("----------clustering genetic algorithm is ended--------")
		print("-following are these final immigration plan -")
		print("fitnessvalue	detailsofplan")
		for i in xrange(len(self.temp_pops)):
			print("|%s|\t%s|%s\t"%(self.temp_pops[i].rate_value, self.temp_pops[i].numsOfitemed(), self.temp_pops[i].solution_vector))
		
		sorted(self.cur_pops,key = attrgetter("rate_value"));
		print("----------traditional genetic algorithm is ended--------")
		print("-following are these final immigration plan -")
		print("fitnessvalue	detailsofplan")
		for i in xrange(len(self.cur_pops)):
			print("|%s|\t%s|%s\t"%(self.cur_pops[i].rate_value, self.cur_pops[i].numsOfitemed(), self.cur_pops[i].solution_vector))
			#print(self.cur_pops[i].solution_vector);
			
	def traditionalseleccross(self):
		self.next_candidates = [];
		for i in range(MAX_POPS):
			index1, index2 = int(random.random()*len(self.cur_pops)),int(random.random()*len(self.cur_pops)); 
			self.addintopops(self.next_candidates, self.do_crossover(self.cur_pops[index1], self.cur_pops[index2]))
	
	def trad_genetic_process(self, error, iterations, p):
		if not self.initalPops:
			self.initalPop();
		for c in xrange(len(BinItemManager.bins)):
			BinItemManager.bins[c].restore();
		for i in range(int(iterations)):
			print("The pops of this generation is: %d"%(len(self.cur_pops)))
			self.traditionalseleccross();	
			print("start mutation...")
			self.mutation(p);
			print('\033[1;31;40m');
			print("Completed! %s th iteration"%(i))
			print('\033[0m');

	
	def genetic_process(self, error, iterations, p, k):
		if not self.initalPops:
			self.initalPop();
		for c in xrange(len(BinItemManager.bins)):
			BinItemManager.bins[c].restore();
		for i in range(int(iterations)):
			print("The pops of this generation is: %d"%(len(self.cur_pops)))
			print("start k-means...")
			self.k_means_classify(k);
			print("clustering is done, start to select and crossover")
			self.selection_and_crossover(k);	
			print("start mutation...")
			self.mutation(p);
			print('\033[1;31;40m');
			print("Completed! %s th iteration"%(i))
			print('\033[0m');
		self.temp_pops = copy.deepcopy(self.cur_pops);
	
	def assigngroup(self, assignment, k):
		for i in range(len(assignment)):
			self.cur_groups[assignment[i]].append(self.cur_pops[i]);

	def initialgroup(self, k):
		self.cur_groups = [];
		for i in range(k):
			self.cur_groups.append([]);
				
	def k_means_classify(self, k):
		#self.cur_groups = [[],[]];
		lst = [];
		self.initialgroup(k)
		print("preparing pn vectors...")
		for i in range(len(self.cur_pops)):
			sample = [];
			for n in range(len(BinItemManager.bins)):
				v = [0, 0, 0, 0, 0];	
				sample.append(v);	
			for j in range(len(self.cur_pops[i].solution_vector)):
				for m in range(len(v)):
					sample[self.cur_pops[i].solution_vector[j]][m] += BinItemManager.items[j].class_vect[m];	
			lst.append(sample);
		print("data for clustering has been prepared...");
		print("The size of pn vectors is %s"%(len(lst)));
		assignment =  kmeans.kmeans(lst, k);
		print("The size of assignment vector is %s"%(len(assignment)));
		#for i in range(len(assignment)):
		#	if(assignment[i]==0):
		#		self.cur_groups[0].append(self.cur_pops[i]);
		#	if(assignment[i]==1):
		#		self.cur_groups[1].append(self.cur_pops[i]);
		self.assigngroup(assignment, k);
		
	def selection_and_crossover(self, k):
		for e in range(k):
			print("The size of group1 is %s"%(len(self.cur_groups[e])))	
		sorted(self.cur_groups[0],key = attrgetter("rate_value"));
		sorted(self.cur_groups[1],key = attrgetter("rate_value"));
		self.next_candidates = [];
		x = int(math.sqrt(2*MAX_POPS)/k);
		for i in range(k):
			for j in range(i,k):
				x1, x2 = x, x;
				if(x>len(self.cur_groups[i])):
					x1 = len(self.cur_groups[i])
				if(x>len(self.cur_groups[j])):
					x2 = len(self.cur_groups[j])
				for m in range(x1):
					for n in range(x2):
						self.addintopops(self.next_candidates, self.do_crossover(self.cur_groups[i][m], self.cur_groups[j][n]));
				if(x1 == 0):
					for q in range(x2):
						self.addintopops(self.next_candidates, self.cur_groups[j][q]);
				if(x2 == 0):
					for p in range(x1):
						self.addintopops(self.next_candidates, self.cur_groups[i][p]);
		self.initialgroup(k);
						
		#self.next_candidates = [];
		#if((len(self.cur_groups[1])!=0) and (len(self.cur_groups[0])!=0)):
		#for i in range(len(self.cur_groups[0])):
		#		for j in range(len(self.cur_groups[1])):
		#			self.addintopops(self.next_candidates, self.do_crossover(self.cur_groups[0][i], self.cur_groups[1][j]));
		#			if(len(self.next_candidates)>MAX_POPS):
		#				self.groups = [[],[]]
		#				return;
		#else:
		#	if(len(self.cur_groups[0])!=0):
		#		for j in xrange(len(self.cur_groups[0])):
		#			self.addintopops(self.next_candidates, self.cur_groups[0][j]);
		#			if(len(self.next_candidates)>MAX_POPS):
		#				self.groups = [[],[]]
		#				return;
		#	
		#	if(len(self.cur_groups[1])!=0):
		#		for j in xrange(len(self.cur_groups[1])):
		#			self.addintopops(self.next_candidates, self.cur_groups[1][j]);
		#			if(len(self.next_candidates)>MAX_POPS):
		#				self.groups = [[],[]]
		#				return;
		#self.groups = [[],[]]
					
	def do_crossover(self, solution1, solution2):
		solution_vector = [];
		p =float((solution1.rate_value/(solution1.rate_value+solution2.rate_value)))
		for i in range(len(solution1.solution_vector)):
			if(random.random()>p):
				solution_vector.append(solution2.solution_vector[i]);
			else:
				solution_vector.append(solution1.solution_vector[i]);
		s = Solution(solution_vector, 0); 
		s.update(0);
		return s;

			
	#def removeduplicity(seq):
		
	#mutation is simplely implemented by randomly swap two element of one solution;	
	def mutation(self, p):
		self.cur_pops = [];
		for i in range(len(self.next_candidates))[::-1]:
			if(random.random()<p):
				Utiltools.do_one_rpermutation(self.next_candidates[i].solution_vector);
				self.next_candidates[i].update(0);
			self.addintopops(self.cur_pops, self.next_candidates[i])
		self.next_candidate = [];

	def initalPop(self):
		self.initalPops = [];
		for i in range(INITAL_MAX_POPS/2):
			#self.addintopops(self.initalPops, Utiltools.firstfit(self.items, self.bins, 0));
			#self.addintopops(self.initalPops, Utiltools.bestfit(self.items, self.bins, 0));
			#Utiltools.do_one_rpermutation(self.items);	
			self.addintopops(self.initalPops, Utiltools.generate(self.items, self.bins))
		self.cur_pops = self.initalPops;
		for i in range(len(self.cur_pops)):
			print(self.cur_pops[i].solution_vector);
		print("The size of inital pops set is: %s"%(len(self.cur_pops)));

		

	def solutionsame(self, sol1, sol2):
		flag = True;
		for i in range(len(sol1.solution_vector)):
			if(sol1.solution_vector[i] != sol2.solution_vector[i]):
				flag = False;
				break;
		return flag;
	def addintopops(self, popseq, solution):
		ifinsert = True;
		for i in range(len(popseq)):
			if(self.solutionsame(popseq[i], solution)):
				ifinsert = False;
				break;
		if(ifinsert):
			popseq.append(solution);
		return ifinsert;
				
		
class classifier:
	def __init__(self):
		self.datahandle = svm.PreProcess();
		self.datahandle.initalclassifier();

	def classify(self, y, x):
		return 	self.datahandle.doOnePreProcess(y,x);
	
	def initializecvec(self, vmlist):
		for i in range(len(vmlist)):
			y = [10];
			x = [];
			x0 = {};
			x0[0] = float(vmlist[i].res_request[0]["NET_CAPs"]);
			x0[1] = float(vmlist[i].res_request[0]["PHY_MEM"]);
			x0[2] = float(vmlist[i].res_request[0]["IO_CAPs"]);
			x0[3] = float(vmlist[i].res_request[0]["CPU_CAPs"]);
			x.append(x0)
			cdict = self.classify(y, x);
			vmlist[i].class_vect.append(cdict["net"]);
			vmlist[i].class_vect.append(cdict["cpu"]);
			vmlist[i].class_vect.append(cdict["io"]);
			vmlist[i].class_vect.append(cdict["mem"]);
			vmlist[i].class_vect.append(cdict["idle"]);
					
if __name__ == "__main__":
	parser = argparse.ArgumentParser();
	parser.add_argument("--error", help="threshold error of GA")
	parser.add_argument("--iter", help="max iterations of GA")
	parser.add_argument("--vms", help="nms of virtual machines")
	parser.add_argument("--pms", help="nms of physical machines")
	parser.add_argument("--mutp", help="prob of mutation")
	parser.add_argument("--k", help="number of clusters")
	args = parser.parse_args()
	#Utiltools.generateRadomTestData(args.vms);


	clsf = classifier();	
	BinItemManager.initalizeItems(args.vms);
	BinItemManager.initalizeBins(args.pms);
	clsf.initializecvec(BinItemManager.items);
	print("initalization is completed...");
	GA = GeneticAlgorithm(BinItemManager.items, BinItemManager.bins);
	GA.genetic_process(args.error, args.iter, args.mutp, int(args.k));
	GA.trad_genetic_process(args.error, args.iter, args.mutp);
	GA.report();

