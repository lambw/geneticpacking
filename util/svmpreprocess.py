#!/usr/bin/env python
import sys
sys.path.append("../")
import os
import modules.libsvm.python.svmutil as svmu
import modules.libsvm.python.svm as svm

CPU_IO = "cpu_io.model"
CPU_MEM = "cpu_mem.model"
CPU_NET = "cpu_net.model"
CPU_IDL = "cpu_idle.model"
IO_MEM = "io_mem.model"
IO_NET = "io_net.model"
IO_IDL = "io_idle.model"
MEM_NET = "mem_net.model"
MEM_IDL = "mem_idle.model" 
NET_IDL = "net_idle.model"
CPU_D = "cpu.txt"
IO_D = "io.txt"
MEM_D = "mem.txt"
NET_D = "net.txt"
IDLE_D = "idle.txt"

modelnames = [CPU_IO, CPU_MEM,CPU_NET,CPU_IDL ,IO_MEM ,IO_NET ,IO_IDL,MEM_NET,MEM_IDL,NET_IDL ];
filenames_d = [CPU_D, IO_D, MEM_D, NET_D, IDLE_D];
labeltable = {10:"cpu", 11:"io", 12:"mem", 13:"net", 14:"idle"}

class Svm:
	def __init__(self):
		self.vector_dict = {"cpu":0,  "io":0,  "mem":0,  "net":0, "idle":0};	
		self.models = None;
		print("start to train original classifier...")
		self.train();

	def train(self):
		if(self.models == None):
			self.models = [];
			if(self.isTrainedModelsExist()):
				for i in range(len(modelnames)):
					self.models.append(svmu.svm_load_model("../models/"+modelnames[i]));
			else:
				if(self.isTrainDataExist()):
					for i in range(len(modelnames)):
						name = modelnames[i].split(".")[0].split("_");
						dataname1, dataname2 = name[0], name[1];
						dataname = dataname1+"_"+dataname2;
						self.combinefiles(dataname1, dataname2)
						y, x = svmu.svm_read_problem("../data/"+dataname+".txt");
						prob = svmu.svm_problem(y, x);
						para = svmu.svm_parameter('-s 0 -t 2 ')
						m = svmu.svm_train(prob, para);
						svmu.svm_save_model("../models/"+dataname+".model", m)
						self.models.append(m);
						
				else:
					print("svm class is initialized without classifiers");
					
	
	def classify(self, y, x):
		if self.models == None:
			self.train();
		for i in range(len(self.models)):
			p_label, p_acc, p_pval = svmu.svm_predict(y, x, self.models[i]);
			self.vector_dict[labeltable[p_label[0]]]+=1;
				

	def isTrainedModelsExist(self):
		i = 0;
		while (os.path.isfile("../models/"+str(modelnames[i]))):
			i += 1;
			if(i>9):
				break;
		if(i>9):
			return True;
		else:
			return False;
	

	def isTrainDataExist(self):
		i = 0;
		while (os.path.isfile("../data/"+filenames_d[i])):
			i += 1;
			if(i>4):
				break;
		if(i>4):
			return True;	
		else:
			return False;	
	def combinefiles(self, file1, file2):
		wr = open("../data/"+file1+"_"+file2+".txt", "w");
		print("combining "+file1+" "+file2+"'s data");
		f1 = open("../data/"+file1+".txt", "r");
		f2 = open("../data/"+file2+".txt", "r");
		for line in f1:
			wr.write(line);
		for line in f2:
			wr.write(line);


class PreProcess:
	def __init___(self):
		print("initalizing svm classifiers...")

	def initalclassifier(self):
		self.svmclassifier = Svm();
		
	def doOnePreProcess(self, y, x):
		 self.svmclassifier.classify(y, x);
		 return self.svmclassifier.vector_dict;



if __name__=='__main__':
	print("---this is a test for training svm classifiers---")
	pp = PreProcess();
	pp.initalclassifier();
	print("---train test end! classify test starts---")
	print(pp.doOnePreProcess([10],[{0:0.1, 1:0.2, 2:0.6, 3:-0.4}]));
	print("---test completed!---")
	


