#!/usr/bin/env python
import os
import sys
sys.path.append("../")
import subprocess as subp
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
CPU_D = "cpu.scale"
IO_D = "io.scale"
MEM_D = "mem.scale"
NET_D = "net.scale"
IDLE_D = "idle.scale"

modelnames = [CPU_IO, CPU_MEM,CPU_NET,CPU_IDL ,IO_MEM ,IO_NET ,IO_IDL,MEM_NET,MEM_IDL,NET_IDL ];
filenames_d = [CPU_D, IO_D, MEM_D, NET_D, IDLE_D];
labeltable = {10:"cpu", 11:"io", 12:"mem", 13:"net", 14:"idle"}

scaledict = [ [258, 321-157, 98687-20, 81], 
	      [258, 1592-168, 627, 64],
              [12262, 321-157, 652-20, 84],
	      [258, 321-159, 572, 85],
	      [219, 1592 - 157, 98687 - 6, 20],
              [12262, 174 - 157, 875 - 571, 10],
	      [219, 174-157, 98687, 11.5],
	      [12262, 1592-157, 652-6, 25-2],
	      [100, 1592-159, 633, 25],
	      [12262, 167-157, 652, 2.8] ]

mindict = [ [0, 157, 20, 5], 
	      [0, 168, 6, 22],
              [0, 157, 20, 2],
	      [0, 159, 0, 0.5],
	      [0, 157, 6, 5],
              [0, 157, 571, 2.25],
	      [0, 157, 1, 0.5],
	      [0, 157, 6, 2],
	      [0, 159, 0.5, 0.5],
	      [0, 157, 0.5, 0.5] ]



class Svm:
	def __init__(self):
		self.vector_dict = {"cpu":0,  "io":0,  "mem":0,  "net":0, "idle":0};	
		self.models = None;
		self.accuracy = 0.8;
		print("start to train original classifier...")
		self.train();

	def train(self):
		if(self.models == None):
			self.models = [];
			if(self.isTrainedModelsExist()):
				for i in range(len(modelnames)):
					self.models.append(svmu.svm_load_model("../models/"+modelnames[i]));
			else:
				#if(self.isTrainDataExist()):
				for i in range(len(modelnames)):
					name = modelnames[i].split(".")[0].split("_");
					dataname1, dataname2 = name[0], name[1];
					dataname = dataname1+"_"+dataname2;
					#self.combinefiles(dataname1+".scale", dataname2+".scale")
					y, x = svmu.svm_read_problem("../data/"+dataname);
					prob = svmu.svm_problem(y, x);
					out = subp.Popen(["python", "../modules/libsvm/tools/grid.py", "../data/"+dataname], stdout=subp.PIPE)						
					paras = out.stdout.read().split(")")[-1];
					c = paras.split(" ")[0];
					g = paras.split(" ")[1];
					acc = paras.split(" ")[2];
 					#para = svm.svm_parameter('-t 2 '+'-c '+c+" -g "+g);
 					para = svm.svm_parameter("-s 0 -c "+c+" -t 2 -g "+g+" -r 1 -d 3");
					m = svmu.svm_train(prob, para);
					svmu.svm_save_model("../models/"+dataname+".model", m)
					self.models.append(m);
						
				#else:
				#	print("svm class is initialized without classifiers");
					
	
	def classify(self, y, x):
		self.vector_dict = {"cpu":0,  "io":0,  "mem":0,  "net":0, "idle":0};	
		x0 = [];x0.append({1:0, 2:0, 3:0, 4:0});
		if self.models == None:
			self.train();
		for i in range(len(self.models)):
	#		print(str(x[0][1])+"----\n")
	#		print(str(x[0][2])+"----\n")
	#		print(str(x[0][3])+"----\n")
	#		print(str(x[0][4])+"----\n")
		#	x0[0][1] = (float)(x[0][1] - mindict[i][0])/scaledict[i][0]
		#	x0[0][2] = (float)(x[0][2] - mindict[i][1])/scaledict[i][1]					
		#	x0[0][3] = (float)(x[0][3] - mindict[i][2])/scaledict[i][2]					
		#	x0[0][4] = (float)(x[0][4] - mindict[i][3])/scaledict[i][3]					
#			print("********************\n")
			print(str(x0[0][1])+"----\n")
			print(str(x0[0][2])+"----\n")
			print(str(x0[0][3])+"----\n")
			print(str(x0[0][4])+"----\n")
#			print(str(scaledict[i][0])+"----\n")
#			print(str(scaledict[i][1])+"----\n")
#			print(str(scaledict[i][2])+"----\n")
#			print(str(scaledict[i][3])+"----\n")
			x0[0][1] = (float)(x[0][1])
			x0[0][2] = (float)(x[0][2])
			x0[0][3] = (float)(x[0][3])
			x0[0][4] = (float)(x[0][4])

			p_label, p_acc, p_pval = svmu.svm_predict(y, x0, self.models[i]);
			self.vector_dict[labeltable[int(p_label[0])]]+=1;
	
	def isfitperformance(self, model, dataname, label):
		y, x = svmu.svm_read_problem("../data/"+dataname+".test");
		p_label, p_acc, p_pval = svmu.svm_predict(y, x, m);
		if(p_acc>self.accuracy):
			return true;
		else:
			return false;
				
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
		wr = open("../data/"+file1+"_"+file2, "w");
		print("combining "+file1+" "+file2+"'s data");
		f1 = open("../data/"+file1, "r");
		f2 = open("../data/"+file2, "r");
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
		
	print(pp.doOnePreProcess([10],[{1:103, 2:168, 3:87, 4:77}]))
	print(pp.doOnePreProcess([10],[{1:15, 2:160, 3:17000, 4:5}]))
	print(pp.doOnePreProcess([10],[{1:12000, 2:160, 3:600, 4:2}]))

	print("---test completed!---")
	


