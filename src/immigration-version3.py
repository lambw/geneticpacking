);
		
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

