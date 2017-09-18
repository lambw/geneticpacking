#-*-coding:utf-8-*-
#!/usr/bin/env python
import sys

f = open("./idle", "r");
fs = open("./idle.new", "w");

for line in f.readlines():
	words = line.split();
	word1 = words[0]
	word2 = "1"+":"+words[1].split(":")[1]
	word3 = "2"+":"+words[2].split(":")[1]
	word4 = "3"+":"+words[3].split(":")[1]
	word5 = "4"+":"+words[4].split(":")[1]
	fs.write(word1+" "+word2+" "+word3+" "+word4+" "+word5+"\n");

f.close();
fs.close();
	
