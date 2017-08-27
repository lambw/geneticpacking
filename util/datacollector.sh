#!/bin/sh


#Feature order: Net Page BI/O Cpu";
touch $1.txt
flag=1
time=0
tag=$2
while [ $flag -eq 1 ]:
do
	sleep 1
	printf "%d " $2 >> ../data/$1.txt
	ifstat -a 1 1 | awk 'BEGIN{row=1}{row++;if(NR==3){printf "%d:%.2f ", 0,$0+$1+$2+$3+$4+$5+$6+$7}}END{}' >> ../data/$1.txt
	vmstat | awk 'BEGIN{row=1}{row++;if(NR==3){printf "%d:%d %d:%d ",1,$6+$7,2,$8+$9}}END{}' >> ../data/$1.txt
	top -n 1 | awk 'BEGIN{row=1}{row++;if(NR==8){printf "%d:%d\n",3,$10}}END{}' >> ../data/$1.txt
	time=$(($time+1));
	if [ $time -eq 300 ];then
		break;
	fi	
done
