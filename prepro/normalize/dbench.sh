#!/bin/bash

for ((i=0;i<100;i++))
do
	vmstat 1 13 > vmstat$i &
	dstat 1 13  > ifstat$i &
	free -m -s 1 -c 13 > free$i &
	dbench 10 -t 10
	free -m >>free$i
	sleep 1
done

for ((i=0;i<100;i++))
do
	sed -i '/^procs.*/d' vmstat$i
	sed -i '/^ r.*/d' vmstat$i
	sed -i '1,4d' ifstat$i
	sed -i 's/|/ /g' ifstat$i
	sed -i 's/B/ /g' ifstat$i
	sed -i '/^   .*/d' free$i
	sed -i '/^-.*/d' free$i
	sed -i '/^Swap.*/d' free$i
	sed -i '/^$/d' free$i
done


for ((i=0;i<100;i++))
do
        cpuutil=`awk '{sum=sum+$13}END{printf sum/NR "\n"}' vmstat$i`
        avgbio=`awk '{sum=sum+$9+$10}END{printf sum/NR "\n"}' vmstat$i`
        memmax=`cat free$i|awk '{printf $3"\n"}'|sort|sed -n '$p'`
        avgnio=`awk '{sum=sum+$9+$10}END{printf sum/NR "\n"}' ifstat$i`

        echo "11 0:$avgnio 1:$memmax 2:$avgbio 3:$cpuutil">>dbenchresult

done

