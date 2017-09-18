#!/bin/bash

max=`awk 'NR==1{max=$9;next}{max=max>$9?max:$9}END{print max}' newio`
min=`awk 'NR==1{min=$9;next}{min=min<$9?min:$9}END{print min}' newio`
#echo $max
#echo $min
range=`bc <<<$max-$min`
#echo $range
cat newio |  while read l1 l11 net l12 mem l13 io l14 cpu
do
	delta=`bc <<<$cpu-$min`
	newcpu=`awk 'BEGIN{printf "%.2f\n",'$delta'/'$range'}'`
	echo "$l1 $l11$net $l12$mem $l13$io $l14$newcpu" >>newdata

done
