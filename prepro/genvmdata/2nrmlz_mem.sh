#!/bin/bash

max=`awk 'NR==1{max=$5;next}{max=max>$5?max:$5}END{print max}' newnet`
min=`awk 'NR==1{min=$5;next}{min=min<$5?min:$5}END{print min}' newnet`
#echo $max
#echo $min
range=`bc <<<$max-$min`
#echo $range
cat newnet |  while read l1 l11 net l12 mem l13 io l14 cpu
do
	delta=`bc <<<$mem-$min`
	newmem=`awk 'BEGIN{printf "%.2f\n",'$delta'/'$range'}'`
	echo "$l1 $l11 $net $l12 $newmem $l13 $io $l14 $cpu" >>newmem

done
