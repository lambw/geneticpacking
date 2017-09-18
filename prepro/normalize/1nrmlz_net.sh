#!/bin/bash

max=`awk 'NR==1{max=$3;next}{max=max>$3?max:$3}END{print max}' orig`
min=`awk 'NR==1{min=$3;next}{min=min<$3?min:$3}END{print min}' orig`
#echo $max
#echo $min
range=`bc <<<$max-$min`
#echo $range
cat orig |  while read l1 l11 net l12 mem l13 io l14 cpu
do
	delta=`bc <<<$net-$min`
	newnet=`awk 'BEGIN{printf "%.2f\n",'$delta'/'$range'}'`
	echo "$l1 $l11 $newnet $l12 $mem $l13 $io $l14 $cpu" >>newnet

done
