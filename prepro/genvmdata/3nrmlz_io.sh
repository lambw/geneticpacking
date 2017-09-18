#!/bin/bash

max=`awk 'NR==1{max=$7;next}{max=max>$7?max:$7}END{print max}' newmem`
min=`awk 'NR==1{min=$7;next}{min=min<$7?min:$7}END{print min}' newmem`
echo $max
echo $min
range=`bc <<<$max-$min`
echo $range
cat newmem |  while read l1 l11 net l12 mem l13 io l14 cpu
do
	delta=`bc <<<$io-$min`
	newio=`awk 'BEGIN{printf "%.2f\n",'$delta'/'$range'}'`
	echo "$l1 $l11 $net $l12 $mem $l13 $newio $l14 $cpu" >>newio

done
