#!/bin/bash

i=0;
cat newdata  |  while read LINE
do
	echo "net mem io cpu" >vm$i.txt
	echo $LINE >> vm$i.txt
	((i=$i+1))
done
