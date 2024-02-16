#!/bin/bash
for ((x=0;x<$2;x++)) do sbatch -G $1 -p research script.sh $3 ; done 

