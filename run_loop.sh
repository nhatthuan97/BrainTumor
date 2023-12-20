#!/bin/bash
for ((x=0;x<10;x++)) do sbatch -G 1 -p research script.sh $1 ; done 
