#!/bin/bash
for file in slurm*.out; do grep 'Test accuracy' "$file" | awk '{printf "%s ", $NF}'; echo; done > $1

