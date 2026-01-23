#!/bin/bash
# embedded options to bsub - start with #BSUB
# -- our name ---
#BSUB -J ResistanceTempSweep
# -- choose queue --
#BSUB -q hpc
# -- specify that we need 2GB of memory per core/slot --
#BSUB -R "rusage[mem=1GB]"
# -- Notify me by email when execution begins --
#BSUB -B
# -- Notify me by email when execution ends   --
#BSUB -N
# -- email address -- 
#BSUB -u s234463@student.dtu.dk
# -- Output File --
#BSUB -o Output_ResTempSweep_%J.out
# -- Error File --
#BSUB -e Output_ResTempSweep_%J.err
# -- estimated wall clock time (execution time): hh:mm -- 
#BSUB -W 24:00 
# -- Number of cores requested -- 
#BSUB -n 16
# -- Specify the distribution of the cores: on a single node --
#BSUB -R "span[hosts=1]"
# -- end of LSF options -- 

# Load modules
module load pandas/2.1.3-python-3.11.7

# Activate virtual environment
source /zhome/80/a/205334/dtu/QuantumMechanicalModelling/.venv/bin/activate

# Run the script unbuffered (-u)
python3 -u run_resistance_vs_temp.py
