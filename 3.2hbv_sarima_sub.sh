#!/bin/bash
#SBATCH --job-name=hbv_sarima            # Job name
#SBATCH --output=hbv_sarima%j.log       # Output file name (%j expands to jobID)
#SBATCH --error=hbv_sarima%j.log        # Error file name (%j expands to jobID)
#SBATCH --time=100:00:00                 # Time limit (HH:MM:SS)
#SBATCH --nodes=1                       #5 Number of nodes
#SBATCH --ntasks=80                    #394 Number of tasks (one for each job), if you don't know numner of tasks beforehand there are ways to make this input dynamic as well
#SBATCH --cpus-per-task=1               # Number of CPU cores per task
#SBATCH --mem=2G                        # Memory per CPU core (adjust as needed)
#SBATCH --exclusive                     # Exclusive node allocation

# Load necessary modules
# All modules are loaded inside the virtual environment so don't need to load here (check: pip list modules when virtual environment is loaded) 
module load python/3.11.5
# Activate your virtual environment if needed
source ~/pyenv-hmodel/bin/activate

# Run your Python script with mpi
mpirun python3 3.2hbv_sarima_run.py
