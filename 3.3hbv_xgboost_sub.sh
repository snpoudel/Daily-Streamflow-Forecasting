#!/bin/bash
#SBATCH --job-name=hbv_xgboost            # Job name
#SBATCH --output=hbv_xgboost%j.log       # Output file name (%j expands to jobID)
#SBATCH --error=hbv_xgboost%j.log        # Error file name (%j expands to jobID)
#SBATCH --time=100:00:00                 # Time limit (HH:MM:SS)
#SBATCH --nodes=4                       #5 Number of nodes
#SBATCH --ntasks=51                    #394 Number of tasks (one for each job), if you don't know numner of tasks beforehand there are ways to make this input dynamic as well
#SBATCH --cpus-per-task=6               # Number of CPU cores per task
#SBATCH --mem=8G                        # Memory per CPU core (adjust as needed)
#SBATCH --exclusive                     # Exclusive node allocation

# Load necessary modules
# All modules are loaded inside the virtual environment so don't need to load here (check: pip list modules when virtual environment is loaded) 
module load python/3.11.5
# Activate your virtual environment if needed
source ~/pyenv-hmodel/bin/activate

# Run your Python script with mpi
mpirun python3 3.3hbv_xgboost_run.py
