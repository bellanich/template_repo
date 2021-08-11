import os
import subprocess
import sys


"""
	This script lets us do parameter tuning via grid search by executing 'train.py' for various hyperparameter values.
	Each time 'train.py' is executed, model's results on the testing dataset are recorded in the file
	'results/results.txt'. 

	After executing this script, run 'results/review_results.py' to figure out what
	the optimal set of hyperparameters is and what the best performing model is. (Those results will be
	published in 'results_analysis.txt'.)
"""


# Get file path of training script.
# Switch that either execute pl_torch or torch version of training script.
use_pl = False  
current_dir = os.getcwd()
training_script = os.path.join('v02_pl', 'train_v02_pl.py') if use_pl  else os.path.join('v01_torch','train_v01_torch.py') 
training_script = os.path.join(current_dir, training_script)

# Hyperparameter that we'll conduct a grid search over.
seeds = [42, 13, 123]
dropout_rates = [0.2, 0.3]
learning_rates = [0.01, 0.001, 0.0001]

# Loop through all posssible combinations.
for seed in seeds:
	for dropout_rate in dropout_rates:
		for lr in learning_rates:
			# Execute script. Note: '-n' suppresses print statements made in the training loop.
			subprocess.run(["python", training_script, f"-s {seed}", f"-d {dropout_rate}", f"-l {lr}", "-n"])

