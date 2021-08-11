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
use_pl_torch = True
current_dir = os.getcwd()
training_script = 'train_v01_torch.py' else'train_v02_pltorch.py' if use_pl_torch
print(training_script)
sys.exit(1)
# os.path.join(current_dir,

# Hyperparameter that we'll conduct a grid search over.
seeds = [42, 13]
dropout_rates = [0.2, 0.3]
learning_rates = [0.001, 0.0001]

# Loop through all posssible combinations.
for seed in seeds:
	for dropout_rate in dropout_rates:
		for lr in learning_rates:
			# Execute script. Note: '-n' suppresses print statements made in the training loop.
			subprocess.run(["python", training_script, f"-s {seed}", f"-d {dropout_rate}", f"-l {lr}", "-n"])
