import os
import subprocess
import sys


# Get file path of training script.
current_dir = os.getcwd()
training_script = os.path.join(current_dir, 'train.py')

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
