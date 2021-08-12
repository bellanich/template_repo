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


def check_model_status(hyperparam_description, results_filename):
	# Check whether or not hyperparam_description already exists in results log.
	results_file = open(results_filename, 'r')
	results = results_file.read()
	model_exists = True if hyperparam_description in results else False
	results_file.close()
	return model_exists

# Get file path of training script.
# Switch that either execute pl_torch or torch version of training script.
use_pl = False  
current_dir = os.getcwd()
training_script = os.path.join('v02_pl', 'train_v02_pl.py') if use_pl  else os.path.join('v01_torch','train_v01_torch.py') 
training_script = os.path.join(current_dir, training_script)

# Check to see if a results.txt already exists.
# Getting a necessary sub_dir from the trainng_script path.
sub_dir = os.path.split(training_script)[-1].replace('.', '_').split('_')[1:3] 
sub_dir = '_'.join(sub_dir)
results_logpath = os.path.join(current_dir, sub_dir, 'results', 'results.txt')
results_file_exists = True if os.path.isfile(results_logpath) else False

# Hyperparameter that we'll conduct a grid search over.
seeds = [42, 13, 123]
dropout_rates = [0.2, 0.3]  
learning_rates = [0.01, 0.001, 0.0001]
# Other model training parameters that you can change.
val_freq = 20
epoch_num = 200

# Loop through all posssible combinations.
for seed in seeds:
	for dropout_rate in dropout_rates:
		for lr in learning_rates:

			# Generating hyperparam_tag. It MUST EXACTLY match the way hyperparams are recorded 
			# in 'results.txt'.
			hyperparam_tag = f'seed={seed}, dropout_rate={dropout_rate}, learn_rate={lr}, ', \
							f'validation_freq={val_freq}, epoch_num={epoch_num}'
			hyperparam_tag = "".join(hyperparam_tag)

			# Use hyperparam_tag to check if the model already exists.
			if results_file_exists:
				model_exists = check_model_status(hyperparam_tag, results_logpath)
				if model_exists:
					print(f'A model with the following hyperparameter values already exists:\n {hyperparam_tag}\n')
					break

			# Otherwise, exeute training loop.
			# Execute script. Note: '-n' suppresses print statements made in the training loop.
			subprocess.run(["python", training_script, f"-s {seed}", f"-d {dropout_rate}", f"-l {lr}", \
							f"-v {val_freq}", f"-e {epoch_num}", "-n"])
