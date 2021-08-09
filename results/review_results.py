import os
import sys
import numpy as np
import textwrap
"""
	This script is intended to be executed after 'tune_parameters.py'. It reviews all of the results recorded in
	'results.txt', returns some basic statistics about the performance of different models, and then identifies the
	best performing model and its hyperparameters.
"""

def reindent(s, numSpaces=0):
    s = s.split('\n')
    s = [(numSpaces * ' ') + line.lstrip() for line in s]
    s = ''.join(s)
    return s

current_dir = os.getcwd()
output_filename = os.path.join(current_dir, 'results_analysis.txt')

# Read file and load it as a list of results.
with open(os.path.join(current_dir, 'results.txt'), 'r') as results_file:
	results = results_file.readlines()  

# We sort results by their average error from low to high.  
results.sort(key=lambda x: x.split('---')[-2],reverse=False)

# Extract loss scores fand convert into list of numbers.
# This allows us to do some statistics.
loss_scores = [float(result.split('---')[-2].split('=')[-1].strip(' ')) for result in results]
loss_scores = np.asarray(loss_scores)
avg_loss, loss_std = loss_scores.mean(), loss_scores.std()

# Generate a results analysis message.
best_result = results[0].split('---')
model_name, best_loss, hyperparameter_vals  = best_result[0], best_result[-2], best_result[-1].split(':')[-1]
# Reformatting hyperparameter_vals, so it's easier to read in final .txt file.
hyperparameter_vals = hyperparameter_vals.split(',')
hyperparameter_vals = [ 'TAB' + param_val + 'LINEBREAK' for param_val in hyperparameter_vals]
formatted_hyperparams = ''.join(hyperparameter_vals)

# Creating message
analysis_message = f"""Out of the {len(results)} trained models we have, 
					the model named '{model_name}' performed the best. The optimal 
					hyperparameters were found to be: LINEBREAK LINEBREAK

					{formatted_hyperparams} LINEBREAK LINEBREAK

					The average loss across all {len(results)} runs was {avg_loss} 
					and the standard deviation was {loss_std}."""
analysis_message = reindent(analysis_message).replace('LINEBREAK', '\n').replace('TAB', '\t')

# Writing message to file. The way this current written, we'll be overwriting the
# outputfile each time that we execute the script.
results_log = open(output_filename, "w+")
results_log.write(analysis_message)
results_log.close()