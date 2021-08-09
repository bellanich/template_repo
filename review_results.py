import os
import sys
import numpy as np
import textwrap
"""
TODO: 
	(1) Create a file called 'results_analysis.txt' --> Make creation dynamic. 
		Only created if it doesn't already exist.
	(2) Extract item from list with lowest error.
	(3) Create string of item's timestamp AND its parameters.
	(4) Write into file.
"""

def reindent(s, numSpaces=0):
    s = s.split('\n')
    s = [(numSpaces * ' ') + line.lstrip() for line in s]
    s = ''.join(s)
    return s

current_dir = os.getcwd()
output_filename = 'results_analysis.txt'

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
print(hyperparameter_vals)
hyperparameter_vals = hyperparameter_vals.split(',')
hyperparameter_vals = [ 'TAB' + param_val + 'LINEBREAK' for param_val in hyperparameter_vals]
formatted_hyperparams = ''.join(hyperparameter_vals)
print(formatted_hyperparams)

analysis_message = f"""Out of the {len(results)} trained models we have, 
					the model named '{model_name}' performed the best. The optimal 
					hyperparameters were found to be: LINEBREAK LINEBREAK

					{formatted_hyperparams} LINEBREAK LINEBREAK

					The average loss across all {len(results)} runs was {avg_loss} 
					and the standard deviation was {loss_std}."""
analysis_message = reindent(analysis_message).replace('LINEBREAK', '\n').replace('TAB', '\t')
print(analysis_message)



# One way to analysis_message, but not the nicest.
# analysis_message = f'Out of the {len(results)} trained models we have, ', \
# 					f'model {model_name} performed the best. The optimal hyperparameters ', \
# 					f'were found to be {hyperparameter_vals}.\n', \
# 					f'The average loss across all {len(results)} runs was {avg_loss} ', \
# 					f'and the standard deviation was {loss_std}.'
# analysis_message = ''.join(analysis_message)
# Cleaning up the message so that it won't have any weird indents.

	