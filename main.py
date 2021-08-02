import pandas as pd
import numpy as np
import os
import sys

import torch
import torch.nn as nn

import argparse
from datetime import datetime
from model import SimpleRegressionModel

# TODO list:
#   (2) Write bash script to do grid search.
#   (3) Make a small SEPARATE script that takes results.txt and outputs the best set of found hyperparameters.
#   
#   NEXT TASK: Generate a pl_torch version of this script.


def normalize_data(data):
    # We're assuming that data is a pd.Series object.
    norm_data = (data-data.mean())/data.std()
    return norm_data


def partition_data(data, labels, training_percent=0.80):
    # Throw error if dim of data doesn't match that of labels. (Partitions generated will be nonsense otherwise.)
    if data.shape[0] != labels.shape[0]:
        raise ValueError('Dimensions of data and labels do not match.')

    # Calculate where we 'cut' the dataset into training and testing data.
    datapoints_num = data.shape[0]
    training_indx = int(np.round(datapoints_num*training_percent))

    # Manual indexing.
    train_data, train_labels = recent_prices[:training_indx], labels[:training_indx]
    test_data, test_labels = recent_prices[training_indx+1:], labels[training_indx+1:]
    return [train_data, train_labels], [test_data, test_labels]


def make_dataloader(dataset, batch_size=64, shuffling=True):
    # Assumptions: dataset = [data, labels].
    # We take our dataset and convert it into a PyTorch dataloader.
    data, labels = dataset[0], dataset[1]
    tensor_dataset = torch.utils.data.TensorDataset(data, labels) 
    my_dataloader = torch.utils.data.DataLoader(tensor_dataset, batch_size=batch_size, shuffle=shuffling)
    return my_dataloader


# Defining default hyperparameter values.
parser = argparse.ArgumentParser()
parser.add_argument('-s', '--seed', default=42, type=int)
parser.add_argument('-d', '--dropout_rate', default=0.3, type=float)
parser.add_argument('-lr', '--learn_rate', default=0.001, type=float)
parser.add_argument('-vf', '--validation_freq', default=20, type=int)
parser.add_argument('-e', '--epoch_num', default=200, type=int)
args = parser.parse_args()

# Generating a hyperparameter description to add to results.txt.
# This will be used in review_results.py to determine the optimal hyperparameter values.
hyperparameters_description = ['='.join([arg, str(assigned_val)]) for arg, assigned_val in vars(args).items()]
hyperparameters_description = ', '.join(hyperparameters_description)

# Reading and loading data as pandas Dataframe.
print('Loading data')
current_dir = os.getcwd() 
data_path = os.path.join(current_dir,
                        'data',
                        'Staten_Island_housing_market_case.csv')
houses_price = pd.read_csv(data_path, low_memory=False)
print('File loaded.')

# Create necessary directories if they don't already exist.
directories = ['figures', 'models']
for directory in directories:
    if not os.path.exists(directory):
        os.makedirs(os.path.join(current_dir, directory))

# Create a results log if it doesn't already exist.
log_filename = os.path.join(current_dir, "results.txt")
if not os.path.isfile(log_filename):
    results_log = open(log_filename, "w+")
    results_log.close()

# Define time-stamp. (Will be used in saving files/results.)
now = datetime.now()
timestamp = datetime.timestamp(now)
timestamp = datetime.fromtimestamp(timestamp)

# WARNING: This code was developed locally on a machine without GPU.
# Some debugging with .to(device) will be required when/if GPU is available.
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
torch.manual_seed(args.seed)  

print('Pre-processing data.')
# Filter dataset by certain values. Then, separate it into data and labels.
recent_prices = houses_price[houses_price.year == 2013] 
labels, recent_prices = recent_prices.price, recent_prices.drop(columns=['price', 'Sale_id', 'bbl_id', 'year']).select_dtypes(['number'])
label_std = labels.std()  # Save this for later.
# Normalize data
recent_prices = normalize_data(recent_prices).fillna(-1.0)
labels = normalize_data(labels)
# Convert to Torch tensors.
labels, recent_prices = torch.from_numpy(labels.to_numpy()).float(), torch.from_numpy(recent_prices.to_numpy()).float()
labels.to(device), recent_prices.to(device)
# Split dataset into two lists of [data_partion, labels].
train_dataset, test_dataset = partition_data(data=recent_prices, labels=labels, training_percent=0.8)

train_dataloader = make_dataloader(train_dataset)
test_data, test_labels = test_dataset[0].to(device), test_dataset[1].to(device)

# Defining model.
print('Initializing model.')
input_size= recent_prices.size()[1]
model = SimpleRegressionModel(input_dim=input_size, dropout_rate=args.dropout_rate).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.learn_rate)
criterion = nn.MSELoss()

print('Beginning training.')
loss_log, early_stop = list(), False  # Used for conditional stopping.
# Training loop -> loop over the dataset multiple times.
for epoch in range(args.epoch_num):  
    running_loss = 0.0
    for i, data in enumerate(train_dataloader, 0):
        # Unpack the inputs, where the data is a list of [inputs, labels].
        inputs, labels = data
      
        # Zero the parameter gradients.
        optimizer.zero_grad()

        # Forward + backward + optimize.
        outputs = model(inputs).squeeze()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # Printing loss.
        running_loss += loss.item()
        if i % args.validation_freq == args.validation_freq-1:    # Print every N mini-batches.
            avg_loss = running_loss / args.validation_freq
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, avg_loss))

            if len(loss_log) == 10:  # Keep list as fixed length of 10.

                # Checking to see if it's time to stop training.
                early_stop = (avg_loss >= np.array(loss_log)).all()
                if early_stop:
                    print(f'Terminating training at epoch {epoch}.')
                    break
                # Otherwise, add most recent loss to loss_log and delete first element.
                else:
                    loss_log.pop(0)

            loss_log.append(avg_loss)

            # Reset loss.
            running_loss = 0.0

    # No need to do another epoch.
    if early_stop:
        break

model_path = os.path.join('models', f'SimpleMLP_{timestamp}.tar')
torch.save(model.state_dict(), model_path)

# Test data.
# Load state dict from the disk.
print('Beginning testing.')
state_dict = torch.load(model_path)

# Create a new model and load the state
model.load_state_dict(state_dict)
model.eval()

# Calculate test results.
loss = criterion(model(test_data).squeeze(), test_labels)
unnormalized_loss = loss*label_std

# Save results in a .txt file.
message_items = f'{timestamp}', \
                    f'Average test loss = {loss:0.8f}', \
                    f'Unnormalized test loss = {unnormalized_loss:0.8f} ', \
                    f'Hyperparameters used: {hyperparameters_description} \n'
results_message = ' --- '.join(message_items)
# Only print out the average test loss and its unnormalized value.
print(' --- '.join(results_message.split(' --- ')[1:-1]))
results_log = open(log_filename, 'a+')
results_log.write(results_message)
results_log.close()
print('Done.')








