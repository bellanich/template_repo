import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import sys
import torch
import torch.nn as nn
from datetime import datetime
from model import SimpleRegression

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


# TODO list:
#   (1) Define variables that need to be parameter tuned. Use an args parser.
#       Vars include: seed, lr, dropout_rate, epoch_num
#   (2) Create stopping condition. If loss doesn't improve for 5 conseuctives batches, then break.
#   (3) Update model_path name and the results_message based on hyper-parameters used.
#   (4) Make a small SEPARATE script that takes results.txt and outputs the best set of found hyperparameters.
#   (5) Generate a pl_torch version of this script.

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
model = SimpleRegression(input_dim=input_size).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()
torch.manual_seed(42)  # Setting the seed.

# Training Loop.
print('Beginning training.')
for epoch in range(2):  # loop over the dataset multiple times
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

        # Printing loss
        running_loss += loss.item()
        if i % 20 == 19:    # Print every 2000 mini-batches.
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 20))
            running_loss = 0.0

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
loss_message = f"""{timestamp} --- Average test loss = {loss:0.8f} --- Unnormalized test loss = {unnormalized_loss:0.8f}\n"""
results_log = open(log_filename, 'a+')
results_log.write(loss_message)
results_log.close()
print('Done.')








