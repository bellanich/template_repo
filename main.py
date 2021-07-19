import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import sys
import torch
import torch.nn as nn
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
# Split dataset into two lists of [data_partion, labels].
train_dataset, test_dataset = partition_data(data=recent_prices, labels=labels, training_percent=0.8)


# Creating dataset. TODO: fixme....
print("Fix the way that we're initializing the TensorDataset. It's really ugly right now.")
training_data = torch.utils.data.TensorDataset(train_dataset[0], train_dataset[1]) 
testing_data = torch.utils.data.TensorDataset(test_dataset[0], test_dataset[1])
sys.exit(1)

# Defining model
print('Initializing model.')
input_size= recent_prices.size()[1]
model = SimpleRegression(input_dim=input_size)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

# TODO: fix this. And look up what's the nicer way to do this.
dataloader = torch.utils.data.DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(testing_data, batch_size=64, shuffle=True)


# Training Loop.
model.train()
for epoch in range(200):  # loop over the dataset multiple times
    running_loss = 0.0
    for i, data in enumerate(dataloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
      
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs).squeeze()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print loss
        running_loss += loss.item()
        if i % 20 == 19:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 20))
            running_loss = 0.0

torch.save(model.state_dict(), "our_model.tar")

# Test data
# Load state dict from the disk (make sure it is the same name as above)
state_dict = torch.load("our_model.tar")

# Create a new model and load the state
model.load_state_dict(state_dict)
model.eval()


# Print result.
loss = criterion(model(test_data).squeeze(), test_labels)
print("Test loss", loss.item())
print("Unnormalized loss", loss*label_std)







