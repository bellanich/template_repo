import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import sys
import torch
import torch.nn as nn

# read data
print('Loading data')
dir_path = '/Users/bellanicholson/Desktop/DA Case - Real Estate Analytics'
file_path = os.path.join(dir_path, 
						 'Staten_Island_housing_market_case.xlsx')
save_path = os.path.join(dir_path, 'figures')
houses_price = pd.read_csv("Staten_Island_housing_market_case.csv", low_memory=False)
print('File loaded.')


# Clean data and drop unnecessary attributes.
recent_prices = houses_price[houses_price.year == 2013] # Only consider one year
labels, recent_prices = recent_prices.price, recent_prices.drop(columns=['price', 'Sale_id', 'bbl_id', 'year']).select_dtypes(['number'])
# Normalize data
recent_prices = (recent_prices-recent_prices.mean())/recent_prices.std()
recent_prices = recent_prices.fillna(-1.0)
label_std = labels.std()  # Save for later.
labels = ((labels-labels.mean())/labels.std())
# Convert to Torch Tensor.
labels, recent_prices = torch.from_numpy(labels.to_numpy()).float(), torch.from_numpy(recent_prices.to_numpy()).float()


# Partition dataset: 80% train, 20% test
training_indx = int(np.round(recent_prices.shape[0]*0.8))
train_data, train_labels = recent_prices[:training_indx], labels[:training_indx]
test_data, test_labels = recent_prices[training_indx+1:], labels[training_indx+1:]

# Defining model
input_dim = recent_prices.size()[1]
training_data = torch.utils.data.TensorDataset(train_data, train_labels)
testing_data = torch.utils.data.TensorDataset(test_data, test_labels)
model = nn.Sequential(
		nn.Linear(input_dim, 64),
		nn.Dropout(0.3),
		nn.LeakyReLU(),
		nn.Linear(64, 1))

dataloader = torch.utils.data.DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(testing_data, batch_size=64, shuffle=True)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

# Training Loop.
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







