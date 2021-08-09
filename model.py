import torch
import torch.nn as nn


class SimpleRegressionModel(nn.Module):

    def __init__(self, input_dim, dropout_rate=0.3):

        super().__init__()
        # Uses an output dimension of one, since this is a regression task
        # and not a classification task.
        self.regression_model = nn.Sequential(nn.Linear(input_dim, 64),
								   nn.Dropout(dropout_rate),
								   nn.LeakyReLU(),
								   nn.Linear(64, 1))


    def forward(self, x):
        # Perform the calculation of the model to determine the prediction
        x = self.regression_model(x)
        return x