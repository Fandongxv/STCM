import torch
import torch.nn as nn
import torch.nn.functional as F


class DishTS(nn.Module):
    def __init__(self, args):
        super().__init__()
        init = args.dish_init #'standard', 'avg' or 'uniform'
        activate = True
        n_series = args.n_series # number of series
        lookback = args.seq_len # lookback length
        if init == 'standard':
            self.reduce_mlayer = nn.Parameter(torch.rand(n_series, lookback, 2)/lookback)
        elif init == 'avg':
            self.reduce_mlayer = nn.Parameter(torch.ones(n_series, lookback, 2)/lookback)
        elif init == 'uniform':
            self.reduce_mlayer = nn.Parameter(torch.ones(n_series, lookback, 2)/lookback+torch.rand(n_series, lookback, 2)/lookback)
        self.gamma, self.beta = nn.Parameter(torch.ones(n_series)), nn.Parameter(torch.zeros(n_series))
        self.activate = activate

    def forward(self, batch_x, mode='forward', dec_inp=None):
        if mode == 'forward':
            # batch_x: B*L*D || dec_inp: B*?*D (for xxformers)
            self.preget(batch_x)
            batch_x = self.forward_process(batch_x)
            dec_inp = None if dec_inp is None else self.forward_process(dec_inp)
            return batch_x, dec_inp
        elif mode == 'inverse':
            # batch_x: B*H*D (forecasts)
            batch_y = self.inverse_process(batch_x)
            return batch_y

    def preget(self, batch_x):
        x_transpose = batch_x.permute(2,0,1) 
        theta = torch.bmm(x_transpose, self.reduce_mlayer).permute(1,2,0)
        if self.activate:
            theta = F.gelu(theta)
        self.phil, self.phih = theta[:,:1,:], theta[:,1:,:] 
        self.xil = torch.sum(torch.pow(batch_x - self.phil,2), axis=1, keepdim=True) / (batch_x.shape[1]-1)
        self.xih = torch.sum(torch.pow(batch_x - self.phih,2), axis=1, keepdim=True) / (batch_x.shape[1]-1)

    def forward_process(self, batch_input):
        #print(batch_input.shape, self.phil.shape, self.xih.shape)
        temp = (batch_input - self.phil)/torch.sqrt(self.xil + 1e-8)
        rst = temp.mul(self.gamma) + self.beta
        return rst
    
    def inverse_process(self, batch_input):
        return ((batch_input - self.beta) / self.gamma) * torch.sqrt(self.xih + 1e-8) + self.phih



import torch
import torch.nn as nn
import torch.nn.functional as F
# Residual Block
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation_rate):
        super(ResBlock, self).__init__()
        padding = (kernel_size - 1) * dilation_rate // 2

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation_rate)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding, dilation=dilation_rate)
        self.dropout = nn.Dropout(0.3)
        
        if in_channels != out_channels:
            self.residual = nn.Conv1d(in_channels, out_channels, 1)
        else:
            self.residual = None

    def forward(self, x):
        residual = x
        x = F.relu(self.conv1(x))
        x = self.dropout(x)
        x = F.relu(self.conv2(x))
        x = self.dropout(x)

        if self.residual is not None:
            residual = self.residual(residual)
            
        return F.relu(x + residual)

# Combined TCN and DishTS Model
class TCN_DishTS(nn.Module):
    def __init__(self, input_channels, output_classes, dish_ts_module):
        super(TCN_DishTS, self).__init__()
        self.dish_ts_module = dish_ts_module
        self.resblock1 = ResBlock(input_channels, 32, kernel_size=3, dilation_rate=1)
        self.resblock2 = ResBlock(32, 32, kernel_size=3, dilation_rate=2)
        self.resblock3 = ResBlock(32, 32, kernel_size=3, dilation_rate=4)
        self.dropout = nn.Dropout(0.3)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(32 * 61, output_classes)

    def forward(self, x):
        #x, _ = self.dish_ts_module(x, mode='forward')
        x = x.permute(0, 2, 1)
        x, _ = self.dish_ts_module(x, mode='forward')
        x = x.permute(0, 2, 1)
        x = self.resblock1(x)
        x = self.resblock2(x)
        x = self.resblock3(x)
        x = self.dropout(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from collections import namedtuple

# Assuming ResBlock, DishTS, and TCN_DishTS classes have been defined as shown in the previous response.

# Create synthetic data for binary classification
num_samples = 1000
seq_length = 61
num_channels = 1
input_data = torch.randn(num_samples, num_channels, seq_length)
labels = torch.randint(0, 2, (num_samples, 1))

# Define the Args namedtuple
Args = namedtuple('Args', ['dish_init', 'n_series', 'seq_len'])

# Create an instance of Args with the desired parameters
args = Args(dish_init='standard', n_series=1, seq_len=61)

# Create DishTS module with the given args
dish_ts_module = DishTS(args)
# Create DataLoader for the dataset
dataset = TensorDataset(input_data, labels)
data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Create DishTS module and combined model (assuming args is already defined)
dish_ts_module = DishTS(args)
model = TCN_DishTS(input_channels=num_channels, output_classes=1, dish_ts_module=dish_ts_module)

# Set the loss function and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters())

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    running_loss = 0.0
    for batch_input, batch_labels in data_loader:
        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(batch_input)

        # Compute loss
        loss = criterion(outputs, batch_labels.float())

        # Backward pass
        loss.backward()

        # Optimize
        optimizer.step()

        # Accumulate loss
        running_loss += loss.item()

    # Print average loss for the epoch
    print(f"Epoch {epoch + 1}, Loss: {running_loss / len(data_loader)}")



