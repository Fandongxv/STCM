import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import sys
sys.path.append('e:\\Code\\STCM\\Data imputation\\SCIN\\module')
#print(sys.path)
from SCINet import SCINet, get_variable
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def mask_data(data, missing_rate):
    num_samples, seq_len = data.shape
    missing_mask = np.random.choice([0, 1], size=(num_samples, seq_len), p=[1 - missing_rate, missing_rate])
    data_missing = data.copy()
    data_missing = np.where(missing_mask, 0, data_missing)
    print("missing_mask:",missing_mask[:])
    return data[:, :], data_missing[:, :], missing_mask

def pad_data(data):
    num_samples, seq_len = data.shape
    new_seq_len = seq_len + 1
    padded_data = np.zeros((num_samples, new_seq_len))
    padded_data[:, :seq_len] = data
    return padded_data

#dataset
path = './data/creditcard.csv'
df = pd.read_csv(path, sep=",", index_col=None)

# Standardize
df['Amount'] = StandardScaler().fit_transform(df['Amount'].values.reshape(-1, 1))
df['Time'] = StandardScaler().fit_transform(df['Time'].values.reshape(-1, 1))
anomalies = df[df["Class"] == 1]
anomalies.to_csv("./data/creditcard_data/anomalies.csv", index=False)
normal = df[df["Class"] == 0]

print("Anomalies shape:", anomalies.shape)
# test
#print(anomalies[400:])
data = anomalies[:].values
data2 = anomalies[:].values
print(data2.shape)

seq_len = 32
missing_rate = 0.1
original_data_padded = pad_data(data)
train_x, train_x_missing, train_missing_mask = mask_data(original_data_padded, missing_rate)
train_x = train_x.reshape(-1, seq_len, 1)
train_x_missing = train_x_missing.reshape(-1, seq_len, 1)
train_x = torch.tensor(train_x, dtype=torch.float32)
train_x_missing = torch.tensor(train_x_missing, dtype=torch.float32)

train_data = TensorDataset(train_x, train_x_missing)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)


input_dim = 1
model = SCINet(output_len=seq_len, input_len=seq_len, input_dim=input_dim)
if torch.cuda.is_available():
    model.cuda()


loss_function = torch.nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)



num_epochs = 100
for epoch in range(num_epochs):
    for x_batch, x_missing_batch in train_loader:
        x_batch = get_variable(x_batch)
        x_missing_batch = get_variable(x_missing_batch)

        
        output = model(x_missing_batch)

       
        loss = loss_function(output, x_batch)

        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    
    print(f'Epoch: {epoch+1}, Loss: {loss.item()}')


torch.save(model.state_dict(), "./train_result/my_model.pth")
torch.save(optimizer.state_dict(), "./train_result/my_optimizer.pth")

#model.load_state_dict(torch.load("./train_result/my_model_0.0780.pth"),torch.load("./train_result/my_optimizer_0.0780.pth"))

with torch.no_grad():
    original_data_padded_2 = pad_data(data2)
    test_x, test_x_missing, test_missing_mask = mask_data(original_data_padded_2, missing_rate)
    test_x = test_x.reshape(-1, seq_len, 1)
    test_x_missing = test_x
    test_x_missing = test_x_missing.reshape(-1, seq_len, 1)
    test_x = torch.tensor(test_x, dtype=torch.float32)
    test_x_missing = torch.tensor(test_x_missing, dtype=torch.float32)

    test_x_missing = get_variable(test_x_missing)
    predictions = model(test_x_missing)
    predictions = predictions.cpu().numpy()


true_values = test_x.cpu().numpy().reshape(-1)
missing_mask = test_missing_mask.reshape(-1)
predictions = predictions.reshape(predictions.shape[0], -1)


reconstructed_data = np.where(missing_mask, predictions.reshape(-1), true_values)

#print("true_values",true_values[1:50])
#print("reconstructed_data",reconstructed_data[1:50])

# import matplotlib.pyplot as plt
# plt.plot(true_values, label='True Values')
# plt.plot(reconstructed_data, label='Reconstructed Values')
# plt.legend()
# plt.show()

print("true_values:",true_values)
print("reconstructed_data:",reconstructed_data)
mse = np.mean((true_values - reconstructed_data)**2)
print('MSE: {:.4f}'.format(mse))


# reconstructed_data-->DataFrame
reconstructed_df = pd.DataFrame(reconstructed_data.reshape(-1,32))

reconstructed_df=reconstructed_df.drop(columns=31,axis=1)
#print(reconstructed_df)

reconstructed_df.columns=anomalies.columns
#print("data imputation:",reconstructed_df)


#df_anomalies=pd.concat([anomalies1[400:],reconstructed_df])

print(reconstructed_df)


reconstructed_df.to_csv("./data/creditcard_data/reconstructed_creditcard.csv", index=False)


#reconstructed_df.to_csv("./data/creditcard_data/reconstructed_creditcard.csv", mode='a',header=False,index=False)