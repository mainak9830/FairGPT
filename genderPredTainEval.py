# Team Members - Aishik Deb, Mainak Adak, Hao Lin, Yuqing Wang
# Description - Python file to train Pytorch Bi-Directional LSTM model on NYC Baby Names Dataset and evaluate the accuracy of the model
# Data FrameWork Used -  PyTorch
# System Used - Google Cloud VM Instance using Ubuntu

# References
# https://github.com/roscibely/gender-classification/blob/main/code/code-v1/Predicting_Gender_of_Brazilian_Names_Using_Deep_Learning.ipynb

import pandas as pd                       
import numpy as np
from sklearn.model_selection import train_test_split
from nameModel import LSTMModel
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score
import json

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

df = pd.read_csv('NameDatasets/StateNames.csv') 

df = df.head(50000)

y = df['Gender'].astype("category").cat.codes.values    # y labels into numbers 0 is F and 1 is M
names = df['Name'].apply(lambda x: x.lower())             # input names

print("M : " + str(sum(y==1)))
print("F : " + str(sum(y==0)))
# print(names)
length = len(y)
print(length)


maxlen = 20                                               # max lenght of a name
#Define a vocabulary which corresponds to all the unique letters encountered
vocab = set(' '.join([str(i) for i in names]))            # creating a vocab

vocab.add('END')

len_vocab = len(vocab)
char_index = dict((c, i) for i, c in enumerate(vocab)) 
jsonf = json.dumps(char_index)
print(vocab)

f = open("encoding.json","w")

# write json object to file
f.write(jsonf)

# close file
f.close()

# Reading at the time of evaluation to load the saved character encodings
# f = open('encoding.json')
# char_index = json.load(f)
# f.close()
len_vocab = len(char_index)
def set_flag(i):
    aux = np.zeros(len_vocab)
    aux[i] = 1
    
    return list(aux)

# Truncate names and create the matrix
def prepare_encod_names(X):
    vec_names = []
    trunc_name = [str(i)[0:maxlen] for i in X]  # consider only the first 20 characters
    for i in trunc_name:
        tmp = [set_flag(char_index[j]) for j in str(i)]
        for k in range(0,maxlen - len(str(i))):
            tmp.append(set_flag(char_index["END"]))
        vec_names.append(tmp)
        
    return vec_names

x = prepare_encod_names(names.values)   # Now the names are encod as a vector of numbers

# 1.4 Split the data into test and train

# train, val, test set will be 60%, 20%, 20% of the dataset respectively
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=28)
# x_train, x_test, y_train, y_test = x[:length], x[length:], y[:length], y[length:]
print("x_train.shape")

# x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.20, random_state=40)
x_train = np.asarray(x_train)
y_train = np.asarray(y_train)
x_test = np.asarray(x_test)
y_test = np.asarray(y_test)
# x_val = np.asarray(x_val)
# y_val = np.asarray(y_val)

# print(x_train.shape, y_train.shape)
x_train = torch.tensor(x_train)
y_train = torch.tensor(y_train, dtype=torch.double)

x_test = torch.tensor(x_test)
y_test = torch.tensor(y_test, dtype=torch.double)

print(x_train.shape, x_test.shape)
batch_size = 64
learning_rate = 0.01
num_epochs = 10


transform = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize((0.1307,), (0.3081,))
])
model = LSTMModel(len_vocab, 64, 1).double()

train_dataset = torch.utils.data.TensorDataset(x_train, y_train)
test_dataset = torch.utils.data.TensorDataset(x_test, y_test)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Training the model
for epoch in range(num_epochs):
    
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(train_loader, 0):
        # Zero the parameter gradients
        optimizer.zero_grad()
        inputs = inputs.to(device)
        labels = labels.to(device)
        # Forward pass
        outputs = model(inputs).squeeze(1)
        #print(outputs)
        loss = criterion(outputs, labels)

        preds = (outputs > 0.5).float()

        # Compute the accuracy
        correct = (preds == labels).sum().item()
        total = labels.shape[0]
        accuracy = correct / total

        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Print statistics
        running_loss += loss.item()
        if i % 100 == 99:
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 100))
            print("accuracy", accuracy)
            running_loss = 0.0


torch.save(model.state_dict(), 'bidirecmodel.pt')

model = model.to(device)
model.load_state_dict(torch.load('bidirecmodel.pt'))

model.eval()
correct = 0
total = 0
# Evaluating the model on testing data
with torch.no_grad():  # disable gradient calculation for evaluation
    
    for data in test_loader:
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs).squeeze(1)
        preds = (outputs > 0.5).float()
        correct += (preds == labels).sum().item()
        total += labels.shape[0]

accuracy = correct/total

print("Testing accuracy : ", accuracy)