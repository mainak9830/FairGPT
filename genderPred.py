# Team Members - Aishik Deb, Mainak Adak, Hao Lin, Yuqing Wang
# Description - Python file to predict the gender of the names extracted from the Wikipedia dataset using the custom trained model
# Data FrameWork Used -  PyTorch
# System Used - Google Cloud VM Instance using Ubuntu

# References
# https://github.com/roscibely/gender-classification/blob/main/code/code-v1/Predicting_Gender_of_Brazilian_Names_Using_Deep_Learning.ipynb


import pandas as pd                       
import numpy as np
from nameModel import LSTMModel
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score
import json

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
file = 'gender_output8.csv'
df = pd.read_csv('NameDatasets/'+file) 




#y = df['Gender'].astype("category").cat.codes.values    # y labels into numbers 0 is F and 1 is M
names = df['name'].apply(lambda x: x.lower())             # input names



maxlen = 20                                               # max lenght of a name

f = open('encoding.json')
char_index = json.load(f)
f.close()
len_vocab = len(char_index)

def set_flag(i):
    aux = np.zeros(len_vocab)
    aux[i] = 1
    
    return list(aux)

# Truncate names and create the matrix
def prepare_encod_names(X):
    
    trunc_name = str(X)[0:maxlen] # consider only the first 20 characters
  
    tmp = [set_flag(char_index[j]) for j in str(trunc_name)]
    for k in range(0,maxlen - len(str(trunc_name))):
        tmp.append(set_flag(char_index["END"]))
        
        
    return tmp

# x = prepare_encod_names(names.values)   # Now the names are encod as a vector of numbers


print("x_train.shape")


batch_size = 1
learning_rate = 0.01
num_epochs = 10



model = LSTMModel(len_vocab, 64, 1).double()


model = model.to(device)
model.load_state_dict(torch.load('bidirecmodel.pt'))

model.eval()
correct = 0
total = 0
gpred = []
with torch.no_grad():  # disable gradient calculation for evaluation
    
    for name in names:
        # print(name)
        inputs = prepare_encod_names(name)
        inputs = torch.tensor(inputs)
        inputs = inputs.to(device)
        inputs = inputs.unsqueeze(0)
        # print(inputs.shape)
        outputs = model(inputs).squeeze(1)
        # print(outputs.shape)
        preds = (outputs > 0.5).float()
        if(preds == 0):
            gpred.append("female")
        else:
            gpred.append("male")

        

df['gender'] = np.array(gpred)
df.to_csv('outputs/'+file, index=False)
        


