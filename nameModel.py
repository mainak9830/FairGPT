# Team Members - Aishik Deb, Mainak Adak, Hao Lin, Yuqing Wang
# Description - Python file to define the PyTorch LSTM Model
# Data FrameWork Used -  PyTorch
# System Used - Google Cloud VM Instance using Ubuntu


import torch
import torch.nn as nn
"""
    ChatGPT
"""
class BidirectionalLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(BidirectionalLSTM, self).__init__()
        self.forward_lstm = nn.LSTM(input_dim, hidden_dim, bidirectional=False, batch_first=True)
        self.backward_lstm = nn.LSTM(input_dim, hidden_dim, bidirectional=False, batch_first=True)

    def forward(self, x):
        # Forward pass through forward LSTM
        out_forward, _ = self.forward_lstm(x)

        # Reverse the input sequence
        x_reverse = torch.flip(x, dims=[1])

        # Forward pass through backward LSTM
        out_backward, _ = self.backward_lstm(x_reverse)

        # Reverse the output sequence from backward LSTM
        out_backward = torch.flip(out_backward, dims=[1])

        # Concatenate the outputs from both LSTMs along the feature dimension
        out = torch.cat((out_forward, out_backward), dim=2)
        
        return out
"""
    ChatGPT
"""

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.lstm1 = BidirectionalLSTM(input_size, hidden_size)
        self.dropout = nn.Dropout(0.2)
        self.lstm2 = nn.LSTM(hidden_size*2, hidden_size, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size*2, output_size)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        out = self.lstm1(x)
        #print(out.shape)
        out = self.dropout(out)
        out, _ = self.lstm2(out)
        out = self.dropout(out)
        out = out[:, -1, :]
        #print(out.shape)
        out = self.fc(out)
        out = self.sigmoid(out)

        return out


data = torch.randn((45, 20, 28))
print(data.shape)
model = LSTMModel(28, 64, 1)
output = model(data)
print(output.shape)