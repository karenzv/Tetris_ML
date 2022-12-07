import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

class QLearningNeuralNetwork(nn.Module):
    def _init_(self):
        super()._init_()
        self.linear1 = nn.Linear(18, 64)
        self.act1    = nn.LeakyReLU(0.3)
        self.linear2 = nn.Linear(64, 32)
        self.act2    = nn.LeakyReLU(0.3)
        self.linear3 = nn.Linear(32, 3)
        self.act3    = nn.SoftMax(0.3)

    def forward(self, x):
        x = self.act1(self.linear1(x))
        x = self.act2(self.linear2(x))
        x = self.act3(self.linear3(x))
        return x

    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)

#nn = QLearningNeuralNetwork()
#opt = optim.Adam(nn.parameters(), lr=1e-3)
#loss_fn = nn.CrossEntropyLoss()