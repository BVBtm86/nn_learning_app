import torch.nn as nn


# ##### Create Model Architecture
class NeuralNetworkModel(nn.Module):

    def __init__(self, hidden_size, num_classes, no_layers, activation_f, drop_layer):
        super().__init__()
        self.layers = nn.Sequential()
        for i in range(no_layers):
            self.layers.append(nn.Linear(hidden_size[i], hidden_size[i+1]))
            self.layers.append(activation_f[i])
            self.layers.append(drop_layer[0])

        self.layers.append(nn.Linear(hidden_size[i+1], num_classes))

    def forward(self, x):
        return self.layers(x)
