import torch.nn as nn


# ##### Create Model Architecture
class NeuralNetworkModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, no_layers, activation_f):
        super().__init__()
        self.no_layers = no_layers
        if self.no_layers == 1:
            self.input_layer = nn.Linear(input_size, hidden_size[0])
            if activation_f[0] == 'relu':
                self.activation_1 = nn.ReLU()
            elif activation_f[0] == 'tanh':
                self.activation_1 = nn.Tanh()
            elif activation_f[0] == 'sigmoid':
                self.activation_1 = nn.Sigmoid()
            else:
                self.activation_1 = nn.Softmax()
            self.linear_1 = nn.Linear(hidden_size[0], num_classes)
        if self.no_layers == 2:
            self.input_layer = nn.Linear(input_size, hidden_size[0])
            if activation_f[0] == 'relu':
                self.activation_1 = nn.ReLU()
            elif activation_f[0] == 'tanh':
                self.activation_1 = nn.Tanh()
            elif activation_f[0] == 'sigmoid':
                self.activation_1 = nn.Sigmoid()
            else:
                self.activation_1 = nn.Softmax()
            self.linear_1 = nn.Linear(hidden_size[0], hidden_size[1])
            if activation_f[1] == 'relu':
                self.activation_2 = nn.ReLU()
            elif activation_f[1] == 'tanh':
                self.activation_2 = nn.Tanh()
            elif activation_f[1] == 'sigmoid':
                self.activation_2 = nn.Sigmoid()
            else:
                self.activation_2 = nn.Softmax()
            self.linear_2 = nn.Linear(hidden_size[1], num_classes)
        if self.no_layers == 3:
            self.input_layer = nn.Linear(input_size, hidden_size[0])
            if activation_f[0] == 'relu':
                self.activation_1 = nn.ReLU()
            elif activation_f[0] == 'tanh':
                self.activation_1 = nn.Tanh()
            elif activation_f[0] == 'sigmoid':
                self.activation_1 = nn.Sigmoid()
            else:
                self.activation_1 = nn.Softmax()
            self.linear_1 = nn.Linear(hidden_size[0], hidden_size[1])
            if activation_f[1] == 'relu':
                self.activation_2 = nn.ReLU()
            elif activation_f[1] == 'tanh':
                self.activation_2 = nn.Tanh()
            elif activation_f[1] == 'sigmoid':
                self.activation_2 = nn.Sigmoid()
            else:
                self.activation_2 = nn.Softmax()
            self.linear_2 = nn.Linear(hidden_size[1], hidden_size[2])
            if activation_f[2] == 'relu':
                self.activation_3 = nn.ReLU()
            elif activation_f[2] == 'tanh':
                self.activation_3 = nn.Tanh()
            elif activation_f[2] == 'sigmoid':
                self.activation_3 = nn.Sigmoid()
            else:
                self.activation_3 = nn.Softmax()
            self.linear_3 = nn.Linear(hidden_size[2], num_classes)
        if self.no_layers == 4:
            self.input_layer = nn.Linear(input_size, hidden_size[0])
            if activation_f[0] == 'relu':
                self.activation_1 = nn.ReLU()
            elif activation_f[0] == 'tanh':
                self.activation_1 = nn.Tanh()
            elif activation_f[0] == 'sigmoid':
                self.activation_1 = nn.Sigmoid()
            else:
                self.activation_1 = nn.Softmax()
            self.linear_1 = nn.Linear(hidden_size[0], hidden_size[1])
            if activation_f[1] == 'relu':
                self.activation_2 = nn.ReLU()
            elif activation_f[1] == 'tanh':
                self.activation_2 = nn.Tanh()
            elif activation_f[1] == 'sigmoid':
                self.activation_2 = nn.Sigmoid()
            else:
                self.activation_2 = nn.Softmax()
            self.linear_2 = nn.Linear(hidden_size[1], hidden_size[2])
            if activation_f[2] == 'relu':
                self.activation_3 = nn.ReLU()
            elif activation_f[2] == 'tanh':
                self.activation_3 = nn.Tanh()
            elif activation_f[2] == 'sigmoid':
                self.activation_3 = nn.Sigmoid()
            else:
                self.activation_3 = nn.Softmax()
            self.linear_3 = nn.Linear(hidden_size[2], hidden_size[3])
            if activation_f[3] == 'relu':
                self.activation_4 = nn.ReLU()
            elif activation_f[3] == 'tanh':
                self.activation_4 = nn.Tanh()
            elif activation_f[3] == 'sigmoid':
                self.activation_4 = nn.Sigmoid()
            else:
                self.activation_4 = nn.Softmax()
            self.linear_4 = nn.Linear(hidden_size[3], num_classes)
        if self.no_layers == 5:
            self.input_layer = nn.Linear(input_size, hidden_size[0])
            if activation_f[0] == 'relu':
                self.activation_1 = nn.ReLU()
            elif activation_f[0] == 'tanh':
                self.activation_1 = nn.Tanh()
            elif activation_f[0] == 'sigmoid':
                self.activation_1 = nn.Sigmoid()
            else:
                self.activation_1 = nn.Softmax()
            self.linear_1 = nn.Linear(hidden_size[0], hidden_size[1])
            if activation_f[1] == 'relu':
                self.activation_2 = nn.ReLU()
            elif activation_f[1] == 'tanh':
                self.activation_2 = nn.Tanh()
            elif activation_f[1] == 'sigmoid':
                self.activation_2 = nn.Sigmoid()
            else:
                self.activation_2 = nn.Softmax()
            self.linear_2 = nn.Linear(hidden_size[1], hidden_size[2])
            if activation_f[2] == 'relu':
                self.activation_3 = nn.ReLU()
            elif activation_f[2] == 'tanh':
                self.activation_3 = nn.Tanh()
            elif activation_f[2] == 'sigmoid':
                self.activation_3 = nn.Sigmoid()
            else:
                self.activation_3 = nn.Softmax()
            self.linear_3 = nn.Linear(hidden_size[2], hidden_size[3])
            if activation_f[3] == 'relu':
                self.activation_4 = nn.ReLU()
            elif activation_f[3] == 'tanh':
                self.activation_4 = nn.Tanh()
            elif activation_f[3] == 'sigmoid':
                self.activation_4 = nn.Sigmoid()
            else:
                self.activation_4 = nn.Softmax()
            self.linear_4 = nn.Linear(hidden_size[3], hidden_size[4])
            if activation_f[4] == 'relu':
                self.activation_5 = nn.ReLU()
            elif activation_f[4] == 'tanh':
                self.activation_5 = nn.Tanh()
            elif activation_f[4] == 'sigmoid':
                self.activation_5 = nn.Sigmoid()
            else:
                self.activation_5 = nn.Softmax()
            self.linear_5 = nn.Linear(hidden_size[4], num_classes)
        if self.no_layers == 6:
            self.input_layer = nn.Linear(input_size, hidden_size[0])
            if activation_f[0] == 'relu':
                self.activation_1 = nn.ReLU()
            elif activation_f[0] == 'tanh':
                self.activation_1 = nn.Tanh()
            elif activation_f[0] == 'sigmoid':
                self.activation_1 = nn.Sigmoid()
            else:
                self.activation_1 = nn.Softmax()
            self.linear_1 = nn.Linear(hidden_size[0], hidden_size[1])
            if activation_f[1] == 'relu':
                self.activation_2 = nn.ReLU()
            elif activation_f[1] == 'tanh':
                self.activation_2 = nn.Tanh()
            elif activation_f[1] == 'sigmoid':
                self.activation_2 = nn.Sigmoid()
            else:
                self.activation_2 = nn.Softmax()
            self.linear_2 = nn.Linear(hidden_size[1], hidden_size[2])
            if activation_f[2] == 'relu':
                self.activation_3 = nn.ReLU()
            elif activation_f[2] == 'tanh':
                self.activation_3 = nn.Tanh()
            elif activation_f[2] == 'sigmoid':
                self.activation_3 = nn.Sigmoid()
            else:
                self.activation_3 = nn.Softmax()
            self.linear_3 = nn.Linear(hidden_size[2], hidden_size[3])
            if activation_f[3] == 'relu':
                self.activation_4 = nn.ReLU()
            elif activation_f[3] == 'tanh':
                self.activation_4 = nn.Tanh()
            elif activation_f[3] == 'sigmoid':
                self.activation_4 = nn.Sigmoid()
            else:
                self.activation_4 = nn.Softmax()
            self.linear_4 = nn.Linear(hidden_size[3], hidden_size[4])
            if activation_f[4] == 'relu':
                self.activation_5 = nn.ReLU()
            elif activation_f[4] == 'tanh':
                self.activation_5 = nn.Tanh()
            elif activation_f[4] == 'sigmoid':
                self.activation_5 = nn.Sigmoid()
            else:
                self.activation_5 = nn.Softmax()
            self.linear_5 = nn.Linear(hidden_size[4], hidden_size[5])
            if activation_f[5] == 'relu':
                self.activation_6 = nn.ReLU()
            elif activation_f[5] == 'tanh':
                self.activation_6 = nn.Tanh()
            elif activation_f[5] == 'sigmoid':
                self.activation_6 = nn.Sigmoid()
            else:
                self.activation_6 = nn.Softmax()
            self.linear_6 = nn.Linear(hidden_size[5], num_classes)
        if self.no_layers == 7:
            self.input_layer = nn.Linear(input_size, hidden_size[0])
            if activation_f[0] == 'relu':
                self.activation_1 = nn.ReLU()
            elif activation_f[0] == 'tanh':
                self.activation_1 = nn.Tanh()
            elif activation_f[0] == 'sigmoid':
                self.activation_1 = nn.Sigmoid()
            else:
                self.activation_1 = nn.Softmax()
            self.linear_1 = nn.Linear(hidden_size[0], hidden_size[1])
            if activation_f[1] == 'relu':
                self.activation_2 = nn.ReLU()
            elif activation_f[1] == 'tanh':
                self.activation_2 = nn.Tanh()
            elif activation_f[1] == 'sigmoid':
                self.activation_2 = nn.Sigmoid()
            else:
                self.activation_2 = nn.Softmax()
            self.linear_2 = nn.Linear(hidden_size[1], hidden_size[2])
            if activation_f[2] == 'relu':
                self.activation_3 = nn.ReLU()
            elif activation_f[2] == 'tanh':
                self.activation_3 = nn.Tanh()
            elif activation_f[2] == 'sigmoid':
                self.activation_3 = nn.Sigmoid()
            else:
                self.activation_3 = nn.Softmax()
            self.linear_3 = nn.Linear(hidden_size[2], hidden_size[3])
            if activation_f[3] == 'relu':
                self.activation_4 = nn.ReLU()
            elif activation_f[3] == 'tanh':
                self.activation_4 = nn.Tanh()
            elif activation_f[3] == 'sigmoid':
                self.activation_4 = nn.Sigmoid()
            else:
                self.activation_4 = nn.Softmax()
            self.linear_4 = nn.Linear(hidden_size[3], hidden_size[4])
            if activation_f[4] == 'relu':
                self.activation_5 = nn.ReLU()
            elif activation_f[4] == 'tanh':
                self.activation_5 = nn.Tanh()
            elif activation_f[4] == 'sigmoid':
                self.activation_5 = nn.Sigmoid()
            else:
                self.activation_5 = nn.Softmax()
            self.linear_5 = nn.Linear(hidden_size[4], hidden_size[5])
            if activation_f[5] == 'relu':
                self.activation_6 = nn.ReLU()
            elif activation_f[5] == 'tanh':
                self.activation_6 = nn.Tanh()
            elif activation_f[5] == 'sigmoid':
                self.activation_6 = nn.Sigmoid()
            else:
                self.activation_6 = nn.Softmax()
            self.linear_6 = nn.Linear(hidden_size[5], hidden_size[6])
            if activation_f[6] == 'relu':
                self.activation_7 = nn.ReLU()
            elif activation_f[6] == 'tanh':
                self.activation_7 = nn.Tanh()
            elif activation_f[6] == 'sigmoid':
                self.activation_7 = nn.Sigmoid()
            else:
                self.activation_7 = nn.Softmax()
            self.linear_7 = nn.Linear(hidden_size[6], num_classes)
        if self.no_layers == 8:
            self.input_layer = nn.Linear(input_size, hidden_size[0])
            if activation_f[0] == 'relu':
                self.activation_1 = nn.ReLU()
            elif activation_f[0] == 'tanh':
                self.activation_1 = nn.Tanh()
            elif activation_f[0] == 'sigmoid':
                self.activation_1 = nn.Sigmoid()
            else:
                self.activation_1 = nn.Softmax()
            self.linear_1 = nn.Linear(hidden_size[0], hidden_size[1])
            if activation_f[1] == 'relu':
                self.activation_2 = nn.ReLU()
            elif activation_f[1] == 'tanh':
                self.activation_2 = nn.Tanh()
            elif activation_f[1] == 'sigmoid':
                self.activation_2 = nn.Sigmoid()
            else:
                self.activation_2 = nn.Softmax()
            self.linear_2 = nn.Linear(hidden_size[1], hidden_size[2])
            if activation_f[2] == 'relu':
                self.activation_3 = nn.ReLU()
            elif activation_f[2] == 'tanh':
                self.activation_3 = nn.Tanh()
            elif activation_f[2] == 'sigmoid':
                self.activation_3 = nn.Sigmoid()
            else:
                self.activation_3 = nn.Softmax()
            self.linear_3 = nn.Linear(hidden_size[2], hidden_size[3])
            if activation_f[3] == 'relu':
                self.activation_4 = nn.ReLU()
            elif activation_f[3] == 'tanh':
                self.activation_4 = nn.Tanh()
            elif activation_f[3] == 'sigmoid':
                self.activation_4 = nn.Sigmoid()
            else:
                self.activation_4 = nn.Softmax()
            self.linear_4 = nn.Linear(hidden_size[3], hidden_size[4])
            if activation_f[4] == 'relu':
                self.activation_5 = nn.ReLU()
            elif activation_f[4] == 'tanh':
                self.activation_5 = nn.Tanh()
            elif activation_f[4] == 'sigmoid':
                self.activation_5 = nn.Sigmoid()
            else:
                self.activation_5 = nn.Softmax()
            self.linear_5 = nn.Linear(hidden_size[4], hidden_size[5])
            if activation_f[5] == 'relu':
                self.activation_6 = nn.ReLU()
            elif activation_f[5] == 'tanh':
                self.activation_6 = nn.Tanh()
            elif activation_f[5] == 'sigmoid':
                self.activation_6 = nn.Sigmoid()
            else:
                self.activation_6 = nn.Softmax()
            self.linear_6 = nn.Linear(hidden_size[5], hidden_size[6])
            if activation_f[6] == 'relu':
                self.activation_7 = nn.ReLU()
            elif activation_f[6] == 'tanh':
                self.activation_7 = nn.Tanh()
            elif activation_f[6] == 'sigmoid':
                self.activation_7 = nn.Sigmoid()
            else:
                self.activation_7 = nn.Softmax()
            self.linear_7 = nn.Linear(hidden_size[6], hidden_size[7])
            if activation_f[7] == 'relu':
                self.activation_8 = nn.ReLU()
            elif activation_f[7] == 'tanh':
                self.activation_8 = nn.Tanh()
            elif activation_f[7] == 'sigmoid':
                self.activation_8 = nn.Sigmoid()
            else:
                self.activation_8 = nn.Softmax()
            self.linear_8 = nn.Linear(hidden_size[7], num_classes)
        if self.no_layers == 9:
            self.input_layer = nn.Linear(input_size, hidden_size[0])
            if activation_f[0] == 'relu':
                self.activation_1 = nn.ReLU()
            elif activation_f[0] == 'tanh':
                self.activation_1 = nn.Tanh()
            elif activation_f[0] == 'sigmoid':
                self.activation_1 = nn.Sigmoid()
            else:
                self.activation_1 = nn.Softmax()
            self.linear_1 = nn.Linear(hidden_size[0], hidden_size[1])
            if activation_f[1] == 'relu':
                self.activation_2 = nn.ReLU()
            elif activation_f[1] == 'tanh':
                self.activation_2 = nn.Tanh()
            elif activation_f[1] == 'sigmoid':
                self.activation_2 = nn.Sigmoid()
            else:
                self.activation_2 = nn.Softmax()
            self.linear_2 = nn.Linear(hidden_size[1], hidden_size[2])
            if activation_f[2] == 'relu':
                self.activation_3 = nn.ReLU()
            elif activation_f[2] == 'tanh':
                self.activation_3 = nn.Tanh()
            elif activation_f[2] == 'sigmoid':
                self.activation_3 = nn.Sigmoid()
            else:
                self.activation_3 = nn.Softmax()
            self.linear_3 = nn.Linear(hidden_size[2], hidden_size[3])
            if activation_f[3] == 'relu':
                self.activation_4 = nn.ReLU()
            elif activation_f[3] == 'tanh':
                self.activation_4 = nn.Tanh()
            elif activation_f[3] == 'sigmoid':
                self.activation_4 = nn.Sigmoid()
            else:
                self.activation_4 = nn.Softmax()
            self.linear_4 = nn.Linear(hidden_size[3], hidden_size[4])
            if activation_f[4] == 'relu':
                self.activation_5 = nn.ReLU()
            elif activation_f[4] == 'tanh':
                self.activation_5 = nn.Tanh()
            elif activation_f[4] == 'sigmoid':
                self.activation_5 = nn.Sigmoid()
            else:
                self.activation_5 = nn.Softmax()
            self.linear_5 = nn.Linear(hidden_size[4], hidden_size[5])
            if activation_f[5] == 'relu':
                self.activation_6 = nn.ReLU()
            elif activation_f[5] == 'tanh':
                self.activation_6 = nn.Tanh()
            elif activation_f[5] == 'sigmoid':
                self.activation_6 = nn.Sigmoid()
            else:
                self.activation_6 = nn.Softmax()
            self.linear_6 = nn.Linear(hidden_size[5], hidden_size[6])
            if activation_f[6] == 'relu':
                self.activation_7 = nn.ReLU()
            elif activation_f[6] == 'tanh':
                self.activation_7 = nn.Tanh()
            elif activation_f[6] == 'sigmoid':
                self.activation_7 = nn.Sigmoid()
            else:
                self.activation_7 = nn.Softmax()
            self.linear_7 = nn.Linear(hidden_size[6], hidden_size[7])
            if activation_f[7] == 'relu':
                self.activation_8 = nn.ReLU()
            elif activation_f[7] == 'tanh':
                self.activation_8 = nn.Tanh()
            elif activation_f[7] == 'sigmoid':
                self.activation_8 = nn.Sigmoid()
            else:
                self.activation_8 = nn.Softmax()
            self.linear_8 = nn.Linear(hidden_size[7], hidden_size[8])
            if activation_f[8] == 'relu':
                self.activation_9 = nn.ReLU()
            elif activation_f[8] == 'tanh':
                self.activation_9 = nn.Tanh()
            elif activation_f[8] == 'sigmoid':
                self.activation_9 = nn.Sigmoid()
            else:
                self.activation_9 = nn.Softmax()
            self.linear_9 = nn.Linear(hidden_size[8], num_classes)
        if self.no_layers == 10:
            self.input_layer = nn.Linear(input_size, hidden_size[0])
            if activation_f[0] == 'relu':
                self.activation_1 = nn.ReLU()
            elif activation_f[0] == 'tanh':
                self.activation_1 = nn.Tanh()
            elif activation_f[0] == 'sigmoid':
                self.activation_1 = nn.Sigmoid()
            else:
                self.activation_1 = nn.Softmax()
            self.linear_1 = nn.Linear(hidden_size[0], hidden_size[1])
            if activation_f[1] == 'relu':
                self.activation_2 = nn.ReLU()
            elif activation_f[1] == 'tanh':
                self.activation_2 = nn.Tanh()
            elif activation_f[1] == 'sigmoid':
                self.activation_2 = nn.Sigmoid()
            else:
                self.activation_2 = nn.Softmax()
            self.linear_2 = nn.Linear(hidden_size[1], hidden_size[2])
            if activation_f[2] == 'relu':
                self.activation_3 = nn.ReLU()
            elif activation_f[2] == 'tanh':
                self.activation_3 = nn.Tanh()
            elif activation_f[2] == 'sigmoid':
                self.activation_3 = nn.Sigmoid()
            else:
                self.activation_3 = nn.Softmax()
            self.linear_3 = nn.Linear(hidden_size[2], hidden_size[3])
            if activation_f[3] == 'relu':
                self.activation_4 = nn.ReLU()
            elif activation_f[3] == 'tanh':
                self.activation_4 = nn.Tanh()
            elif activation_f[3] == 'sigmoid':
                self.activation_4 = nn.Sigmoid()
            else:
                self.activation_4 = nn.Softmax()
            self.linear_4 = nn.Linear(hidden_size[3], hidden_size[4])
            if activation_f[4] == 'relu':
                self.activation_5 = nn.ReLU()
            elif activation_f[4] == 'tanh':
                self.activation_5 = nn.Tanh()
            elif activation_f[4] == 'sigmoid':
                self.activation_5 = nn.Sigmoid()
            else:
                self.activation_5 = nn.Softmax()
            self.linear_5 = nn.Linear(hidden_size[4], hidden_size[5])
            if activation_f[5] == 'relu':
                self.activation_6 = nn.ReLU()
            elif activation_f[5] == 'tanh':
                self.activation_6 = nn.Tanh()
            elif activation_f[5] == 'sigmoid':
                self.activation_6 = nn.Sigmoid()
            else:
                self.activation_6 = nn.Softmax()
            self.linear_6 = nn.Linear(hidden_size[5], hidden_size[6])
            if activation_f[6] == 'relu':
                self.activation_7 = nn.ReLU()
            elif activation_f[6] == 'tanh':
                self.activation_7 = nn.Tanh()
            elif activation_f[6] == 'sigmoid':
                self.activation_7 = nn.Sigmoid()
            else:
                self.activation_7 = nn.Softmax()
            self.linear_7 = nn.Linear(hidden_size[6], hidden_size[7])
            if activation_f[7] == 'relu':
                self.activation_8 = nn.ReLU()
            elif activation_f[7] == 'tanh':
                self.activation_8 = nn.Tanh()
            elif activation_f[7] == 'sigmoid':
                self.activation_8 = nn.Sigmoid()
            else:
                self.activation_8 = nn.Softmax()
            self.linear_8 = nn.Linear(hidden_size[7], hidden_size[8])
            if activation_f[8] == 'relu':
                self.activation_9 = nn.ReLU()
            elif activation_f[8] == 'tanh':
                self.activation_9 = nn.Tanh()
            elif activation_f[8] == 'sigmoid':
                self.activation_9 = nn.Sigmoid()
            else:
                self.activation_9 = nn.Softmax()
            self.linear_9 = nn.Linear(hidden_size[8], hidden_size[9])
            if activation_f[9] == 'relu':
                self.activation_10 = nn.ReLU()
            elif activation_f[9] == 'tanh':
                self.activation_10 = nn.Tanh()
            elif activation_f[9] == 'sigmoid':
                self.activation_10 = nn.Sigmoid()
            else:
                self.activation_10 = nn.Softmax()
            self.linear_10 = nn.Linear(hidden_size[9], num_classes)

    def forward(self, x):
        if self.no_layers >= 1:
            output = self.input_layer(x)
            output = self.activation_1(output)
            output = self.linear_1(output)
        if self.no_layers >= 2:
            output = self.activation_2(output)
            output = self.linear_2(output)
        if self.no_layers >= 3:
            output = self.activation_3(output)
            output = self.linear_3(output)
        if self.no_layers >= 4:
            output = self.activation_4(output)
            output = self.linear_4(output)
        if self.no_layers >= 5:
            output = self.activation_5(output)
            output = self.linear_5(output)
        if self.no_layers >= 6:
            output = self.activation_6(output)
            output = self.linear_6(output)
        if self.no_layers >= 7:
            output = self.activation_7(output)
            output = self.linear_7(output)
        if self.no_layers >= 8:
            output = self.activation_8(output)
            output = self.linear_8(output)
        if self.no_layers >= 9:
            output = self.activation_9(output)
            output = self.linear_9(output)
        if self.no_layers == 10:
            output = self.activation_10(output)
            output = self.linear_10(output)

        return output
