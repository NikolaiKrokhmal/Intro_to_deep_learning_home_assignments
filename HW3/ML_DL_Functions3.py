import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as f


def ID1():
    '''
        Personal ID of the first student.
    '''
    # Insert your ID here
    return 320717184


def ID2():
    '''
        Personal ID of the second student. Fill this only if you were allowed to submit in pairs, Otherwise leave it zeros.
    '''
    # Insert your ID here
    return 000000000


class CNN(nn.Module):
    def __init__(self):  # Do NOT change the signature of this function
        super(CNN, self).__init__()
        n = 16
        kernel_size = 3
        padding = (kernel_size - 1)//2

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=n, kernel_size=kernel_size, padding=padding)
        self.conv2 = nn.Conv2d(n, 2 * n, kernel_size=kernel_size, padding=padding)
        self.conv3 = nn.Conv2d(2 * n, 4 * n, kernel_size=kernel_size, padding=padding)
        self.conv4 = nn.Conv2d(4 * n, 8 * n, kernel_size=kernel_size, padding=padding)
        # self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(8 * n * 28 * 14, 100)
        self.fc2 = nn.Linear(100, 2)

    def forward(self,inp):  # Do NOT change the signature of this function
        '''
          prerequests:
          parameter inp: the input image, pytorch tensor.
          inp.shape == (N,3,448,224):
            N   := batch size
            3   := RGB channels
            448 := Height
            224 := Width
          
          return output, pytorch tensor
          output.shape == (N,2):
            N := batch size
            2 := same/different pair
        '''

        out = self.conv1(inp)
        out = f.relu(out)
        out = f.max_pool2d(out, kernel_size=2)

        out = self.conv2(out)
        out = f.relu(out)
        out = f.max_pool2d(out, kernel_size=2)

        out = self.conv3(out)
        out = f.relu(out)
        out = f.max_pool2d(out, kernel_size=2)

        out = self.conv4(out)
        out = f.relu(out)
        out = f.max_pool2d(out, kernel_size=2)

        out = out.reshape(out.size(0), -1)
        # out = self.dropout(out)
        out = self.fc1(out)
        out = f.relu(out)
        # out = self.dropout(out)
        out = self.fc2(out)
        out = f.log_softmax(out, dim=1)

        return out


class CNNChannel(nn.Module):
    def __init__(self):  # Do NOT change the signature of this function
        super(CNNChannel, self).__init__()
        n = 32
        kernel_size = 3
        padding = (kernel_size - 1)//2

        self.conv1 = nn.Conv2d(in_channels=6, out_channels=n, kernel_size=kernel_size, padding=padding)
        self.conv2 = nn.Conv2d(n, 2 * n, kernel_size=3, padding=padding)
        self.conv3 = nn.Conv2d(2 * n, 4 * n, kernel_size=3, padding=padding)
        self.conv4 = nn.Conv2d(4 * n, 8 * n, kernel_size=3, padding=padding)
        # self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(8 * n * 14 * 14, 100)
        self.fc2 = nn.Linear(100, 2)

    # TODO: complete this class
    def forward(self, inp):  # Do NOT change the signature of this function
        '''
          prerequests:
          parameter inp: the input image, pytorch tensor
          inp.shape == (N,3,448,224):
            N   := batch size
            3   := RGB channels
            448 := Height
            224 := Width
          
          return output, pytorch tensor
          output.shape == (N,2):
            N := batch size
            2 := same/different pair
        '''
        # TODO start by changing the shape of the input to (N,6,224,224)
        # TODO: complete this function
        img_top = inp[:, :, :224, :]
        img_bot = inp[:, :, 224:, :]
        out = torch.cat((img_top, img_bot), 1)

        out = self.conv1(out)
        out = f.relu(out)
        out = f.max_pool2d(out, kernel_size=2)

        out = self.conv2(out)
        out = f.relu(out)
        out = f.max_pool2d(out, kernel_size=2)

        out = self.conv3(out)
        out = f.relu(out)
        out = f.max_pool2d(out, kernel_size=2)

        out = self.conv4(out)
        out = f.relu(out)
        out = f.max_pool2d(out, kernel_size=2)

        out = out.reshape(out.size(0), -1)
        # out = self.dropout(out)
        out = self.fc1(out)
        out = f.relu(out)
        # out = self.dropout(out)
        out = self.fc2(out)
        out = f.log_softmax(out, dim=1)

        return out

if __name__ == '__main__':
    pass
