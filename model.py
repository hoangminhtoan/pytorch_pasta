from inspect import Parameter
import os 
import numpy as np 

import torch 
import torch.nn as nn 
import torch.functional as F 

class CroStchNet3D(nn.Module):
    def __init__(self, in_planes, out_planes, dropout_rate=0.2, kernel_size=3, padding='same') -> None:
        super(CroStchNet3D, self).__init__()
        self.dpr = dropout_rate
        self.padding = (kernel_size - 1) // 2 if padding=='same' else 0

        self.conv = nn.Conv3d(in_planes, out_planes, kernel_size=3, padding=self.padding, bias=False)
        self.bn = nn.BatchNorm3d()
        self.act = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout3d(p=self.dpr)
        self.pooling = nn.AvgPool3d(kernel_size=2)

    def forward(self, x):
        for _ in range(2):
            x = self.conv(x)
            x = self.bn(x)
            x = self.act(x)
            x = self.dropout(x)
        
        x = self.pooling(x)

        return x 

class CrossStitch3D(nn.Module):
    def __init__(self, input_shape) -> None:
        super(CrossStitch3D, self).__init__()

        self.shape = np.prod(input_shape[1:])
        self.input_shape_1 = self.shape._value
        self.input_shape_2 = self.shape._value
        self.output_shape = self.input_shape_1 + self.input_shape_2
        # self.output_shape = [input_shape[1], input_shape[2], input_shape[3]

        self.cross_stitch = nn.init.eye_(Parameter(torch.empy((self.output_shape, self.output_shape))))

    def forward(self, x):
        x1 = x[0].view((self.shape,))
        x2 = x[1].view((self.shape,))

        inputss = torch.concat((x1, x2), dim=1)
        output = torch.matmul(inputss, self.cross_stitch)
        output1 = output[:, :self.input_shape_1]
        output2 = output[:, :self.input_shape_2]
        #print(f'output1_shape {output1.shape}\t output2_shape {output2.shape}\n')

        s1 = x[0].shape[1]._value
        s2 = x[0].shape[2]._value
        s3 = x[0].shape[3]._value

        output1 = output1.view(x[0][0], s1, s2, s3)
        output2 = output2.view(x[0][0], s1, s2, s3)

        return output1, output2

class Model():
    src = torch.randn(1, 64, 64) # 
    gt = torch.randn(1, 64, 64) # ground-truth

    x1 = src
    x2 = gt 

    num_channels = [ 64, 32, 32, 32, 16, 16]
    if unit_num == 1:
        cs_level = [                 5]
    elif unit_num == 2:
        cs_level = [                4, 5]
    elif unit_num == 3:
        cs_level = [              3,  4, 5]
    elif unit_num == 4:
        cs_level = [              2, 3,  4, 5]
    elif unit_num == 5:
        cs_level = [              1, 2, 3,  4, 5]
    elif unit_num == 6:
        cs_level = [              0, 1, 2, 3,  4, 5]

