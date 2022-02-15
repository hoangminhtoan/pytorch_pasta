import numpy as np 

import torch 
import torch.nn as nn
import torch.functional as F


def DC_Score(y_true, y_pred):
    y_true = torch.reshape(y_true, (torch.shape(y_true)[0], 1, 64, 64, 64))
    y_pred = torch.reshape(y_pred, (torch.shape(y_true)[0], 1, 64, 64, 64))

    y_true = y_true.permute(0, 2, 3, 4, 1)
    y_pred = y_pred.permute(0, 2, 3, 4, 1)

