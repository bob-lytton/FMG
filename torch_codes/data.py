import os
import pickle

import numpy as np
import torch
from torch.utils.data import Dataset

def make_embedding(user_features, item_features):
    user_concat = torch.cat(user_features, 1)
    item_concat = torch.cat(item_features, 1)

if __name__ == "__main__":
    pass