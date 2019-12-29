import os
import pickle

import numpy as np
import torch


class AdjacentMatrixLoader(Dataset):

    def __init__(self, paths, metapaths, bin=False):
        r"""
        load the pickle files, including:
        adjacency matrix, rates and matrix factorization embeddings

        Parameters
        ----------
        paths: list
            file paths
        metapaths: list
            list of metapaths to be used
        bin: Bool.
            Decide if the files to read is binary file.
        """
        self.paths = paths
        if bin:
            with open(path, 'rb') as fw: self. = pickle.load(, fw)


class FeatureLoader(Dataset):
    r"""
    concat all features and original rates of a user into a single vector
    """