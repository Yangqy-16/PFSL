import numpy as np
import pandas as pd
from torch.utils.data import Sampler


class NegSampler(Sampler):
    def __init__(self, df: pd.DataFrame, neg_sample_ratio: int | float = 1.0):
        self.pos_indices = df[df['2'] == 1].index.tolist()
        self.neg_indices = df[df['2'] == 0].index.tolist()
        self.neg_sample_ratio = neg_sample_ratio

    def __iter__(self):
        pos_count = len(self.pos_indices)
        neg_count = int(pos_count * self.neg_sample_ratio)
        neg_sampled_indices = np.random.choice(self.neg_indices, neg_count, replace=False)

        sampled_indices = self.pos_indices + neg_sampled_indices.tolist()
        np.random.shuffle(sampled_indices)
        return iter(sampled_indices)

    def __len__(self):
        return len(self.pos_indices) + int(len(self.pos_indices) * self.neg_sample_ratio)
