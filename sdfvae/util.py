import torch
import os
import torch.utils.data as data

class KpiReader(data.Dataset):
    def __init__(self, path, size):
        self.path = path
        self.length = size

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        item = torch.load(self.path+'/%d.seq' % (idx+1))
        return item['ts'], item['label'], item['value']

