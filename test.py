import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
def eval(net, c, dataloader, device):

    scores = []
    labels = []
    indices = []
    
    net.eval()
    print('Testing...')
    with torch.no_grad():
        for i, (x, y) in enumerate(tqdm(dataloader, desc='Testing')):
            x = x.float().to(device)
            z = net(x)
            dist = torch.sum((z - c) ** 2, dim=1)
            scores.extend(dist.cpu().numpy())
            labels.extend(y.cpu().numpy())
            indices.extend(list(range(i * dataloader.batch_size, (i + 1) * dataloader.batch_size)))

    return indices, labels, scores