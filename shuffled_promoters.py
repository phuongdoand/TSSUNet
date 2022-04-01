from bin.model import *
from bin.utils import evaluation_metrics, evaluate_model, sliding
from bin.dataset import SeqDataset_Mono, SeqDataset_BEND, SeqDataset_SE, SeqDataset_SE_BEND, energy_ref_normed, bendability_ref_normed

import torch
import random
import numpy as np
import pandas as pd

from torch.utils.data import Dataset, DataLoader

BATCH_SIZE=256

# Load data
with open("data/shuflled_primers.txt", "r") as f_in:
    negative = np.array([])
    seq = ""
    for line in f_in:
        if not line.startswith(">"):
            if len(line) > 1:
                seq += line.rstrip()
            if len(line) == 1 and len(seq) > 0:
                negative = np.append(negative, seq)
                seq = ""   
                
# Dataloader
test_neg_label = np.zeros((150,1)).astype(int)
test_neg_labels = [test_neg_label for i in range(negative.shape[0])]

test_set = SeqDataset_BEND(negative, test_neg_labels, bendability_ref_normed)
test_dl = DataLoader(test_set, batch_size=BATCH_SIZE)

# Load model
unet = Unet1D(5, 1)
unet.cuda()
model = "best_model_BEND.pt"
unet.load_state_dict(torch.load(model))

recon_data = []
recon_conf_data = []
labels = []
with torch.no_grad():    
    for i, (x, x_label) in enumerate(test_dl):
        x = x.cuda()
        recon, recon_conf = unet(x)
        if i == 0:
            recon_data = recon.cpu().numpy()
            for i in np.argmax(recon_conf.cpu().numpy(), axis = 1).tolist():
                recon_conf_data.append(i)
            for i in torch.squeeze(torch.sum(x_label, dim = -1)).numpy().tolist():
                labels.append(i)
        else:
            recon_data = np.vstack((recon_data, recon.cpu().numpy()))
            for i in np.argmax(recon_conf.cpu().numpy(), axis = 1).tolist():
                recon_conf_data.append(i)
            for i in torch.squeeze(torch.sum(x_label, dim = -1)).numpy().tolist():
                labels.append(i)
                
for i in range(len(recon_data)):
    recon_data[i][recon_data[i] >= 0.023] = 1
    recon_data[i][recon_data[i] < 0.023] = 0

pos_list = []
neg_list = []
correct_neg = 0
correct_pos = 0

for i in range(len(test_set)):
    if sum(test_set[i][1][0].detach().numpy()) == 0.0:
        neg_list.append(i)
        if sum(recon_data[i][0]) == 0.0:
            correct_neg += 1
    else:
        pos_list.append(i)
        if sum(recon_data[i][0]) != 0.0:
            correct_pos += 1

fn = len(pos_list) - correct_pos
fp = len(neg_list) - correct_neg
print(correct_pos, fp, correct_neg, fn)

print(correct_neg/(correct_neg+fp))