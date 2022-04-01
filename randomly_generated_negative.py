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
sigma70 = pd.read_csv("data/20211213.Sigma70.txt", sep = '\t', names = ["name", "seq", "strand", "express"])
sigma_strong = sigma70[sigma70["express"] != "Weak"]

seqs = []
for i, row in sigma_strong.iterrows():
    seq = row["seq"][400:550]
    seqs.append(seq)
    
# Calculate A, C, G, T percentage
As = Cs = Gs = Ts = tot = 0
for i in range(len(seqs)):
    As = As + (seqs[i]).count('a') + (seqs[i]).count('A')
    Cs = Cs + (seqs[i]).count('c') + (seqs[i]).count('C')
    Gs = Gs + (seqs[i]).count('g') + (seqs[i]).count('G')
    Ts = Ts + (seqs[i]).count('t') + (seqs[i]).count('T')
    tot = tot + len(seqs[i])
    
As = As/tot
Cs = Cs/tot
Gs = Gs/tot
Ts = Ts/tot

random.seed(11)

def rand_nt_sig70():
    rnd = random.uniform(0,1)
    if(rnd <= As):
        return('A')
    elif(rnd > As and rnd <= As+Cs):
        return('C')
    elif(rnd > As+Cs and rnd <= As+Cs+Gs):
        return('G')
    elif(rnd > As+Cs+Gs):
        return('T')

# Generate negative data based on the calculated percentage
number = 6000
negative = np.array([])
for i in range(1,number+1):
    seq = []
    for bp in range(150):
        seq.append(rand_nt_sig70())
    negative = np.append(negative, "".join(seq))

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