from bin.loss import FocalLoss
from bin.model import *
from bin.utils import evaluation_metrics, evaluate_model, sliding
from bin.dataset import SeqDataset_Mono, SeqDataset_BEND, SeqDataset_SE, SeqDataset_SE_BEND, energy_ref_normed, bendability_ref_normed

import os, sys
sys.path.append(os.getcwd())

import csv
import time
import json
import torch
import random
import itertools
import numpy as np
import pandas as pd
import seaborn as sns
import torch.nn as nn

from torchsummary import summary
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold
from torch.utils.data import Dataset, DataLoader

# Load data
sigma70 = pd.read_csv("data/20211213.Sigma70.txt", sep = '\t', names = ["name", "seq", "strand", "express"])

# Create label
label = []
for i in range(500):
    label.append(0)
label.append(1)
for i in range(500):
    label.append(0)
    
sigma70["labels"] = [label] * len(sigma70)

torch.manual_seed(1)
use_cuda = torch.cuda.is_available()
if use_cuda:
    gpu = 0
    
BATCH_SIZE = 256 # Batch size
    
# Split independent testing dataset
ind_sigma70 = sigma70.sample(n=200, random_state=37)
train_sigma70 = sigma70[~sigma70.index.isin(ind_sigma70.index)]

val_sigma70 = ind_sigma70.sample(n=100, random_state=11)
test_sigma70 = ind_sigma70[~ind_sigma70.index.isin(val_sigma70.index)]

# Test the model's performance on the test set
unet = Unet1D(5, 1)
unet.cuda()
test_seqs, test_labels = sliding(test_sigma70, random_sampling=True)

model = "best_model_BEND.pt"
unet.load_state_dict(torch.load(model))

test_set = SeqDataset_BEND(test_seqs, test_labels, bendability_ref_normed)
test_dl = DataLoader(test_set, batch_size=BATCH_SIZE)

recon_data = []
with torch.no_grad():    
    for i, (x, _) in enumerate(test_dl):
        x = x.cuda()
        recon, _ = unet(x)
        if i == 0:
            recon_data = recon.cpu().numpy()
        else:
            recon_data = np.vstack((recon_data, recon.cpu().numpy()))
pred = recon_data.copy()
            
for i in range(len(recon_data)):
    recon_data[i][recon_data[i] >= 0.023] = 1
    recon_data[i][recon_data[i] < 0.023] = 0
    
# Identify correctly predicted positive sequences
pos_list = []
pos_id = []
correct_pos = 0

for i in range(len(test_set)):
    if sum(test_set[i][1][0].detach().numpy()) != 0.0:
        pos_list.append(i)
        if sum(recon_data[i][0]) != 0.0:
            pos_id.append(i)
            correct_pos += 1

# Identify correctly predicted negative sequences
neg_list = []
neg_id = []
correct_pos = 0

for i in range(len(test_set)):
    if sum(test_set[i][1][0].detach().numpy()) == 0.0:
        neg_list.append(i)
        if sum(recon_data[i][0]) == 0.0:
            neg_id.append(i)
            correct_pos += 1
            
distances = {}
for i in pos_id:
    pred_pos = np.argmax(pred[i][0])
    real_pos = np.argmax(test_set[i][1][0].numpy())
    distance = np.absolute(pred_pos - real_pos)
    
    if distance not in distances:
        distances[distance] = 1
    else:
        distances[distance] += 1

# Save distance to csv file
w = csv.writer(open("distances.csv", "w"))
w.writerow(["Distance", "Count"])

for key, val in distances.items():
    w.writerow([key, val])

distances = []
for i in pos_id:
    pred_pos = np.argmax(pred[i][0])
    real_pos = np.argmax(test_set[i][1][0].numpy())
    distances.append(np.absolute(pred_pos - real_pos))
    
# Plot distribution
ax = sns.histplot(distances, bins=100)
ax.set(xlabel='Distance')

plt.savefig("Distance_Distribution.pdf", format="pdf")
plt.show()

# Histogram with KDE
ax = sns.distplot(distances, hist=True, kde=True,  
             hist_kws={'edgecolor':'black'},
             kde_kws={'color': 'red'})

ax.set(xlabel='Distance')

plt.savefig("Distance_Distribution_Hist_KDE.pdf", format="pdf")
plt.show()

# KDE only
ax = sns.distplot(distances, hist=False, kde=True,  
             kde_kws={'color': 'red'})

ax.set(xlabel='Distance')

plt.savefig("Distance_Distribution_KDE.pdf", format="pdf")
plt.show()

# Histogram with cumulative
ax = sns.distplot(distances, hist=True, kde=True,  
             hist_kws={'edgecolor':'black'},
             kde_kws={'color': 'red', 'cumulative': True})

ax.set(xlabel='Distance')

plt.savefig("Distance_Distribution_Hist_Cumulative.pdf", format="pdf")
plt.show()

# Plot sliding aggregation

sliding = {}
for i, row in ind_sigma70.iterrows():
    sliding[row["name"]] = {}
    idx = 0
    for j in range(371, 431):
        sub_seq = row['seq'][j:j+150]
        sliding[row["name"]][idx] = sub_seq
        idx += 1
        
avg_bend = sum(bendability_ref_normed.values())/len(bendability_ref_normed)

def seq2onehot(seq):   
    module = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
    promoter_onehot = []
    for item in seq:
        if item == 't' or item == 'T':
            promoter_onehot.append(module[0])
        elif item == 'a' or item == 'A':
            promoter_onehot.append(module[1])
        elif item == 'g' or item == 'G':
            promoter_onehot.append(module[2])
        elif item == 'c' or item == 'C':
            promoter_onehot.append(module[3])
        else:
            promoter_onehot.append([0,0,0,0])

    data = np.array(promoter_onehot)
    data = np.float32(data)
    data = np.transpose(data, (1,0))

    return data

def dnaBendability(seq):
    bend = []
    bend.append(avg_bend)

    for i in range(len(seq) - 2):
        trimer = ''.join(seq[i:i+3])
        trimer_bend = bendability_ref_normed[trimer]
        bend.append(trimer_bend)

    bend.append(avg_bend)

    return np.float32(np.array(bend))

cmd = "rm *Stacked_Prediction.pdf"
os.system(cmd)

label = []
for i in range(130):
    label.append(0)
label.append(1)
for i in range(79):
    label.append(0)

for s in sliding:
    with torch.no_grad(): 
        recon_data = []
        
        for i in sliding[s]:
            seq = list(itertools.chain.from_iterable(sliding[s][i]))
            onehot = seq2onehot(seq)
            bend = dnaBendability(seq)
            seq = np.vstack([onehot, bend])
            seq = np.reshape(seq, (1,5,150))
            seq = torch.tensor(seq)

            recon, recon_conf = unet(seq.cuda())
            if i == 0:
                recon_data = recon.cpu().numpy()
            else:
                recon_data = np.vstack((recon_data, recon.cpu().numpy()))
        pred = recon_data.copy()

        for i in range(len(recon_data)):
            recon_data[i][recon_data[i] >= 0.023] = 1
            recon_data[i][recon_data[i] < 0.023] = 0

        fig = plt.figure(figsize = (20, 6))
        ax1 = fig.add_subplot(2, 1, 1)
        ax1.plot(label)
        ax1.set_xticks(range(0,211)[::10])
        ax1.set_xticklabels(range(-130,81)[::10])
        ax1.title.set_text(s)
           
        ax2 = fig.add_subplot(2, 1, 2)
        for j in range(recon_data.shape[0]):
            x1 = np.linspace(j, j + 150, 150)
            y1 = [recon_data.shape[0] - j] * 150
            if np.sum(recon_data[j]) != 0:
                plt.plot(x1, y1, color = 'red')
                pred_pos = np.argmax(pred[j])
                x2 = x1[pred_pos]
                y2 = recon_data.shape[0] - j
                ax2.plot(x2, y2, color = 'green', marker = '.', markersize = 10)
            else:
                ax2.plot(x1, y1, color = 'black')
        ax2.set_xticks(range(0,211)[::10])
        ax2.set_xticklabels(range(-130,81)[::10])
        
        fileName = "Sample_" + str(s) + "_Stacked_Prediction.pdf"
        plt.savefig(fileName, format="pdf")
        plt.show()

cmd = "mkdir Stacked_Images"
os.system(cmd)
cmd = "mv *Stacked_Prediction.pdf Stacked_Images"
os.system(cmd)
cmd = "tar -zcvf Stacked_Images.tar.gz Stacked_Images"
os.system(cmd)