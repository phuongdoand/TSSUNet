from bin.loss import FocalLoss
from bin.model import *
from bin.utils import evaluation_metrics, evaluate_model, sliding
from bin.dataset import SeqDataset_Mono, SeqDataset_BEND, SeqDataset_SE, SeqDataset_SE_BEND, energy_ref_normed, bendability_ref_normed

import os, sys
sys.path.append(os.getcwd())

import time
import json
import torch
import random
import itertools
import numpy as np
import pandas as pd
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
ind_sigma70 = sigma70.sample(n=100, random_state=37)
train_sigma70 = sigma70[~sigma70.index.isin(ind_sigma70.index)]

# Define loss
criterion = FocalLoss()
class_loss = nn.CrossEntropyLoss()

# Create cross validation
kf = KFold(n_splits = 10, shuffle = True, random_state = 247)

train_loss_dict = {}
val_loss_dict = {}

# Cross-validation training
for fold, (train_idx,val_idx) in enumerate(kf.split(np.arange(len(train_sigma70)))):
    train_df = train_sigma70.iloc[train_idx]
    val_df  = train_sigma70.iloc[val_idx]
    
    # Generate sliding sequences
    train_seqs, train_labels = sliding(train_df)
    val_seqs, val_labels = sliding(val_df)
    
    # Create dataset
    train_set = SeqDataset_SE_BEND(train_seqs, train_labels, energy_ref=energy_ref_normed, bend_ref=bendability_ref_normed)
    val_set = SeqDataset_SE_BEND(val_seqs, val_labels, energy_ref=energy_ref_normed, bend_ref=bendability_ref_normed)
    
    # Create dataloader
    train_dl = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_dl = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=True)
    
    # Define model
    unet = Unet1D(6, 1)
    unet.cuda()
    optimizer = torch.optim.AdamW(unet.parameters(), lr=0.001)
    
    train_loss_his = []
    val_loss_his = []
    
    print("Fold number {}".format(int(fold) + 1))

    for epoch in range(1000):
        start = time.time()
        train_eloss = 0
        val_eloss = 0

        unet.train()
        for x, x_label in train_dl:
            x = x.cuda()
            x_label = x_label.cuda()
            recon, recon_conf = unet(x)
            focal_loss = criterion(recon, x_label)
            conf = class_loss(recon_conf, torch.sum(x_label, dim = -1))
            loss = focal_loss + 0.1 * conf

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_eloss += loss.cpu().item()

        unet.eval()
        with torch.no_grad():
            for i, (x, x_label) in enumerate(val_dl):
                x = x.cuda()
                x_label = x_label.cuda()
                recon, recon_conf = unet(x)
                val_focal_loss = criterion(recon, x_label) 
                val_conf = class_loss(recon_conf, torch.sum(x_label, dim = -1))
                val_loss = val_focal_loss + 0.1 * val_conf

                val_eloss += val_loss.cpu().item()

        train_loss_his.append(train_eloss/len(train_dl))
        val_loss_his.append(val_eloss/len(val_dl))

        if epoch == 0:
            loss_val_history = val_loss_his[-1]
            patience = 0
        else:
            loss_val_history = np.append(loss_val_history, val_loss_his[-1])

        if val_loss_his[-1] < 0.00000000000000001 + np.min(loss_val_history):
            patience = 0
            model = "best_model_BEND_SE_fold_" + str(fold + 1) + ".pt"
            torch.save(unet.state_dict(), model)
        else:
            patience +=1

        end = time.time()
        hours, rem = divmod(end-start, 3600)
        minutes, seconds = divmod(rem, 60)
        print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))
        print(epoch, patience, val_eloss/len(val_dl), np.min(loss_val_history))

        if patience == 5:
            break
            
        train_loss_dict[fold] = train_loss_his
        val_loss_dict[fold] = val_loss_his

# Test the model's performance on the test set
unet = Unet1D(6, 1)
unet.cuda()
test_seqs, test_labels = sliding(ind_sigma70, random_sampling=True)

test_set = SeqDataset_SE_BEND(test_seqs, test_labels, energy_ref=energy_ref_normed, bend_ref=bendability_ref_normed)
test_dl = DataLoader(test_set, batch_size=BATCH_SIZE)

for fold in range(10):
    model = "best_model_BEND_SE_fold_" + str(fold + 1) + ".pt"
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
                    
    print("Fold: {}".format(fold + 1))
    for thres in np.linspace(0.01, 0.15, num=15):
        pred = recon_data.copy()
    
        for i in range(len(recon_data)):
            pred[i][pred[i] >= thres] = 1
            pred[i][pred[i] < thres] = 0

        pos_list = []
        neg_list = []
        correct_neg = 0
        correct_pos = 0

        for i in range(len(test_set)):
            if sum(test_set[i][1][0].detach().numpy()) == 0.0:
                neg_list.append(i)
                if sum(pred[i][0]) == 0.0:
                    correct_neg += 1
            else:
                pos_list.append(i)
                if sum(pred[i][0]) != 0.0:
                    correct_pos += 1

        fn = len(pos_list) - correct_pos
        fp = len(neg_list) - correct_neg
        print(correct_pos, fp, correct_neg, fn)

        acc, spec, recall, f1, mcc = evaluation_metrics(correct_pos, fp, correct_neg, fn)
        print("Threshold: {}, Acc: {:.3f}, Specificity: {:.3f}, Sensitivity: {:.3f}, F1: {:.3f}, MCC: {:.3f}".format(thres, acc, spec, recall, f1, mcc))
