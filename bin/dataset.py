import torch
import itertools
import numpy as np

from torch.utils.data import Dataset, DataLoader

# +
energy_ref = {'GC': -14.59, 'GT': -10.51, 'AC': -10.51, 'GA': -9.81, 'TC': -9.81, 'CG': -9.69,
       'CC': -8.26, 'GG': -8.26, 'AT': -6.57, 'CA': -6.57, 'TG': -6.57, 'CT': -6.78,
       'AG': -6.78, 'TT': -5.37, 'AA': -5.37, 'TA': -3.82}
energy = -np.array(list(energy_ref.values()))
energy_normed = 2*(energy - np.min(energy))/(np.max(energy) - np.min(energy)) - 1
energy_ref_normed = dict(zip(energy_ref.keys(), energy_normed))

bendability_ref = {'AAT': -0.28, 'AAA': -0.274, 'CCA': -0.246, 'AAC': -0.205, 'ACT': -0.183, 'CCG': -0.136,
    'ATC': -0.11, 'AAG': -0.081, 'CGC': -0.077, 'AGG': -0.057, 'GAA': -0.037, 'ACG': -0.033,
    'ACC': -0.032, 'GAC': -0.013, 'CCC': -0.012, 'ACA': -0.006, 'CGA': -0.003, 'GGA': 0.013,
    'CAA': 0.015, 'AGC': 0.017, 'GTA': 0.025, 'AGA': 0.027, 'CTC': 0.031, 'CAC': 0.04,
    'TAA': 0.068, 'GCA': 0.076, 'CTA': 0.09, 'GCC': 0.107, 'ATG': 0.134, 'CAG': 0.175,
    'ATA': 0.182, 'TCA': 0.194, 'ATT': -0.28, 'TTT': -0.274, 'TGG': -0.246, 'GTT': -0.205,
    'AGT': -0.183, 'CGG': -0.136, 'GAT': -0.11, 'CTT': -0.081, 'GCG': -0.077, 'CCT': -0.057,
    'TTC': -0.037, 'CGT': -0.033, 'GGT': -0.032, 'GTC': -0.013, 'GGG': -0.012, 'TGT': -0.006,
    'TCG': -0.003, 'TCC': 0.013, 'TTG': 0.015, 'GCT': 0.017, 'TAC': 0.025, 'TCT': 0.027,
    'GAG': 0.031, 'GTG': 0.04, 'TTA': 0.068, 'TGC': 0.076, 'TAG': 0.09, 'GGC': 0.107,
    'CAT': 0.134, 'CTG': 0.175, 'TAT': 0.182, 'TGA': 0.194}

bendability = np.array(list(bendability_ref.values()))
bendability_normed = 2*(bendability - np.min(bendability))/(np.max(bendability) - np.min(bendability)) - 1
bendability_ref_normed = dict(zip(bendability_ref.keys(), bendability_normed))


# -

class SeqDataset_Mono(Dataset):
    def __init__(self, seqs, labels):
        self.seqs = seqs
        self.labels = labels
        assert len(self.labels) == len(self.seqs)
    
    def seq2onehot(self, seq):   
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
    
    def energyStacking(self, seq):
        energy = []
        energy.append(self.avg_energy)
        
        for i in range(len(seq) - 1):
            dimer = ''.join(seq[i:i+2])
            dimer_val = self.energy_ref[dimer]
            #energy.append(energy[-1] + dimer_val)
            energy.append(dimer_val)
        
        return np.float32(np.array(energy))
    
    def dnaBendability(self, seq):
        bend = []
        bend.append(self.avg_bend)
        
        for i in range(len(seq) - 2):
            trimer = ''.join(seq[i:i+3])
            trimer_bend = self.bend_ref[trimer]
            bend.append(trimer_bend)
            
        bend.append(self.avg_bend)
        
        return np.float32(np.array(bend))

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        seq = self.seqs[idx]
        seq = list(itertools.chain.from_iterable(seq))
        onehot = self.seq2onehot(seq)
        
        label = self.labels[idx]
        label = np.float32(label)
        label = np.transpose(label, (1,0))
        label = torch.tensor(label)

        return onehot, label


class SeqDataset_SE(SeqDataset_Mono):
    def __init__(self, seqs, labels, energy_ref):
        self.seqs = seqs
        self.labels = labels
        self.energy_ref = energy_ref
        self.avg_energy = sum(energy_ref.values())/len(energy_ref)
        
        assert len(self.labels) == len(self.seqs)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        seq = self.seqs[idx]
        seq = list(itertools.chain.from_iterable(seq))
        onehot = self.seq2onehot(seq)
        energy = self.energyStacking(seq)
        seq = np.vstack([onehot, energy])
        
        label = self.labels[idx]
        label = np.float32(label)
        label = np.transpose(label, (1,0))
        label = torch.tensor(label)

        return seq, label


class SeqDataset_BEND(SeqDataset_Mono):
    def __init__(self, seqs, labels, bend_ref):
        self.seqs = seqs
        self.labels = labels
        self.bend_ref = bend_ref
        self.avg_bend = sum(bend_ref.values())/len(bend_ref)
        
        assert len(self.labels) == len(self.seqs)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        seq = self.seqs[idx]
        seq = list(itertools.chain.from_iterable(seq))
        onehot = self.seq2onehot(seq)
        bend = self.dnaBendability(seq)
        seq = np.vstack([onehot, bend])
        
        label = self.labels[idx]
        label = np.float32(label)
        label = np.transpose(label, (1,0))
        label = torch.tensor(label)

        return seq, label


class SeqDataset_SE_BEND(SeqDataset_Mono):
    def __init__(self, seqs, labels, bend_ref, energy_ref):
        self.seqs = seqs
        self.labels = labels
        self.bend_ref = bend_ref
        self.avg_bend = sum(bend_ref.values())/len(bend_ref)
        
        self.energy_ref = energy_ref
        self.avg_energy = sum(energy_ref.values())/len(energy_ref)
        
        assert len(self.labels) == len(self.seqs)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        seq = self.seqs[idx]
        seq = list(itertools.chain.from_iterable(seq))
        onehot = self.seq2onehot(seq)
        bend = self.dnaBendability(seq)
        energy = self.energyStacking(seq)
        seq = np.vstack([onehot, energy, bend])
        
        label = self.labels[idx]
        label = np.float32(label)
        label = np.transpose(label, (1,0))
        label = torch.tensor(label)

        return seq, label
