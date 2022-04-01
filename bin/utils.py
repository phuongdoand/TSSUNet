import math
import random
import numpy as np


def evaluation_metrics(tp, fp, tn, fn):
    acc = (tp+tn)/(tp+fp+tn+fn)
    recall = tp/(tp+fn)
    spec = tn/(tn+fp)
    f1 = 2*tp/(2*tp+fp+fn)
    mcc = (tp*tn-fp*fn)/math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
    return acc, spec, recall, f1, mcc


def evaluate_model(model, dataset, dataloader, threshold=0.05):
    recon_data = []
    recon_conf_data = []
    labels = []
    with torch.no_grad():    
        for i, (x, x_label) in enumerate(dataloader):
            x = x.cuda()
            recon, recon_conf = model(x)
            if i == 0:
                recon_data = recon.cpu().numpy()
            else:
                recon_data = np.vstack((recon_data, recon.cpu().numpy()))
    
    for i in range(len(recon_data)):
        recon_data[i][recon_data[i] >= threshold] = 1
        recon_data[i][recon_data[i] < threshold] = 0
    
    pos_list = []
    neg_list = []
    correct_neg = 0
    correct_pos = 0

    for i in range(len(dataset)):
        if sum(dataset[i][1][0].detach().numpy()) == 0.0:
            neg_list.append(i)
            if sum(recon_data[i][0]) == 0.0:
                correct_neg += 1
        else:
            pos_list.append(i)
            if sum(recon_data[i][0]) != 0.0:
                correct_pos += 1
                
    fn = len(pos_list) - correct_pos
    fp = len(neg_list) - correct_neg
    
    return evaluation_metrics(correct_pos, fp, correct_neg, fn)


def sliding(df, random_sampling=False):
    negative = []
    seqs = []
    labels = []

    for idx, row in df.iterrows():
        # Negative data
        for i in range(0, 371): # 371 sequences
            e = i + 150
            negative.append(row["seq"][i:e])

        for i in range(431, 851): # 420 sequences
            e = i + 150
            negative.append(row["seq"][i:e])
        
        if random_sampling:
            # Random sampling the negative data to 60 sequences
            random.seed(idx)
            random_negative_seq = random.sample(negative, 60)
            for negative_seq in random_negative_seq:
                seqs.append(negative_seq)

            for i in range(len(random_negative_seq)):
                labels.append([0]*150)
        else:
            random.seed(idx)
            random_negative_seq = random.sample(negative, 400)
            for negative_seq in random_negative_seq:
                seqs.append(negative_seq)

            for i in range(len(random_negative_seq)):
                labels.append([0]*150)

        # Positive data
        for i in range(371, 431): # 60 sequences
            e = i + 150
            seqs.append(row["seq"][i:e])
            labels.append(row["labels"][i:e])

    labels_com = []
    for lab in labels:
        lab = [lab]
        lab_arr = np.asarray(lab)
        lab_arr = lab_arr.transpose(1,0)
        labels_com.append(lab_arr)

    return seqs, np.asarray(labels_com)


