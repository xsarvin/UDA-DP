import torch.nn as nn
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from tqdm import tqdm
import torch
import os
from code_dataset import *
from torch.utils.data import DataLoader
import numpy as np


def feat_update(model, loader1, loader2, img_num1, img_num2, embedding_dim):
    feat_memory1 = torch.zeros((img_num1, embedding_dim), dtype=torch.float32)
    feat_memory2 = torch.zeros((img_num2, embedding_dim), dtype=torch.float32)
    label_memory1 = torch.zeros(img_num1, dtype=torch.long)
    label_memory2 = torch.zeros(img_num2, dtype=torch.long)

    model.eval()
    for step, (batch_ast_x, idx, batch_y) in enumerate(loader1):
        (feat, cls), _ = model(batch_ast_x.long().cuda(), batch_ast_x.long().cuda())
        feat = feat / (torch.norm(feat, 2, 1, True) + 1e-8)
        feat_memory1[idx] = feat.detach().cpu()
        label_memory1[idx] = batch_y.long().flatten().cpu()

    for step, (batch_ast_x, idx, batch_y) in enumerate(loader2):
        (feat, cls), _ = model(batch_ast_x.long().cuda(), batch_ast_x.long().cuda())
        feat = feat / (torch.norm(feat, 2, 1, True) + 1e-8)
        feat_memory2[idx] = feat.detach().cpu()
        pse_udo_label = torch.max(cls, 1)[1]
        label_memory2[idx] = pse_udo_label.cpu()

    return feat_memory1, feat_memory2, label_memory1, label_memory2


def pretrain(nn_params, optim, model, init_lr, train_loader, test_loader, path, dir):
    best_f1 = 0
    best_p = 0
    best_r = 0

    for epoch in range(nn_params['N_EPOCH']):
        optimizer = optim.Adam(model.parameters(), lr=init_lr, betas=(0.9, 0.999), eps=1e-8,
                               weight_decay=nn_params['L2_WEIGHT'], amsgrad=False)

        predict = []
        label = []

        for step, (batch_ast_x, idx, batch_y) in enumerate(train_loader):
            model.train()
            (feat, cls), _ = model(batch_ast_x.long().cuda(), state="pretraining")
            criterion = nn.CrossEntropyLoss()
            batch_y = batch_y.long().flatten()
            loss = criterion(cls, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        for step, (batch_ast_x, idx, batch_y) in enumerate(test_loader):
            model.eval()
            (feat, cls), _ = model(batch_ast_x.long().cuda())
            pre = torch.max(cls, 1)[1]
            batch_y = batch_y.long().flatten()
            # Loss

            label += batch_y.cpu()
            predict += pre.cpu()

        f1 = f1_score(label, predict)
        p = precision_score(label, predict)
        r = recall_score(label, predict)
        print('Epoch: [{}/{}], f1_score: {:.6f} best_f1:{:.6f}'.format(epoch, nn_params['N_EPOCH'], f1, best_f1))
        if best_f1 < f1:
            best_f1 = f1
            best_r = r
            best_p = p
            if not os.path.exists(dir):
                os.mkdir(dir)
            torch.save(model.state_dict(), "{}/{}_{}".format(dir, path[0], path[1]))
            print("the model {}_{}_{} saved !".format(dir, path[0], path[1]))

    return best_p, best_r, best_f1


def compute_knn_idx(feat_memory1, feat_memory2, topk=5):
    simmat = torch.matmul(feat_memory1, feat_memory2.T)

    _, knnidx_topk = torch.topk(simmat, k=topk, dim=0)

    count_target_usage(knnidx_topk, len(feat_memory1))

    return knnidx_topk


def count_target_usage(knnidx_topk, length,match=True):
    if match:

        topk_id=knnidx_topk.cpu().numpy()
        target_id = set()
        for i in range(len(knnidx_topk)):
            for id in topk_id[i]:
                target_id.add(id)

        ratio = float(len(target_id) / length)
        print("the labeled sample matched ratio is {}".format(ratio))
    else:
        target_id=set(torch.tensor(knnidx_topk).numpy())
        ratio = float(len(target_id) / length)
        print("the labeled sample filtered ratio is {}".format(ratio))

def generate_new_dataset(tok_k_id, s_dataset, t_dataset, label1, target_pseudo_label, num1, num2, with_pseudo_label_filter=True):
    # generate new dataset
    train_set = []
    new_target_knnidx = []
    new_targetidx = []

    # combine_target_sample:
    for idx, data in enumerate(tqdm(t_dataset)):
        t_ast, t_idx, t_label = data
        curidx = tok_k_id.transpose(1,0)[t_idx]
        for i in range(len(curidx)):
            source_data = s_dataset[curidx[i]]
            s_ast, s_idx, s_label = source_data
            flag = t_label == target_pseudo_label[t_idx]
            if (with_pseudo_label_filter and flag) or not with_pseudo_label_filter:
                new_targetidx.append(t_idx)
                new_target_knnidx.append(curidx[i])
                train_set.append((s_ast, t_ast, s_label.cpu().numpy()[0], t_label.cpu().numpy()[0],
                                  s_idx, t_idx))

    count_target_usage(new_target_knnidx,len(label1),match=False)

    new_dataset = Transform_Dadaset(train_set)

    train_loader = DataLoader(
        new_dataset, batch_size=16,
        shuffle=True, drop_last=True,
        collate_fn=source_target_train_collate_fn,
        pin_memory=True
    )

    return train_loader
