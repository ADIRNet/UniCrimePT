import math
import pickle
import pandas as pd
import torch
import numpy as np
from torch.utils.data import TensorDataset, Dataset, DataLoader
import scipy.sparse as sp
from STGNN import B_TCN, D_GCN, ST_NB_ZeroInflated, NBNorm_ZeroInflated, NBNorm_MSE, ST_MSE, Transfer_mode
import tqdm
from sklearn import metrics
from torch import nn
import os
from itertools import cycle
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_normalized_adj(A):
    """
    Returns the degree normalized adjacency matrix. This is for K_GCN
    """
    if A[0, 0] == 0:
        A = A + np.diag(np.ones(A.shape[0], dtype=np.float32)) # if the diag has been added by 1s
    D = np.array(np.sum(A, axis=1)).reshape((-1,))
    D[D <= 10e-5] = 10e-5    # Prevent infs
    diag = np.reciprocal(np.sqrt(D))
    A_wave = np.multiply(np.multiply(diag.reshape((-1, 1)), A),
                         diag.reshape((1, -1)))
    return A_wave


def calculate_random_walk_matrix(adj_mx):
    """
    Returns the random walk adjacency matrix. This is for D_GCN
    """
    adj_mx = sp.coo_matrix(adj_mx)
    d = np.array(adj_mx.sum(1))
    d_inv = np.power(d, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat_inv = sp.diags(d_inv)
    random_walk_mx = d_mat_inv.dot(adj_mx).tocoo()
    return random_walk_mx.toarray()


class CustomDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feature = torch.tensor(self.features[idx]).to(device).float()
        label = torch.tensor(self.labels[idx]).to(device).float()

        return feature, label


class MadridDatasetLoader(object):
    def __init__(self, data_norm, device="cuda"):
        super(MadridDatasetLoader, self).__init__()
        self.data = data_norm

    def process_data(self, time_step, pre_step):
        # 假设边的权重为 1
        self.data = np.expand_dims(self.data, axis=-1)
        nodes, timeLength = self.data.shape[0], self.data.shape[1]
        train_seq, train_label = [], []
        for i in range(0, timeLength - time_step - pre_step + 1, pre_step):
            train_seq.append(self.data[:, i: i + time_step, :])
            train_label.append(self.data[:, i + time_step: i + time_step + pre_step, :])

        self.features = train_seq
        self.label = train_label

    def generate_train_data(self, time_step, pre_step, batch_size):
        self.process_data(time_step, pre_step)
        dataset = CustomDataset(self.features, self.label)
        data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, drop_last=True)
        return data_loader

    def generate_test_data(self, time_step=3, pre_step=1):
        self.process_data(time_step, pre_step)
        dataset = CustomDataset(self.features, self.label)
        data_loader = DataLoader(dataset=dataset, batch_size=len(self.label), shuffle=False, drop_last=False)
        return data_loader


def get_model(Adj_matrix, data_robbery, mode):
    length = data_robbery.shape[1]
    if mode == 'source':
        train_data, test_data = data_robbery[:, :366], data_robbery[:, 366:]
    else:

        train_data = data_robbery[:, : ] #-365 - int(0.2 * 365)
        test_data = data_robbery[:, -365 + int(0.0 * 365): -365 + int(0.25 * 365)]  #
        print('target', train_data.shape, test_data.shape)

    time_step, out_step, batch_size = 3, 3, 8
    train_dataloader, test_dataloader = MadridDatasetLoader(train_data).generate_train_data(time_step, out_step, batch_size), MadridDatasetLoader(test_data).generate_test_data(time_step, out_step)

    A_wave = get_normalized_adj(Adj_matrix)
    A_q = torch.from_numpy((calculate_random_walk_matrix(A_wave).T).astype('float32'))
    A_h = torch.from_numpy((calculate_random_walk_matrix(A_wave.T).T).astype('float32'))
    A_q = A_q.to(device=device)
    A_h = A_h.to(device=device)
    space_dim = data_robbery.shape[0]
    hidden_dim_s = 70
    hidden_dim_t = 7
    rank_s = 20
    rank_t = 4

    # Initial networks
    TCN1 = B_TCN(space_dim, hidden_dim_t, kernel_size=3).to(device=device)
    TCN2 = B_TCN(hidden_dim_t, rank_t, kernel_size=3, activation='linear').to(device=device)
    TCN3 = B_TCN(rank_t, hidden_dim_t, kernel_size=3).to(device=device)
    # TNB = NBNorm_ZeroInflated(hidden_dim_t, space_dim).to(device=device)
    TNB = NBNorm_MSE(hidden_dim_t, space_dim).to(device=device)
    SCN1 = D_GCN(time_step, hidden_dim_s, 3).to(device=device)
    SCN2 = D_GCN(hidden_dim_s, rank_s, 2, activation='linear').to(device=device)
    SCN3 = D_GCN(rank_s, hidden_dim_s, 2).to(device=device)
    SNB = NBNorm_MSE(hidden_dim_s, out_step).to(device=device)
    STmodel = ST_MSE(SCN1, SCN2, SCN3, TCN1, TCN2, TCN3, SNB, TNB).to(device=device)

    return STmodel, train_dataloader, test_dataloader, A_q, A_h, Adj_matrix


def test_transfer(model, test_dataloader, A_q_target, A_h_target):
    with torch.no_grad():
        model.eval()
        for val_input, val_target in tqdm.tqdm(test_dataloader):
            val_target = val_target.squeeze(-1)
            val_pred = model(val_input, A_q_target, A_h_target, val_input, A_q_target, A_h_target,mode='test')
            val_pred = val_pred.detach().cpu().numpy()

            mae = np.mean(np.abs(val_pred - val_target.detach().cpu().numpy()))
            Rmse = math.sqrt(metrics.mean_squared_error(val_target.detach().cpu().numpy().reshape(-1), val_pred.reshape(-1)))
            final = pd.DataFrame(val_target.detach().cpu().numpy().reshape(-1))
            final1 = pd.DataFrame(val_pred.reshape(-1))
            # 步骤1: 对final1中的小于1的值置为0
            final1[final1 <= 0] = 0
            final1[final1 > 0] = 1
            final[final > 0] = 1

            f1 = metrics.f1_score(final, final1,  average='macro')
            f2 = metrics.f1_score(final, final1,  average='micro')
            recall = metrics.recall_score(final, final1,  average='macro')

            print(f"f1_macro% = {f1:.3}", f"recall% = {recall:.3}")
            return f"{f1:.3%}", f"{f2:.3%}"


def train_transfer(model, train_dataloader, train_dataloader_target, A_q, A_h, A_q_target, A_h_target, test_dataloader):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)  #
    epoch_training_losses, training_nll = [], []
    result = []
    hit, temp_mae = -1, 1000
    final_model = None
    final, final1 = None, None
    criterion_task = nn.MSELoss()
    criterion_domain = nn.CrossEntropyLoss()
    for epoch in range(500):
        # 用 cycle 包装目标域 dataloader，让它自动从头开始
        tgt_iter = cycle(train_dataloader_target)

        # tqdm 只包住源域 dataloader，跑满一个 epoch
        for (x_src, y_src) in tqdm.tqdm(train_dataloader, desc=f"Epoch {epoch}", leave=False):
            x_tgt, y_tgt = next(tgt_iter)  # 每次从目标域迭代一个 batch
            model.train()
            y_src, y_tgt = y_src.squeeze(-1), y_tgt.squeeze(-1)

            out_src, out_tgt, domain_logits, domain_labels = model(x_src, A_q, A_h, x_tgt, A_q_target, A_h_target, mode='train')
            # 任务损失
            loss_src = criterion_task(out_src, y_src)
            loss_tgt = criterion_task(out_tgt, y_tgt)

            # 对抗损失
            loss_domain = criterion_domain(domain_logits, domain_labels)

            total_loss = loss_src + loss_tgt + loss_domain  # 可设置权重
            total_loss.backward()
            optimizer.step()
            epoch_training_losses.append(total_loss.detach().cpu().numpy())
        avg_loss = sum(epoch_training_losses) / len(epoch_training_losses)
        training_nll.append(avg_loss)
        # print(f"Epoch {epoch} Average Total Loss: {avg_loss:.4f}\n")
        f1, f2 = test_transfer(model, test_dataloader, A_q_target, A_h_target)
        result.append([f1, f2])
    return pd.DataFrame(result)


def train():
    device = torch.device("cuda:0")
    data_source = r'/opt/data/shanmx_projects/general_crime/data/CHI_Theft/Crime_CHI_2016_2017.pkl'
    Adj_source = r'/opt/data/shanmx_projects/general_crime/data/CHI_Theft/Adj_CHI_2016_2017.pkl'
    data_target = r'/opt/data/shanmx_projects/general_crime/data/LOS_Theft/Crime_Los Angeles_2016_2017.pkl'
    Adj_target = r'/opt/data/shanmx_projects/general_crime/data/LOS_Theft/Adj_Los Angeles_2016_2017.pkl'


    data_source, Adj_source = pickle.load(open(data_source, "rb")), pickle.load(open(Adj_source, "rb"))
    data_target, Adj_target = pickle.load(open(data_target, "rb")), pickle.load(open(Adj_target, "rb"))
    STmodel, train_dataloader, test_dataloader, A_q, A_h, Adj_matrix = get_model(Adj_source, data_source,
                                                                                 mode='source')
    STmodel_target, train_dataloader_target, test_dataloader_target, A_q_target, A_h_target, Adj_matrix_target = get_model(
        Adj_target, data_target, 'target' )
    ST_Transfer = Transfer_mode(STmodel, STmodel_target).to(device)
    result = train_transfer(ST_Transfer, train_dataloader, train_dataloader_target, A_q, A_h, A_q_target,
                            A_h_target, test_dataloader_target)


train()