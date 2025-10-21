
import os
import sys
file_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(file_dir)
sys.path.append(file_dir)

import torch
import numpy as np
import torch.nn as nn
import argparse
import configparser
from datetime import datetime
from Trainer import Trainer
from UniCrimePT import UniCrimePT as Network_Pretrain
from UniCrimePT import UniCrimePT as Network_Predict
from lib.TrainInits import init_seed
from lib.TrainInits import print_model_parameters
from lib.metrics import MAE_torch, MSE_torch, huber_loss
from lib.predifineGraph import *
from lib.data_process import define_dataloder, get_val_tst_dataloader, data_type_init, define_dataloder_test
from conf.FlashST.Params_pretrain import parse_args
import torch.nn.functional as F

# 2km few los 0.01 1e-5 0.01 100 phi  0.001 1e-4 0.01 100 NEW 0.01 1e-4 0.01 100 NYC 0.01 1e-3 0.03 100
# 2km full NYC 0.01 1e-3 0.03 30 NEW 0.01 1.0e-3 0.05 30 PHI 0.01 1e-4 0.03 30 LOS 0.01 1.0e-5 0.05 30
# 1km full LOS 0.001 1.0e-5 0.01 20 （-365） PHI 0.001 1e-3 0.01 30 NEW  0.01 1e-5 0.2 100 NYC 0.001 1e-4 0.01 50
# 1km few NYC 0.001 1e-4 0.01 200 NEW 0.01， 1e-6 0.1 200 PHI 0.01， 1e-5, 0.05 100  LOS 0.001 1.0e-4 0.05 50

weight = 0.001
eps = 1.0e-3
# ************************************************************************* #
 #1km时候全样本batch是20， 小样本是200  0.001  1.0e-4 NYC
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
args = parse_args(device)
args.mode = 'eval'  # 可选：pretrain, eval, ori, test
args.dataset_test = 'LOS_Theft'  # 可选：CHI_Theft, NEW_Theft, NYC_Theft, PHI_Theft LOS_Theft
args.dataset_use = ['CHI_Theft']  # 可选：'NYC_Theft', 'CHI_Theft'
args.model = 'STGCN'  # 可选模型列表
###改下面的  save ori 或者 pretrain
# args.load_pretrain_path = 'STGCN_NEW_Theft.pth_ep87_EL0.64_PL0.6.pth'
# args.load_pretrain_path = 'STGCN_NYC_Theft.pth_ep87_EL0.83_PL0.83.pth'
args.load_pretrain_path = 'STGCN_NYC_Theft.pth_ep0_EL1.14_PL1.14.pth'  # zero shot 1km
# args.load_pretrain_path = 'STGCN_NYC_Theft.pth_ep6_EL0.99_PL0.99.pth'  # zero shot 1km
# args.load_pretrain_path = 'STGCN_NYC_Theft.pth_ep30_EL0.65_PL0.61.pth'  # zero shot 2km
args.save_pretrain_path = 'PDF_NYC_Theft.pth'
print('Mode: ', args.mode, '  model: ', args.model, '  DATASET: ', args.dataset_test,
      '  load_pretrain_path: ', args.load_pretrain_path, '  save_pretrain_path: ', args.save_pretrain_path)


def Mkdir(path):
    if os.path.isdir(path):
        pass
    else:
        os.makedirs(path)


# 计算Cosine相似度
def nb():
    def nb_zeroinflated_nll_loss(out, y, scaler):
        """
        y: true values
        y_mask: whether missing mask is given
        https://stats.idre.ucla.edu/r/dae/zinb/
        """
        if scaler:
            out = scaler.inverse_transform(out)
            y = scaler.inverse_transform(y)
        n = out[..., 0]  # 负二项分布的 n
        p = out[..., 1]  # 负二项分布的 p
        pi = out[..., 2]  # 零膨胀的概率
        y = y[..., 0]

        idx_yeq0 = y == 0
        idx_yg0 = y > 0

        n_yeq0 = n[idx_yeq0]
        p_yeq0 = p[idx_yeq0]
        pi_yeq0 = pi[idx_yeq0]
        yeq0 = y[idx_yeq0]

        n_yg0 = n[idx_yg0]
        p_yg0 = p[idx_yg0]
        pi_yg0 = pi[idx_yg0]
        yg0 = y[idx_yg0]
        epsilon = 1e-8
        L_yeq0 = torch.log(pi_yeq0 + epsilon) + torch.log((1 - pi_yeq0 + epsilon) * torch.pow(p_yeq0, n_yeq0))
        L_yg0 = torch.log(1 - pi_yg0 + epsilon) + torch.lgamma(n_yg0 + yg0) - torch.lgamma(yg0 + 1) - torch.lgamma(
            n_yg0) + \
                n_yg0 * torch.log(p_yg0 + epsilon) + yg0 * torch.log(1 - p_yg0 + epsilon)

        return -torch.sum(L_yeq0) - torch.sum(L_yg0), -torch.sum(L_yeq0) - torch.sum(L_yg0)
    return nb_zeroinflated_nll_loss


# def infoNCEloss():
#     def loss(q, k):
#         T = 0.3
#         pos_sim = torch.sum(torch.mul(q, q), dim=-1)
#         neg_sim = torch.matmul(q, q.transpose(-1, -2))
#         pos = torch.exp(torch.div(pos_sim, T))
#         neg = torch.sum(torch.exp(torch.div(neg_sim, T)), dim=-1)
#         denominator = neg + pos
#         return torch.mean(-torch.log(torch.div(pos, denominator)))
#     return loss


def infoNCEloss(mu=0.0, sigma=1.0, T=0.3, eps=1e-8):
    """
    对比学习损失，正样本为 (q, 高斯采样)，负样本为 (q, 其他样本)
    q: (B, ...)  可以是2D/3D/4D，最终展平成 (B, D)
    """
    def loss(q, k):
        B = q.shape[0]
        q_flat = q.reshape(B, -1)  # (B, D)

        # 归一化，避免数值爆炸
        q_flat = F.normalize(q_flat, dim=-1)

        # 从 N(mu, sigma^2 I) 抽样作为正样本
        z = torch.randn_like(q_flat) * sigma + mu
        z = F.normalize(z, dim=-1)

        # 计算正样本相似度 (q, z)
        pos_sim = torch.sum(q_flat * z, dim=-1)  # (B,)
        pos = torch.exp(pos_sim / T)

        # 负样本相似度 (q, q')
        neg_sim = torch.matmul(q_flat, q_flat.T)  # (B, B)
        neg_sim = neg_sim - torch.diag_embed(torch.diag(neg_sim))  # 去掉自己
        neg = torch.sum(torch.exp(neg_sim / T), dim=-1)  # (B,)

        # denominator
        denominator = pos + neg + eps
        loss_val = torch.mean(-torch.log(pos / denominator))
        return loss_val
    return loss


def scaler_mae_loss(mask_value):
    def loss(preds, labels, scaler, mask=None):
        if scaler:
            preds = scaler.inverse_transform(preds)
            labels = scaler.inverse_transform(labels)
        mae, mae_loss = MAE_torch(pred=preds, true=labels, mask_value=mask_value)
        # print(mae.shape, mae_loss.shape)
        return mae, mae_loss
    return loss

def scaler_huber_loss(mask_value):
    def loss(preds, labels, scaler, mask=None):
        if scaler:
            preds = scaler.inverse_transform(preds)
            labels = scaler.inverse_transform(labels)
        mae, mae_loss = huber_loss(pred=preds, true=labels, mask_value=mask_value)
        # print(mae.shape, mae_loss.shape)
        return mae, mae_loss
    return loss

def sp_class(mask_value):
    def loss(preds, labels, scaler ):
        if scaler:
            preds = scaler.inverse_transform(preds)
            labels = scaler.inverse_transform(labels)
            labels = (labels >= 1).float()
        # 正负样本统计
        pos_weight = (labels == 0).sum() / (labels == 1).sum()
        pos_weight = torch.tensor(pos_weight).to(preds.device)

        # 加入 pos_weight
        c_loss = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight).to(preds.device)
        beeloss = c_loss(preds, labels)
        # print(mae.shape, mae_loss.shape)
        return beeloss, beeloss
    return loss


if args.model == 'GWN' or args.model == 'MTGNN' or args.model == 'STFGNN' or args.model == 'STGODE' or args.model == 'DMSTGCN':
    seed_mode = False   # for quick running
else:
    seed_mode = True

init_seed(args.seed, seed_mode)

# config log path
current_dir = os.path.dirname(os.path.realpath(__file__))
log_dir = os.path.join(current_dir, '../SAVE', args.mode, args.model)
Mkdir(log_dir)
args.log_dir = log_dir

#predefine Graph
dataset_graph = []
if args.mode == 'pretrain':
    dataset_graph = args.dataset_use.copy()
else:
    dataset_graph.append(args.dataset_test)
args.dataset_graph = dataset_graph
pre_graph_dict(args)
data_type_init(args.dataset_test, args)

if args.model == 'STGODE' or args.model == 'AGCRN' or args.model == 'ASTGCN':
    xavier = True
else:
    xavier = False

args.xavier = xavier

#load dataset
if args.mode == 'pretrain':
    x_trn_dict, y_trn_dict, _, _, _, _, scaler_dict = define_dataloder(stage='Train', args=args, val_ratio=0.7, test_ratio=0.1)
    eval_train_loader, eval_val_loader, eval_test_loader, eval_scaler_dict = None, None, None, None
    eval_test_loader1 = None
    eval_test_loader2 = None
    eval_test_loader3 = None
    eval_test_loader4 = None
elif args.mode == 'test':
    x_trn_dict, y_trn_dict, scaler_dict = None, None, None
    eval_x_tst_dict1, eval_y_tst_dict1, eval_scaler_dict = define_dataloder_test(
        stage='eval', val_ratio=0, test_ratio=0.25, args=args)
    eval_x_tst_dict2, eval_y_tst_dict2, _ = define_dataloder_test(stage='eval', val_ratio=0.25, test_ratio=0.5,
                                                                         args=args)
    eval_x_tst_dict3, eval_y_tst_dict3, _ = define_dataloder_test(stage='eval', val_ratio=0.5, test_ratio=0.75,
                                                                         args=args)
    eval_x_tst_dict4, eval_y_tst_dict4, _ = define_dataloder_test(stage='eval', val_ratio=0.75, test_ratio=1,
                                                                         args=args)
    eval_train_loader = get_val_tst_dataloader(eval_x_tst_dict1, eval_y_tst_dict1, args, shuffle=True)
    eval_val_loader = get_val_tst_dataloader(eval_x_tst_dict1, eval_y_tst_dict1, args, shuffle=False)
    eval_test_loader1 = get_val_tst_dataloader(eval_x_tst_dict1, eval_y_tst_dict1, args, shuffle=False)
    eval_test_loader2 = get_val_tst_dataloader(eval_x_tst_dict2, eval_y_tst_dict2, args, shuffle=False)
    eval_test_loader3 = get_val_tst_dataloader(eval_x_tst_dict3, eval_y_tst_dict3, args, shuffle=False)
    eval_test_loader4 = get_val_tst_dataloader(eval_x_tst_dict4, eval_y_tst_dict4, args, shuffle=False)
else:
    x_trn_dict, y_trn_dict, scaler_dict = None, None, None
    eval_x_trn_dict, eval_y_trn_dict, eval_x_val_dict, eval_y_val_dict, eval_x_tst_dict1, eval_y_tst_dict1, eval_scaler_dict = define_dataloder(stage='eval', val_ratio=0, test_ratio=0.25, args=args)
    _, _, _, _, eval_x_tst_dict2, eval_y_tst_dict2, _ = define_dataloder(stage='eval', val_ratio=0.25, test_ratio=0.5, args=args)
    _, _, _, _, eval_x_tst_dict3, eval_y_tst_dict3, _  = define_dataloder(stage='eval', val_ratio=0.5, test_ratio=0.75, args=args)
    _, _, _, _, eval_x_tst_dict4, eval_y_tst_dict4, _  = define_dataloder(stage='eval', val_ratio=0.75, test_ratio=1, args=args)
    eval_train_loader = get_val_tst_dataloader(eval_x_trn_dict, eval_y_trn_dict, args, shuffle=True)
    eval_val_loader = get_val_tst_dataloader(eval_x_val_dict, eval_y_val_dict, args, shuffle=False)
    eval_test_loader1 = get_val_tst_dataloader(eval_x_tst_dict1, eval_y_tst_dict1, args, shuffle=False)
    eval_test_loader2 = get_val_tst_dataloader(eval_x_tst_dict2, eval_y_tst_dict2, args, shuffle=False)
    eval_test_loader3 = get_val_tst_dataloader(eval_x_tst_dict3, eval_y_tst_dict3, args, shuffle=False)
    eval_test_loader4 = get_val_tst_dataloader(eval_x_tst_dict4, eval_y_tst_dict4, args, shuffle=False)


#init model
if args.mode == 'pretrain':
    model = Network_Pretrain(args)
    # if torch.cuda.device_count() > 1:
    #     model = nn.DataParallel(model)
    model = model.to(args.device)

else:
    model = Network_Predict(args)
    # if torch.cuda.device_count() > 1:
    #     model = nn.DataParallel(model)
    model = model.to(args.device)
    if args.mode == 'eval':
        load_dir = os.path.join(current_dir, '../SAVE', 'pretrain', args.model)
        # load_dir = os.path.join(current_dir, '../SAVE', 'ori', args.model)
        # 加载预训练参数
        model_dict = model.state_dict()
        # 更新模型参数
        model.load_state_dict(model_dict, strict=False)
        print("✅ Pretrain model loaded with flexible feature dim!")
        model.load_state_dict(torch.load(load_dir + '/' + args.load_pretrain_path), strict=False)
        print(load_dir + '/' + args.load_pretrain_path)
        print('load pretrain model!!!')
    elif args.mode == 'test':

        # load_dir = os.path.join(current_dir, '../SAVE', 'pretrain', args.model)
        load_dir = os.path.join(current_dir, '../SAVE', 'ori', args.model)
        model_dict = model.state_dict()
        model.load_state_dict(torch.load(load_dir + '/' + args.load_pretrain_path), strict=False)
        print(load_dir + '/' + args.load_pretrain_path)
        print('load pretrain model!!!')


print_model_parameters(model, only_num=False)
print(args.loss_func)
#init loss function, optimizer
if args.loss_func == 'mask_mae':
    if (args.model == 'STSGCN' or args.model == 'STFGNN' or args.model == 'STGODE'):
        loss = scaler_huber_loss(mask_value=args.mape_thresh)
        print('============================scaler_huber_loss')
    else:
        loss = scaler_mae_loss(mask_value=args.mape_thresh)
        print('============================scaler_mae_loss')
    # print(args.model, Mode)
elif args.loss_func == 'mae':
    loss = torch.nn.L1Loss().to(args.device)
elif args.loss_func == 'mse':
    loss = torch.nn.MSELoss().to(args.device)
elif args.loss_func == 'nb':
    loss = nb()
elif args.loss_func == 'class':
    loss = sp_class(mask_value=args.mape_thresh)
else:
    raise ValueError


optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr_init, eps=eps,
                             weight_decay=weight, amsgrad=False)
#learning rate decay
lr_scheduler = None
if args.lr_decay:
    print('Applying learning rate decay.')
    lr_decay_steps = [int(i) for i in list(args.lr_decay_step.split(','))]
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer,
                                                        milestones=lr_decay_steps,
                                                        gamma=args.lr_decay_rate)

#start training
loss_mse = torch.nn.MSELoss().to(args.device)
loss_ssl = infoNCEloss()
trainer = Trainer(model, loss, loss_ssl, optimizer, x_trn_dict, y_trn_dict, args.A_dict, args.lpls_dict, eval_train_loader,
                       eval_val_loader, eval_test_loader1, eval_test_loader2, eval_test_loader3, eval_test_loader4, scaler_dict, eval_scaler_dict, args,
                       lr_scheduler=lr_scheduler)  #
# trainer = Trainer(model, loss, loss_ssl, optimizer, x_trn_dict, y_trn_dict, args.A_dict, args.lpls_dict, eval_train_loader,
#                        eval_val_loader, eval_test_loader, scaler_dict, eval_scaler_dict, args,
#                        lr_scheduler=lr_scheduler)  #

if args.mode == 'pretrain':
    trainer.train_pretrain()
elif args.mode == 'eval':
    trainer.train_eval()
elif args.mode == 'ori':
    trainer.train_eval()
elif args.mode == 'test':
    # model.load_state_dict(torch.load(log_dir + '/' + args.load_pretrain_path), strict=True)
    # print("Load saved model")
    trainer.eval_test(model, trainer.args, args.A_dict, args.lpls_dict, eval_test_loader1, eval_scaler_dict[args.dataset_test], trainer.logger)
    trainer.eval_test(model, trainer.args, args.A_dict, args.lpls_dict, eval_test_loader2, eval_scaler_dict[args.dataset_test], trainer.logger)
    trainer.eval_test(model, trainer.args, args.A_dict, args.lpls_dict, eval_test_loader3, eval_scaler_dict[args.dataset_test], trainer.logger)
    trainer.eval_test(model, trainer.args, args.A_dict, args.lpls_dict, eval_test_loader4, eval_scaler_dict[args.dataset_test], trainer.logger)
else:
    raise ValueError
