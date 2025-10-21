import torch
import torch.nn as nn
import math
from torch.autograd import Function
import torch.nn.functional as F
from torch.nn import GRU, LSTM


class MMDLoss(nn.Module):
    '''
    计算源域数据和目标域数据的MMD距离
    Params:
    source: 源域数据（n * len(x))
    target: 目标域数据（m * len(y))
    kernel_mul:
    kernel_num: 取不同高斯核的数量
    fix_sigma: 不同高斯核的sigma值
    Return:
    loss: MMD loss
    '''

    def __init__(self, kernel_type='rbf', kernel_mul=2.0, kernel_num=5, fix_sigma=None, **kwargs):
        super(MMDLoss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None
        self.kernel_type = kernel_type

    def guassian_kernel(self, source, target, kernel_mul, kernel_num, fix_sigma):
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0 - total1) ** 2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul ** i)
                          for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp)
                      for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def linear_mmd2(self, f_of_X, f_of_Y):
        loss = 0.0
        delta = f_of_X.float().mean(0) - f_of_Y.float().mean(0)
        loss = delta.dot(delta.T)
        return loss

    def forward(self, source, target):
        if self.kernel_type == 'linear':
            return self.linear_mmd2(source, target)
        elif self.kernel_type == 'rbf':
            batch_size = int(source.size()[0])
            kernels = self.guassian_kernel(
                source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
            XX = torch.mean(kernels[:batch_size, :batch_size])
            YY = torch.mean(kernels[batch_size:, batch_size:])
            XY = torch.mean(kernels[:batch_size, batch_size:])
            YX = torch.mean(kernels[batch_size:, :batch_size])
            loss = torch.mean(XX + YY - XY - YX)
            return loss


class ReverseGradFunction(Function):
    @staticmethod
    def forward(ctx, data, alpha):
        ctx.alpha = alpha
        return data

    @staticmethod
    def backward(ctx, grad_outputs):
        grad = None

        if ctx.needs_input_grad[0]:
            grad = -ctx.alpha * grad_outputs

        return grad, None

class ReverseGrad(nn.Module):
    def __init__(self):
        super(ReverseGrad, self).__init__()

    def forward(self, x, alpha):
        return ReverseGradFunction.apply(x, alpha)

class NBNorm_ZeroInflated(nn.Module):
    def __init__(self, c_in, c_out):
        super(NBNorm_ZeroInflated, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.n_conv = nn.Conv2d(in_channels=c_in,
                                out_channels=c_out,
                                kernel_size=(1, 1),
                                bias=True)

        self.p_conv = nn.Conv2d(in_channels=c_in,
                                out_channels=c_out,
                                kernel_size=(1, 1),
                                bias=True)

        self.pi_conv = nn.Conv2d(in_channels=c_in,
                                 out_channels=c_out,
                                 kernel_size=(1, 1),
                                 bias=True)

        self.out_dim = c_out  # output horizon

    def forward(self, x):
        x = x.permute(0, 2, 1, 3)
        (B, _, N, _) = x.shape  # B: batch_size; N: input nodes
        n = self.n_conv(x).squeeze_(-1)
        p = self.p_conv(x).squeeze_(-1)
        pi = self.pi_conv(x).squeeze_(-1)

        # Reshape
        n = n.view([B, self.out_dim, N])
        p = p.view([B, self.out_dim, N])
        pi = pi.view([B, self.out_dim, N])

        # Ensure n is positive and p between 0 and 1
        n = F.softplus(n)  # Some parameters can be tuned here
        p = F.sigmoid(p)
        pi = F.sigmoid(pi)
        return n.permute([0, 2, 1]), p.permute([0, 2, 1]), pi.permute([0, 2, 1])


class D_GCN(nn.Module):
    """
    Neural network block that applies a diffusion graph convolution to sampled location
    """

    def __init__(self, in_channels, out_channels, orders, activation='relu'):
        """
        :param in_channels: Number of time step.
        :param out_channels: Desired number of output features at each node in
        each time step.
        :param order: The diffusion steps.
        """
        super(D_GCN, self).__init__()
        self.orders = orders
        self.activation = activation
        self.num_matrices = 2 * self.orders + 1
        self.Theta1 = nn.Parameter(torch.FloatTensor(in_channels * self.num_matrices,
                                                     out_channels))
        self.bias = nn.Parameter(torch.FloatTensor(out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.Theta1.shape[1])
        self.Theta1.data.uniform_(-stdv, stdv)
        stdv1 = 1. / math.sqrt(self.bias.shape[0])
        self.bias.data.uniform_(-stdv1, stdv1)

    def _concat(self, x, x_):
        x_ = x_.unsqueeze(0)
        return torch.cat([x, x_], dim=0)

    def forward(self, X, A_q, A_h):
        """
        :param X: Input data of shape (batch_size, num_nodes, num_timesteps)
        :A_q: The forward random walk matrix (num_nodes, num_nodes)
        :A_h: The backward random walk matrix (num_nodes, num_nodes)
        :return: Output data of shape (batch_size, num_nodes, num_features)
        """
        batch_size = X.shape[0]  # batch_size
        num_node = X.shape[1]
        input_size = X.size(2)  # time_length
        supports = []
        supports.append(A_q)
        supports.append(A_h)

        x0 = X.permute(1, 2, 0)  # (num_nodes, num_times, batch_size)
        x0 = torch.reshape(x0, shape=[num_node, input_size * batch_size])
        x = torch.unsqueeze(x0, 0)
        for support in supports:
            x1 = torch.mm(support, x0)
            x = self._concat(x, x1)
            for k in range(2, self.orders + 1):
                x2 = 2 * torch.mm(support, x1) - x0
                x = self._concat(x, x2)
                x1, x0 = x2, x1

        x = torch.reshape(x, shape=[self.num_matrices, num_node, input_size, batch_size])
        x = x.permute(3, 1, 2, 0)  # (batch_size, num_nodes, input_size, order)
        x = torch.reshape(x, shape=[batch_size, num_node, input_size * self.num_matrices])
        x = torch.matmul(x, self.Theta1)  # (batch_size * self._num_nodes, output_size)
        x += self.bias
        if self.activation == 'relu':
            x = F.relu(x)
        elif self.activation == 'selu':
            x = F.selu(x)

        return x


## Code of BTCN from Yuankai
class B_TCN(nn.Module):
    """
    Neural network block that applies a bidirectional temporal convolution to each node of
    a graph.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, activation='relu', device='cuda:0'):
        """
        :param in_channels: Number of nodes in the graph.
        :param out_channels: Desired number of output features.
        :param kernel_size: Size of the 1D temporal kernel.
        """
        super(B_TCN, self).__init__()
        # forward dirction temporal convolution
        self.kernel_size = kernel_size
        self.out_channels = out_channels
        self.activation = activation
        self.device = device
        self.conv1 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))
        self.conv2 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))
        self.conv3 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))

        self.conv1b = nn.Conv2d(in_channels, out_channels, (1, kernel_size))
        self.conv2b = nn.Conv2d(in_channels, out_channels, (1, kernel_size))
        self.conv3b = nn.Conv2d(in_channels, out_channels, (1, kernel_size))

    def forward(self, X):
        """
        :param X: Input data of shape (batch_size, num_timesteps, num_nodes)
        :return: Output data of shape (batch_size, num_timesteps, num_features)
        """
        batch_size = X.shape[0]
        seq_len = X.shape[1]
        Xf = X.unsqueeze(1)  # (batch_size, 1, num_timesteps, num_nodes)

        inv_idx = torch.arange(Xf.size(2) - 1, -1, -1).long().to(
            device=self.device)  # .to(device=self.device).to(device=self.device)
        Xb = Xf.index_select(2, inv_idx)  # inverse the direction of time

        Xf = Xf.permute(0, 3, 1, 2)
        Xb = Xb.permute(0, 3, 1, 2)  # (batch_size, num_nodes, 1, num_timesteps)
        tempf = self.conv1(Xf) * torch.sigmoid(self.conv2(Xf))  # +
        outf = tempf + self.conv3(Xf)
        outf = outf.reshape([batch_size, seq_len - self.kernel_size + 1, self.out_channels])

        tempb = self.conv1b(Xb) * torch.sigmoid(self.conv2b(Xb))  # +
        outb = tempb + self.conv3b(Xb)
        outb = outb.reshape([batch_size, seq_len - self.kernel_size + 1, self.out_channels])

        rec = torch.zeros([batch_size, self.kernel_size - 1, self.out_channels]).to(
            device=self.device)  # .to(device=self.device)
        outf = torch.cat((outf, rec), dim=1)
        outb = torch.cat((outb, rec), dim=1)  # (batch_size, num_timesteps, out_features)

        inv_idx = torch.arange(outb.size(1) - 1, -1, -1).long().to(device=self.device)  # .to(device=self.device)
        outb = outb.index_select(1, inv_idx)
        out = outf + outb
        if self.activation == 'relu':
            out = F.relu(outf) + F.relu(outb)
        elif self.activation == 'sigmoid':
            out = F.sigmoid(outf) + F.sigmoid(outb)
        return out


class ST_NB_ZeroInflated(nn.Module):
    """
  wx_t  + wx_s
    |       |
   TC4     SC4
    |       |
   TC3     SC3
    |       |
   z_t     z_s
    |       |
   TC2     SC2
    |       |
   TC1     SC1
    |       |
   x_m     x_m
    """
    def __init__(self, SC1, SC2, SC3, TC1, TC2, TC3, SNB,TNB):
        super(ST_NB_ZeroInflated, self).__init__()
        self.TC1 = TC1
        self.TC2 = TC2
        self.TC3 = TC3
        self.TNB = TNB

        self.SC1 = SC1
        self.SC2 = SC2
        self.SC3 = SC3
        self.SNB = SNB


    def forward(self, X, A_q, A_h):
        """
        :param X: Input data of shape (batch_size, num_timesteps, num_nodes)
        :A_hat: The Laplacian matrix (num_nodes, num_nodes)
        :return: Reconstructed X of shape (batch_size, num_timesteps, num_nodes)
        """
        X = X[:,:,:, 0]#.to(device='cuda') # Dummy dimension deleted
        X_T = X.permute(0,2,1)
        X_t1 = self.TC1(X_T)
        X_t2 = self.TC2(X_t1) #num_time, rank
        self.temporal_factors = X_t2
        X_t3 = self.TC3(X_t2)
        _b,_h,_ht = X_t3.shape
        n_t_nb,p_t_nb,pi_t_nb = self.TNB(X_t3.view(_b,_h,_ht,1))

        X_s1 = self.SC1(X, A_q, A_h)
        X_s2 = self.SC2(X_s1, A_q, A_h) #num_nodes, rank
        self.space_factors = X_s2
        X_s3 = self.SC3(X_s2, A_q, A_h)
        _b,_n,_hs = X_s3.shape
        n_s_nb,p_s_nb,pi_s_nb = self.SNB(X_s3.view(_b,_n,_hs,1))
        n_res = n_t_nb.permute(0, 2, 1) * n_s_nb
        p_res = p_t_nb.permute(0, 2, 1) * p_s_nb
        pi_res = pi_t_nb.permute(0, 2, 1) * pi_s_nb

        return n_res,p_res,pi_res


class NBNorm_MSE(nn.Module):
    def __init__(self, c_in, c_out):
        super(NBNorm_MSE, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.n_conv = nn.Conv2d(in_channels=c_in,
                                out_channels=c_out,
                                kernel_size=(1, 1),
                                bias=True)

        self.p_conv = nn.Conv2d(in_channels=c_in,
                                out_channels=c_out,
                                kernel_size=(1, 1),
                                bias=True)

        self.pi_conv = nn.Conv2d(in_channels=c_in,
                                 out_channels=c_out,
                                 kernel_size=(1, 1),
                                 bias=True)

        self.out_dim = c_out  # output horizon

    def forward(self, x):
        x = x.permute(0, 2, 1, 3)
        (B, _, N, _) = x.shape  # B: batch_size; N: input nodes
        n = self.n_conv(x).squeeze_(-1)

        # Reshape
        n = n.view([B, self.out_dim, N])

        # Ensure n is positive and p between 0 and 1
        n = F.relu(n)  # Some parameters can be tuned here

        return n.permute([0, 2, 1])


class ST_MSE(nn.Module):
    """
  wx_t  + wx_s
    |       |
   TC4     SC4
    |       |
   TC3     SC3
    |       |
   z_t     z_s
    |       |
   TC2     SC2
    |       |
   TC1     SC1
    |       |
   x_m     x_m
    """
    def __init__(self, SC1, SC2, SC3, TC1, TC2, TC3, SNB,TNB):
        super(ST_MSE, self).__init__()
        self.TC1 = TC1
        self.TC2 = TC2
        self.TC3 = TC3
        self.TNB = TNB

        self.SC1 = SC1
        self.SC2 = SC2
        self.SC3 = SC3
        self.SNB = SNB

    def forward(self, X, A_q, A_h):
        """
        :param X: Input data of shape (batch_size, num_timesteps, num_nodes)
        :A_hat: The Laplacian matrix (num_nodes, num_nodes)
        :return: Reconstructed X of shape (batch_size, num_timesteps, num_nodes)
        """
        X = X[:,:,:,0]#.to(device='cuda') # Dummy dimension deleted
        X_T = X.permute(0,2,1)
        X_t1 = self.TC1(X_T)
        X_t2 = self.TC2(X_t1) #num_time, rank
        self.temporal_factors = X_t2
        X_t3 = self.TC3(X_t2)
        _b,_h,_ht = X_t3.shape
        n_t_nb = self.TNB(X_t3.view(_b,_h,_ht,1))

        X_s1 = self.SC1(X, A_q, A_h)
        X_s2 = self.SC2(X_s1, A_q, A_h) #num_nodes, rank
        self.space_factors = X_s2
        X_s3 = self.SC3(X_s2, A_q, A_h)
        _b,_n,_hs = X_s3.shape
        n_s_nb = self.SNB(X_s3.view(_b,_n,_hs,1))
        n_res = n_t_nb.permute(0, 2, 1) * n_s_nb
        return n_res, X_t3


class DomainDiscriminator(nn.Module):
    def __init__(self, in_dim=3*7, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),                          # [B, 7, 7] → [B, 49]
            nn.Linear(in_dim, hidden_dim),         # [B, 49] → [B, 64]
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)               # [B, 64] → [B, 2]
        )

    def forward(self, x):
        return self.net(x)


class Transfer_mode(nn.Module):
    """
  wx_t  + wx_s
    |       |
   TC4     SC4
    |       |
   TC3     SC3
    |       |
   z_t     z_s
    |       |
   TC2     SC2
    |       |
   TC1     SC1
    |       |
   x_m     x_m
    """
    def __init__(self, SC1, TC1):
        super(Transfer_mode, self).__init__()
        self.model_source = SC1
        self.model_target = TC1
        self.domain_classifier = DomainDiscriminator()

    def forward(self, X, A_q, A_h, X_target, A_q_target, A_h_target, mode='train'):
        """
        :param X: Input data of shape (batch_size, num_timesteps, num_nodes)
        :A_hat: The Laplacian matrix (num_nodes, num_nodes)
        :return: Reconstructed X of shape (batch_size, num_timesteps, num_nodes)
        """
        if mode == 'train':
            out_source, T_source = self.model_source(X, A_q, A_h)
            out_target, T_target = self.model_target(X_target, A_q_target, A_h_target)

            # 合并特征并加标签（0=src, 1=tgt）
            features = torch.cat([T_source, T_target], dim=0)
            domain_labels = torch.cat([
                torch.zeros(T_source.size(0)).long(),
                torch.ones(T_target.size(0)).long()
            ]).to(features.device)
            reverse_feat = ReverseGrad()(features, 0.01)

            domain_logits = self.domain_classifier(reverse_feat)

            return out_source, out_target, domain_logits, domain_labels

        elif mode == 'test':
            out_target, T_target = self.model_target(X_target, A_q_target, A_h_target)
            return out_target


class Transfer_mode_mmd(nn.Module):
    """
  wx_t  + wx_s
    |       |
   TC4     SC4
    |       |
   TC3     SC3
    |       |
   z_t     z_s
    |       |
   TC2     SC2
    |       |
   TC1     SC1
    |       |
   x_m     x_m
    """
    def __init__(self, SC1, TC1):
        super(Transfer_mode_mmd, self).__init__()
        self.model_source = SC1
        self.model_target = TC1
        self.domain_classifier = DomainDiscriminator()

    def forward(self, X, A_q, A_h, X_target, A_q_target, A_h_target, mode='train'):
        """
        :param X: Input data of shape (batch_size, num_timesteps, num_nodes)
        :A_hat: The Laplacian matrix (num_nodes, num_nodes)
        :return: Reconstructed X of shape (batch_size, num_timesteps, num_nodes)
        """
        if mode == 'train':
            out_source, T_source = self.model_source(X, A_q, A_h)
            out_target, T_target = self.model_target(X_target, A_q_target, A_h_target)
            return out_source, out_target, T_source.view(T_source.size(0), -1), T_target.view(T_source.size(0), -1)

        elif mode == 'test':
            out_target, T_target = self.model_target(X_target, A_q_target, A_h_target)
            return out_target


class Transfer_GCN(nn.Module):
    """
  wx_t  + wx_s
    |       |
   TC4     SC4
    |       |
   TC3     SC3
    |       |
   z_t     z_s
    |       |
   TC2     SC2
    |       |
   TC1     SC1
    |       |
   x_m     x_m
    """
    def __init__(self, SC1, TC1, max_grid):
        super(Transfer_GCN, self).__init__()
        self.model_source = SC1
        self.model_target = TC1
        self.domain_classifier = DomainDiscriminator(max_grid * 128)

    def forward(self, X, A_q, X_target, A_q_target, mode='train'):
        """
        :param X: Input data of shape (batch_size, num_timesteps, num_nodes)
        :A_hat: The Laplacian matrix (num_nodes, num_nodes)
        :return: Reconstructed X of shape (batch_size, num_timesteps, num_nodes)
        """
        if mode == 'train':
            out_source, T_source = self.model_source(X, A_q)
            out_target, T_target = self.model_target(X_target, A_q_target)
            T_source, T_target = T_source.unsqueeze(0), T_target.unsqueeze(0)
            # 合并特征并加标签（0=src, 1=tgt）
            features = torch.cat([T_source, T_target], dim=0)
            domain_labels = torch.cat([
                torch.zeros(T_source.size(0)).float(),
                torch.ones(T_target.size(0)).float()
            ]).to(features.device)
            reverse_feat = ReverseGrad()(features, 0.01)
            domain_logits = self.domain_classifier(reverse_feat)
            return out_source, out_target, domain_logits, domain_labels

        elif mode == 'test':
            out_target, T_target = self.model_target(X_target, A_q_target)
            return out_target
