import torch.nn as nn
import torch
import torch.nn.functional as F


class GATLayer(nn.Module):
    """GAT层"""

    def __init__(self, input_feature, output_feature, dropout, alpha, concat=True):
        super(GATLayer, self).__init__()
        self.input_feature = input_feature
        self.output_feature = output_feature
        self.alpha = alpha
        self.dropout = dropout
        self.concat = concat
        self.a = nn.Parameter(torch.empty(size=(2 * output_feature, 1)))
        self.w = nn.Parameter(torch.empty(size=(input_feature, output_feature)))
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.w.data, gain=1.414)
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

    def forward(self, h, adj):
        Wh = torch.mm(h, self.w)
        e = self._prepare_attentional_mechanism_input(Wh)
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)  # adj>0的位置使用e对应位置的值替换，其余都为-9e15，这样设定经过Softmax后每个节点对应的行非邻居都会变为0。
        attention = F.softmax(attention, dim=1)  # 每行做Softmax，相当于每个节点做softmax
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.mm(attention, Wh)  # 得到下一层的输入

        if self.concat:
            return F.elu(h_prime)  # 激活
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):

        Wh1 = torch.matmul(Wh, self.a[:self.output_feature, :])  # N*out_size @ out_size*1 = N*1

        Wh2 = torch.matmul(Wh, self.a[self.output_feature:, :])  # N*1

        e = Wh1 + Wh2.T  # Wh1的每个原始与Wh2的所有元素相加，生成N*N的矩阵
        return self.leakyrelu(e)


class GAT(nn.Module):
    """GAT模型"""


    def __init__(self, input_size, hidden_size, output_size, dropout, alpha, nheads, concat,device):
        super(GAT, self).__init__()
        self.dropout = dropout
        self.attention = [GATLayer(input_size, hidden_size, dropout=dropout, alpha=alpha, concat=True).to(device) for _ in
                          range(nheads)]
        for i, attention in enumerate(self.attention):
            self.add_module('attention_{}'.format(i), attention)
        self.gat_linear=GATLayer(input_size, hidden_size, dropout=dropout, alpha=alpha, concat=True).to(device)
        self.out_att = GATLayer(hidden_size * nheads, output_size, dropout=dropout, alpha=alpha, concat=True).to(device)
        self.linear=nn.Linear(input_size,hidden_size)
        self.out_linear=nn.Linear(hidden_size,hidden_size)

    def forward(self, x, adj):
        x_residual=self.linear(x)
        # x_residual=x
        # x=self.gat_linear(x,adj)

        # x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attention], dim=1)

        x = F.dropout(x, self.dropout, training=self.training)
        # x = F.elu(self.out_att(x, adj))
        x = (self.out_att(x, adj))

        # x=self.out_linear(torch.cat([x,x_residual],dim=-1))
        # x=self.out_linear((x+x_residual))
        x=self.out_linear(x_residual+x)
        # x=self.out_linear((x))

        # x=(x+x_residual)
        x=F.elu(x)
        return x

