import torch
from torch import nn
import numpy as np


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask,d_k=64):                      # Q: [batch_size, n_heads, len_q, d_k] 551
                                                                       # K: [batch_size, n_heads, len_k, d_k] 5
                                                                       # V: [batch_size, n_heads, len_v(=len_k), d_v]
                                                                       # attn_mask: [batch_size, n_heads, seq_len, seq_len]
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)   # scores : [batch_size, n_heads, len_q, len_k]  551 5   scores.masked_fill_(attn_mask, -1e9)                           # 如果时停用词P就等于 0 
        attn = nn.Softmax(dim=-1)(scores) # lhy: 这个分数有很大问题       # 此处是注意力是 服务节点对于样本的注意力
                                                                       # 之后乘上样本表示，得到服务节点的表示。因此，如果进来的每一批样本都是一样的，那么基本没啥意义。所以，应该喂进来不同类别的样本。
        context = torch.matmul(attn, V)                                # [batch_size, n_heads, len_q, d_v] 551 d_v
        
        return context, attn

class MultiHeadAttention(torch.nn.Module):
    def __init__(self,d_model=512,n_heads=8,d_k='none',d_v='none',device="cpu"):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.device = device
        self.d_k = d_k if d_k !='none' else int(d_model/n_heads)
        self.d_v = d_v if d_v !='none' else int(d_model/n_heads)
        self.W_Q = nn.Linear(d_model, self.d_k * self.n_heads, bias=False)
        self.W_K = nn.Linear(d_model, self.d_k * self.n_heads, bias=False)
        self.W_V = nn.Linear(d_model, self.d_v * self.n_heads, bias=False)
        
    def forward(self, query, context, attn_mask='none'):    
        input_Q = query
        input_K = context
        input_V = context
                                                                # input_Q: [batch_size, len_q, d_model]
                                                                # input_K: [batch_size, len_k, d_model]
                                                                # input_V: [batch_size, len_v(=len_k), d_model]
                                                                # attn_mask: [batch_size, seq_len, seq_len]
        residual, batch_size = input_Q, input_Q.size(0)
        Q = nn.functional.relu(self.W_Q(input_Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1,2))  # Q: [batch_size, n_heads, len_q, d_k]
        K = self.W_K(input_K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1,2)  # K: [batch_size, n_heads, len_k, d_k]
        V = self.W_V(input_V).view(batch_size, -1, self.n_heads, self.d_v).transpose(1,2)  # V: [batch_size, n_heads, len_v(=len_k), d_v]
        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1)  if attn_mask!='none' else torch.zeros(batch_size,self.n_heads,Q.size(2),K.size(2)).bool().to(self.device)           # attn_mask : [batch_size, n_heads, seq_len, seq_len]
        context, attn = ScaledDotProductAttention()(Q, K, V, attn_mask,d_k=self.d_k)          # context: [batch_size, n_heads, len_q, d_v]
                                                                                 # attn: [batch_size, n_heads, len_q, len_k]
        context = context.transpose(1, 2).reshape(batch_size, -1, self.n_heads * self.d_v) # context: [batch_size, len_v, n_heads * d_v]
        # output = self.fc(context)                                                # [batch_size, len_v, d_model]
        output = context
        return nn.LayerNorm(self.d_model).cuda()(output.reshape(batch_size,-1) + residual), attn

class RCA(torch.nn.Module):
    def __init__(self, a, embed_dim,adj, hidden_dim, device, target_class, *args, **kwargs):
        super(RCA, self).__init__()
        self.a = a
        self.adj=adj
        self.embed_dim=embed_dim
        self.hidden_dim=hidden_dim
        self.device=device
        self.target_class = target_class
        self.kwargs = kwargs
        self.linear2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = torch.nn.Linear(embed_dim, hidden_dim)
        if self.kwargs["gat_open"]:
            self.linear9 = torch.nn.Linear(self.kwargs["gat_hidden_size"], hidden_dim)
        else:
            self.linear9 = torch.nn.Linear(embed_dim, hidden_dim)
        self.linear8 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.linear4 = torch.nn.Linear(target_class, target_class)
        self.linear5 = torch.nn.Linear(hidden_dim, hidden_dim)
        

        # self.cross_attn = MultiHeadAttention(d_model=embed_dim,device=device).to(device)
        self.mse_loss = torch.nn.MSELoss()
        self.l1_loss=torch.nn.L1Loss()
        # self.gated_tanh = nn.Tanh()
        # self.gated_relu = nn.ReLU()
        self.sigmod = nn.Sigmoid()
        # self.linear6=torch.nn.Linear(2*embed_dim,embed_dim)
        
    def dynamic_slot(self, prob_static, quey, memory_slots_keys, memory_slots_values):
        if memory_slots_keys is not None:  # query [batch_size,node_nums]       slot_keys [slots_num,node_nums]
            key = self.linear2(self.linear3(memory_slots_keys[:self.kwargs["memory_k"]])) 
            sim = torch.matmul(quey, key.T) 
            output = torch.matmul(sim, key)
            output = self.linear5(output)
            return output
        else:
            return self.linear5(quey)
            
    def dynamic_slot_old(self, prob_static, quey, memory_slots_keys, memory_slots_values):
        if memory_slots_keys is not None:  # query [batch_size,node_nums]       slot_keys [slots_num,node_nums]
            key = self.linear2(self.linear3(memory_slots_keys[:self.kwargs["memory_k"]])) 
            sim = torch.matmul(quey, key.T) 
            output = torch.matmul(sim, memory_slots_values[:self.kwargs["memory_k"]]) 
            output = self.linear4(output)
            output = torch.nn.functional.softmax(output,dim=-1)
            return output
        else:
            return prob_static
        
    def dynamic_slot_history(self, nodes,prob_static, quey, memory_slots_keys, memory_slots_values):
        if memory_slots_keys is not None:  # query [batch_size,node_nums]       slot_keys [slots_num,node_nums]
            key = self.linear2(self.linear3(memory_slots_keys[:self.kwargs["memory_k"]])) 
            sim = torch.matmul(key.reshape(-1, self.hidden_dim), nodes.permute(1, 0))
            prob_dyn = torch.nn.functional.softmax(sim,dim=-1)
            return prob_dyn
        else:
            return prob_static

    def inference(self,query_orl,label, nodes,is_train,memory_slots_keys,memory_slots_values):
        b=query_orl.shape[0]
        query_l = self.linear2(self.linear3(query_orl))
        nodes = self.linear8(self.linear9(nodes))
        sim= torch.matmul(query_l.reshape(b, self.hidden_dim), nodes.permute(1, 0))
        prob_static = torch.nn.functional.softmax(sim,dim=-1)
        
        # with man v = k
        if self.kwargs["memory_open"]:
            if self.kwargs["memory_type"] == "query":
                query_new = self.dynamic_slot(prob_static, query_l, memory_slots_keys, memory_slots_values)
                sim2 = torch.matmul(query_new.reshape(b, self.hidden_dim), nodes.permute(1, 0))
                prob_dyn = torch.nn.functional.softmax(sim2,dim=-1)
                prob = prob_static + prob_dyn * self.kwargs["memory_weight_c"]
            elif self.kwargs["memory_type"] == "prediction":
                prob_dyn= self.dynamic_slot_old(prob_static, query_l, memory_slots_keys, memory_slots_values)
                prob = prob_static + prob_dyn * self.kwargs["memory_weight_c"]
            elif self.kwargs["memory_type"] == "sample":
                prob_dyn= self.dynamic_slot_history(nodes,prob_static, query_l, memory_slots_keys, memory_slots_values)
                prob = prob_static 
        else:
            prob = prob_static
        
        # prob = self.sigmod(prob)
        # prob = torch.nn.functional.softmax(prob,dim=-1)
        # prob = prob_static
        if is_train:
            loss = - torch.sum(torch.log(prob) * label)
            if self.kwargs["memory_open"]:
                if self.kwargs["memory_type"] == "sample":
                    loss += - torch.sum(torch.log(prob_dyn) * memory_slots_values[:self.kwargs["memory_k"]]) * self.kwargs["memory_weight_c"] 
            return loss, sim
        return prob, query_orl
    
    def forward(self,aq,aa,pq,pa,nq,na, nodes,memory_slots_keys,memory_slots_values):

        # nodes = self.linear6(torch.cat([self.gat(nodes, self.adj),nodes],dim=-1))
        
        anchor_loss,a_prob_static = self.inference(aq, aa, nodes, True, memory_slots_keys, memory_slots_values)
        positive_loss,p_prob_static = self.inference(pq, pa, nodes, True, memory_slots_keys, memory_slots_values)
        negative_loss,n_prob_static = self.inference(nq, na, nodes, True, memory_slots_keys, memory_slots_values)
        if self.kwargs["contrastive_type"] == "mse":
            sim1 = self.mse_loss(a_prob_static, n_prob_static)
            sim2 = self.mse_loss(a_prob_static, p_prob_static)
        elif self.kwargs["contrastive_type"] == "l1":
            sim1 = self.l1_loss(a_prob_static, n_prob_static)
            sim2 = self.l1_loss(a_prob_static, p_prob_static)
        elif self.kwargs["contrastive_type"] == "cos":
            sim1 = torch.cosine_similarity(a_prob_static, p_prob_static).mean()
            sim2 = torch.cosine_similarity(a_prob_static, n_prob_static).mean()
            
        contrastive_loss = torch.max(torch.tensor(0), self.a - sim1 + sim2)
        
        if self.kwargs["contrastive_open"]:
            loss = anchor_loss + contrastive_loss* self.kwargs["loss_fusion_b"]
        else:
            loss = anchor_loss
        return loss, contrastive_loss, sim1, sim2