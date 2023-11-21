# -*- coding: utf-8 -*-
import pickle as pkl
import networkx as nx
import jieba
from gensim.models import word2vec
from sklearn import metrics
from tqdm import tqdm
import numpy as np
import json
import pandas as pd
import random
from sentence_transformers import SentenceTransformer
from rca import RCA
from dataset import Bu_Dataset
from torch.utils.data import DataLoader
import torch
from gat_network import GAT
from tool import *
import time


import argparse
parser = argparse.ArgumentParser()
# Model params
# GAT 
parser.add_argument("--gat_open", default=False, type=str2bool)
parser.add_argument("--gat_input_size", default=512, type=int)
parser.add_argument("--gat_hidden_size", default=128, type=int)
# RCA
parser.add_argument("--rca_input_size", default=512, type=int)
parser.add_argument("--rca_hidden_size", default=128, type=int)
# Contrastive
parser.add_argument("--contrastive_open", default=False, type=str2bool)
parser.add_argument("--contrastive_a", default=1.1, type=float)
parser.add_argument("--loss_fusion_b", default=1, type=float)
parser.add_argument("--contrastive_type", default="l1", type=str, choices=["mse", "cos","l1"])
# Memory Network
parser.add_argument("--memory_open", default=False, type=str2bool)
parser.add_argument("--memory_type", default="sample", type=str, choices=["prediction", "query","sample"])
parser.add_argument("--memory_weight_c", default=1, type=float)
parser.add_argument("--memory_k", default=10, type=int)
# Context
parser.add_argument("--random_seed", default=42, type=int)
parser.add_argument("--batch_train_batch_num", default=10, type=int)
parser.add_argument("--batch_train_open", default=False, type=str2bool)
parser.add_argument("--online_train_open", default=True, type=str2bool)
parser.add_argument("--record_max_batch_num", default=37, type=int)
parser.add_argument("--gpu", default="0", type=str)
# 获取参数列表
params = vars(parser.parse_args())
# 设置gpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.environ ["CUDA_VISIBLE_DEVICES"] = params["gpu"]
# 固定随机种子
seed_everything(params["random_seed"])

def main(params):

    with open("data/datasets_samples.pkl","rb") as f:
        datasets = pkl.load(f) 
    with open("data/datasets_graphs.pkl","rb") as f:
        graphs = pkl.load(f) 
    with open("data/root_bu2root_cause.pkl","rb") as f:
        root_bu2root_cause = pkl.load(f)    
    with open("data/apps.pkl","rb") as f:
        apps = pkl.load(f)
    model = SentenceTransformer("sentence-transformers/distiluse-base-multilingual-cased-v2",
                                cache_folder="/home/hongyi/workspace/change_localization/model")

    root_buss = [[...], # 本地生活数据集 
                 [...], # 物流数据集
                 [...] # 电商数据集 
                 ]

    ress = []
    ress_mid = []
    infer_times = []
    for i in tqdm(range(0,len(datasets))):

        # 初始化记忆slot
        memory_slots_keys=None
        memory_slots_values=None
        
        # 导入graph node 表示
        sub_graph = graphs[i]
        adj = torch.tensor(nx.adjacency_matrix(sub_graph).todense()).to(device)
        node_list = list(sub_graph.nodes())
        nodes_embed = [model.encode(apps[j]["des"]) for j in node_list]
        nodes_embed = torch.from_numpy(np.array(nodes_embed)).to(device) # 要乘batch_size 

        # 构建数据集，对于每一个样本，构建正负样本集合
        test_data_with_root_bu2 = datasets[i]
        bu_dataset = Bu_Dataset(test_data_with_root_bu2,node_list,root_bu2root_cause,root_buss[i])
        bu_dataloader = DataLoader(bu_dataset,batch_size=5)
        
        # 构建模型
        rca_model = RCA(a=params["contrastive_a"], embed_dim=params["rca_input_size"],
                        adj=adj, hidden_dim=params["rca_hidden_size"], 
                        device=device, target_class = len(node_list), **params).to(device)
        optimizer=torch.optim.Adadelta(params=rca_model.parameters())
        gat = GAT(input_size=params["gat_input_size"], hidden_size=params["gat_hidden_size"], 
              output_size=params["gat_hidden_size"], dropout=0.1,
              alpha=0.01, nheads=3, concat=True, device=device).to(device)
        optimizer_gat = torch.optim.Adam(params=gat.parameters())
        
        # 离线训练
        temp_memory_open = params["memory_open"]
        temp_contrastive_open = params["contrastive_open"]
        if params["batch_train_open"]:
            params["memory_open"] = False
            for i in range(5):
                ind = 0
                for aq,aa,pq,pa,nq,na,fault_des in tqdm(bu_dataloader):   
                    ind += 1
                    if ind > params["batch_train_batch_num"]:
                        continue
                    
                    aq = aq.to(device)
                    aa = aa.to(device).to(torch.float32)
                    pq = pq.to(device)
                    pa = pa.to(device)
                    nq = nq.to(device)
                    na = na.to(device)
                    
                    # 模型推断
                    
                    rca_model.eval()
                    gat.eval()
                    if params["gat_open"]:
                        nodes =  gat(nodes_embed, adj)
                        pred, embed = rca_model.inference(aq, aa,  nodes, False,memory_slots_keys, memory_slots_values)
                    else:
                        pred, embed = rca_model.inference(aq, aa,  nodes_embed, False,memory_slots_keys, memory_slots_values)
                    
                    if memory_slots_keys is None:  ###########单个更新，batch_size得for循环更新
                        memory_slots_keys = embed.detach().clone()
                        memory_slots_values = aa
                    else:
                        memory_slots_keys = torch.cat([memory_slots_keys, embed.detach().clone()])
                        memory_slots_values = torch.cat([memory_slots_values, aa])
                    
                    
                    # Train RCA model
                    rca_model.train()
                    gat.eval()
                    optimizer.zero_grad()
                    if params["gat_open"]:
                        gat.train()
                        optimizer_gat.zero_grad()
                        nodes = gat(nodes_embed, adj)
                        loss,contrastive_loss,_,_ = rca_model.forward(aq, aa, pq, pa, nq, na, nodes, memory_slots_keys, memory_slots_values) # nodes 与 nodes_embed 不能同名
                    else:
                        loss,contrastive_loss,_,_ = rca_model.forward(aq, aa, pq, pa, nq, na, nodes_embed, memory_slots_keys, memory_slots_values)
                    loss.backward()
                    optimizer.step()
                    if params["gat_open"]:
                        optimizer_gat.step()
                    print(f"epoch {i} train loss: {loss.detach().cpu().numpy()} contrastive_loss: {contrastive_loss.detach().cpu().numpy()}")

        # 在线处理
        preds = []
        labels = []
        ind = 0
        max_batch_num = params["record_max_batch_num"]
        params["memory_open"] = temp_memory_open
        params["contrastive_open"] = temp_contrastive_open
        infer_time = []
        for aq,aa,pq,pa,nq,na,fault_des in tqdm(bu_dataloader):    
            ind += 1
            if params["batch_train_open"]:
                if ind <= params["batch_train_batch_num"]:
                    continue
            
            aq = aq.to(device)
            aa = aa.to(device).to(torch.float32)
            pq = pq.to(device)
            pa = pa.to(device)
            nq = nq.to(device)
            na = na.to(device)
            
            # 模型推断
            start_time = time.time()
            rca_model.eval()
            gat.eval()
            if params["gat_open"]:
                nodes =  gat(nodes_embed, adj)
                pred, embed = rca_model.inference(aq, aa,  nodes, False,memory_slots_keys, memory_slots_values)
            else:
                pred, embed = rca_model.inference(aq, aa,  nodes_embed, False,memory_slots_keys, memory_slots_values)
            label = torch.argmax(aa, dim=-1)
            preds.append(pred.detach().cpu().numpy())
            labels.append(label.detach().cpu().numpy())
            
            
            if ind <= max_batch_num:
                labels_np = np.concatenate(labels, axis=0)
                preds_np = np.concatenate(preds, axis=0)
                res_50 = metrics.top_k_accuracy_score(labels_np, preds_np, k=20, labels=np.arange(len(preds_np[0])))
                res_100 = metrics.top_k_accuracy_score(labels_np, preds_np, k=50, labels=np.arange(len(preds_np[0])))
                res_150 = metrics.top_k_accuracy_score(labels_np, preds_np, k=100, labels=np.arange(len(preds_np[0])))
                res = {"res_20":res_50,"res_50":res_100,"res_100":res_150}
                ress_mid.append(res)
            # else:
            #     print("point")

            if memory_slots_keys is None:  ###########单个更新，batch_size得for循环更新
                memory_slots_keys = embed.detach().clone()
                memory_slots_values = aa
            else:
                memory_slots_keys = torch.cat([memory_slots_keys, embed.detach().clone()])
                memory_slots_values = torch.cat([memory_slots_values, aa])
            end_time = time.time()
            infer_time.append(end_time-start_time)
            # Train RCA model
            if params["online_train_open"]:
                for i in range(5):
                    rca_model.train()
                    gat.eval()
                    optimizer.zero_grad()
                    if params["gat_open"]:
                        gat.train()
                        optimizer_gat.zero_grad()
                        nodes = gat(nodes_embed, adj)
                        loss,contrastive_loss,sim1,sim2 = rca_model.forward(aq, aa, pq, pa, nq, na, nodes, memory_slots_keys, memory_slots_values) # nodes 与 nodes_embed 不能同名
                    else:
                        loss,contrastive_loss,sim1,sim2 = rca_model.forward(aq, aa, pq, pa, nq, na, nodes_embed, memory_slots_keys, memory_slots_values)
                    loss.backward()
                    optimizer.step()
                    if params["gat_open"]:
                        optimizer_gat.step()
                    print(f"epoch {i} train loss: {loss.detach().cpu().numpy()} contrastive_loss: {contrastive_loss.detach().cpu().numpy()} sim1: {sim1.detach().cpu().numpy()} sim2: {sim2.detach().cpu().numpy()}")
        
        print(f"bu_dataset.miss_count:{bu_dataset.miss_count}") 
        
        labels_np = np.concatenate(labels, axis=0)
        preds_np = np.concatenate(preds, axis=0)
        res_50 = metrics.top_k_accuracy_score(labels_np, preds_np, k=20, labels=np.arange(len(preds_np[0])))
        res_100 = metrics.top_k_accuracy_score(labels_np, preds_np, k=50, labels=np.arange(len(preds_np[0])))
        res_150 = metrics.top_k_accuracy_score(labels_np, preds_np, k=100, labels=np.arange(len(preds_np[0])))
        res = {"res_20":res_50,"res_50":res_100,"res_100":res_150}
        ress.append(res)
        print(res)
        infer_times.append(np.array(infer_time).sum())
    print(ress)
    infer_times = np.array(infer_times)
    print(infer_times)

    ress2 = []
    for v in ress:
        ress2.append(np.array(list(v.values())))
    ress2 = np.concatenate(ress2).reshape(1,-1)
    
    ress_mid2 = []
    for v in ress_mid:
        ress_mid2.append(np.array(list(v.values())))
    ress_mid2 = np.concatenate(ress_mid2).reshape(3,-1,3)
    ress_mid2_np = np.concatenate([ress_mid2[0],ress_mid2[1],ress_mid2[2]],axis=-1)
    if params["batch_train_open"]:
        ress_mid2_np = np.concatenate([np.zeros_like(ress_mid2_np)[:params["batch_train_batch_num"]],ress_mid2_np])
    ress_mid2_np = np.concatenate([ress_mid2_np,ress2])
    anss2 = pd.DataFrame(ress_mid2_np)
    anss2.to_csv(f"result/our_incremental_{params['batch_train_open']}_{params['online_train_open']}_{params['contrastive_open']}_{params['contrastive_type']}_{params['contrastive_a']}_{params['loss_fusion_b']}_{params['memory_open']}_{params['memory_type']}_{params['memory_k']}_{params['memory_weight_c']}_{params['batch_train_batch_num']}.csv",sep=",",float_format = '%.4f')
    
    
main(params)