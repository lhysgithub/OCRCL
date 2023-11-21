import torch
import jieba
import random
import numpy as np
from torch.utils.data import Dataset
from sentence_transformers import SentenceTransformer
from gensim.models import word2vec

def sentence_to_vector_tfidf(sentence, model, tfidf_model):
    words = list(jieba.cut(sentence)) 
    flag = 0
    for word in words:
        if word not in tfidf_model.vocabulary_:
            flag = 1
            break
    
    if flag:
        vector = sum([model.wv[word] for word in words if word in model.wv])
    else:
        vector = sum([model.wv[word] * tfidf_model.idf_[tfidf_model.vocabulary_[word]]
                    for word in words if word in model.wv and word in tfidf_model.vocabulary_])
        
    return vector / len(vector)

class Bu_Dataset(Dataset):
    def __init__(self,data,node_list,root_bu2root_cause,bu_list,model=None,tfidf_model=None):
        super(Bu_Dataset,self).__init__()
        self.data = data
        self.node_list = node_list
        self.bu_list = bu_list
        self.root_bu2root_cause = root_bu2root_cause 
        self.miss_count = 0
        self.model = SentenceTransformer("sentence-transformers/distiluse-base-multilingual-cased-v2",
                                cache_folder="/home/hongyi/workspace/change_localization/model")
    
    def get_one_hot_label(self, name, node_list):
            index_ = node_list.index(name)
            one_hot_lable = np.zeros(len(node_list))
            one_hot_lable[index_] = 1
            return one_hot_lable
    
    def __getitem__(self, index):
         # anchor 
        k = list(self.data.keys())[index]
        i = 1
        if k not in self.node_list:
            self.miss_count+=1
        while k not in self.node_list:
            k = list(self.data.keys())[index-i]
            i += 1
        v = self.data[k]
        root_bu = v["root_bu"]
        fault_des = v["title"]
        aq = self.model.encode(fault_des)
        # aq = sentence_to_vector_tfidf(fault_des, self.model, self.tfidf_model)
        aa = self.get_one_hot_label(k,self.node_list)
        
        # positive 
        # 在当前bu下找正样本
        current_bu_apps = self.root_bu2root_cause[root_bu]
        candidate_apps = [i for i in current_bu_apps if i!=k]
        p_random_k = random.choice(candidate_apps)
        while p_random_k not in self.node_list:
             p_random_k = random.choice(candidate_apps)
        pa = self.get_one_hot_label(p_random_k,self.node_list)
        p_fault_des = self.data[p_random_k]["title"] 
        pq = self.model.encode(p_fault_des)

        
        # negative 
        # 在其他bu下找负样本
        random_bu = random.choice([i for i in self.bu_list if i!=root_bu])
        selected_bu_apps = self.root_bu2root_cause[random_bu]
        n_random_k = random.choice(selected_bu_apps)
        while n_random_k not in self.node_list:
            n_random_k = random.choice(selected_bu_apps)
        n_fault_des = self.data[n_random_k]["title"] 
        nq = self.model.encode(n_fault_des)
        na = self.get_one_hot_label(n_random_k,self.node_list)
        
        
        return aq,aa,pq,pa,nq,na,fault_des
    
    def __len__(self):
        return len(self.data)