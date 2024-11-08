import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as functional
import torch_sparse
from utils import bpr,generate_gumbel_masks,generate_rw_adj_matrix,normalized_unique,gnn_augmentation,gumbel_process

class Model(nn.Module):
    def __init__(self, Data, opt, device):
        super(Model, self).__init__()
        self.interact_train = Data.interact_train.reset_index(drop=True)
        self.user_num = Data.user_num
        self.item_num = Data.item_num
        self.device = device
        self.user_Embedding = nn.Embedding(self.user_num, opt.embedding_size)
        self.item_Embedding = nn.Embedding(self.item_num, opt.embedding_size)
        self.L = opt.L
        self.rec_loss_reg = opt.rec_loss_reg 
        self.ssl_loss_reg = opt.ssl_loss_reg 
        self.walk_length = opt.walk_length 
        self.ssl_temp = opt.ssl_temp
        self.choosing_tmp = opt.choosing_tmp
        self.sparse_reg = opt.sparse_reg
        self.create_sparse_adjaceny()
        self.rw=generate_rw_adj_matrix(self, self.walk_length, self.row, self.col)




    def create_sparse_adjaceny(self):
        index = [self.interact_train['userid'].tolist(), self.interact_train['itemid'].tolist()] # [[userid],[itemid]]表示交互索引
        value = [1.0] * len(self.interact_train)

   
        self.interact_matrix = torch.sparse_coo_tensor(index, value, (self.user_num, self.item_num))

        tmp_index = [self.interact_train['userid'].tolist(), (self.interact_train['itemid'] + self.user_num).tolist()]
        tmp_adj = torch.sparse_coo_tensor(tmp_index, value,
                                          (self.user_num + self.item_num, self.user_num + self.item_num))

        self.joint_adjaceny_matrix = (tmp_adj + tmp_adj.t())


   
        degree = torch.sparse.sum(self.joint_adjaceny_matrix, dim=1).to_dense()
        degree = torch.pow(degree, -0.5)
        degree[torch.isinf(degree)] = 0
        D_inverse = torch.diag(degree, diagonal=0).to_sparse()

        self.joint_adjaceny_matrix_normal = torch.sparse.mm(torch.sparse.mm(D_inverse, self.joint_adjaceny_matrix),
                                                            D_inverse).coalesce()

        joint_indices = self.joint_adjaceny_matrix_normal.indices()
        self.row = joint_indices[0]
        self.col = joint_indices[1]


        self.joint_adjaceny_matrix_normal = self.joint_adjaceny_matrix_normal.to(self.device)


    def forward(self, user_id, pos_item, neg_item, opt):
        # GNN agumentor
        cur_embedding = torch.cat([self.user_Embedding.weight, self.item_Embedding.weight], dim=0)
        all_embeddings = [cur_embedding]
        edge_mask_list, node_mask_list = generate_gumbel_masks(
            cur_embedding, self.edge_mask_learner, self.node_mask_learner, self.L, self.ssl_temp, self.device
        )
        for i in range(self.L):
            cur_embedding = torch.mm(self.joint_adjaceny_matrix_normal, cur_embedding) 
            all_embeddings.append(cur_embedding)
        all_embeddings = torch.stack(all_embeddings, dim=0)
        all_embeddings = torch.mean(all_embeddings, dim=0, keepdim=False)#
        user_embeddings, item_embeddings = torch.split(all_embeddings, [self.user_num, self.item_num])
        cur_embedding_edge = torch.cat([self.user_Embedding.weight, self.item_Embedding.weight], dim=0)
        all_embeddings_edge = [cur_embedding_edge]
        cur_embedding_node = torch.cat([self.user_Embedding.weight, self.item_Embedding.weight], dim=0)
        all_embeddings_node = [cur_embedding_node]
        edge_reg = 0
        node_reg=0
        edge_mask_list, node_mask_list = gumbel_process(
        cur_embedding, self.edge_mask_learner, self.node_mask_learner, self.L, self.ssl_temp, self.device
    )

        all_embeddings_edge, edge_reg = gnn_augmentation(
            cur_embedding, edge_mask_list, self.joint_adjaceny_matrix_normal, is_edge=True, L=self.L
        )
        all_embeddings_node, node_reg = gnn_augmentation(
            cur_embedding, node_mask_list, self.joint_adjaceny_matrix_normal, rw_adj=self.rw_adj, is_edge=False, L=self.L
        )
        
        node_reg = node_reg / self.L
        edge_reg=  edge_reg/self.L
        all_embeddings_node = torch.stack(all_embeddings_node, dim=0)
        all_embeddings_node = torch.mean(all_embeddings_node, dim=0, keepdim=False)
        user_embeddings_node, item_embeddings_node = torch.split(all_embeddings_node,
                                                                           [self.user_num, self.item_num])
        all_embeddings_edge = torch.stack(all_embeddings_edge, dim=0)
        all_embeddings_edge = torch.mean(all_embeddings_edge, dim=0, keepdim=False)
        user_embeddings_edge, item_embeddings_edge = torch.split(all_embeddings_node,
                                                                           [self.user_num, self.item_num])

        ###################compute rec loss#########################

        user_embedded = user_embeddings[user_id]
        pos_item_embedded = item_embeddings[pos_item]
        neg_item_embedded = item_embeddings[neg_item]

   
        # rec loss
        rec_loss = bpr(user_embedded,pos_item_embedded,neg_item_embedded)

        ############
        user_embedded_edge = user_embeddings_edge[user_id]
        pos_item_embedded_edge = item_embeddings_edge[pos_item]
        neg_item_embedded_edge = item_embeddings_edge[neg_item]

        rec_loss_edge=bpr(user_embedded_edge,pos_item_embedded_edge,neg_item_embedded_edge)
       

        # rec loss
       

        user_embedded_node = user_embeddings_node[user_id]
        pos_item_embedded_node = item_embeddings_node[pos_item]
        neg_item_embedded_node = item_embeddings_node[neg_item]

        
        # rec loss
        rec_loss_node= bpr(user_embedded_node,pos_item_embedded_node,neg_item_embedded_node)

     
        ###################compute ssl mi#########################
        def sampling(sim_matrix, K_pos,device):
            N = len(sim_matrix)
            identity_matrix = torch.ones(N, N)
            diag = identity_matrix.diagonal(0)
            diag.zero_()
            identity_matrix = identity_matrix.to(self.device)

            K_pos = int(K_pos * N) 
            # K_neg = int(K_neg * N) 
            pos_matrix = torch.zeros(N, N).to(device)
            # neg_matrix = torch.zeros(N, N).to(device)

            # pos sampling
            for i in range(N):
                _, indices = torch.topk(sim_matrix[i], k=K_pos, largest=True) 
                pos_matrix[i][indices] = 1

            sim_matrix = torch.mul(sim_matrix, ~pos_matrix.bool())

            # neg sampling
            # for i in range(N):
            #     _, indices = torch.topk(sim_matrix[i], k=K_neg, largest=False)
            #     neg_matrix[i][indices] = 1

            return torch.mul(identity_matrix, pos_matrix)

        def judge(martix, z1, z2):
            nonzero_indices = torch.nonzero(martix)
            row_indices, col_indices = nonzero_indices[:, 0], nonzero_indices[:, 1]

            embedded_row = z1[row_indices]
            embedded_col = z2[col_indices]
            concatenated_embeddings = torch.cat((embedded_row, embedded_col), dim=1)

            MLP_output = self.MLP(concatenated_embeddings)
            bias = 0.0 + 0.0001
            eps = (bias - (1 - bias)) * torch.rand(MLP_output.size()) + (1 - bias)  # eps ~ Uniform(0,1)
            mlp_gate_inputs = torch.log(eps) - torch.log(1 - eps)
            mlp_gate_inputs = mlp_gate_inputs.to(self.device)
            mlp_gate_inputs = (MLP_output + mlp_gate_inputs) / self.choosing_tmp
            MLP_output = torch.sigmoid(mlp_gate_inputs).squeeze(1)  
            martix[row_indices, col_indices] = MLP_output

            return martix


        def ssl_compute(normalized_embedded_s1, normalized_embedded_s2, opt):
            similarity = torch.mm(normalized_embedded_s1, normalized_embedded_s2.t()) 
            f = lambda x: torch.exp(x / self.ssl_temp)
            all_score = f(similarity)
            

            pos_matrix = sampling(all_score, K_pos=opt.K_pos_ratio,
                                              device=self.device) #

            pos_mlp = judge(pos_matrix, normalized_embedded_s1, normalized_embedded_s2) #
            # neg_mlp = judge(neg_matrix, normalized_embedded_s1, normalized_embedded_s2)

            pos_matrix = (pos_mlp > opt.tao_sample).float()
            # neg_matrix = (neg_mlp > opt.tao_sample).float()
            
            pos = torch.sum(torch.mul(normalized_embedded_s1, normalized_embedded_s2), dim=1, keepdim=False)
            pos_score = torch.sum(torch.mul(pos_matrix, all_score), dim=1, keepdim=False)
            # neg_score = torch.sum(torch.mul(neg_matrix, all_score), dim=1, keepdim=False)

            ssl_mi = -torch.log((pos_score + pos)).mean()
            # ssl_mi = -torch.log(pos_score + pos / (pos_score + pos + neg_score)).mean()
            return ssl_mi
        

        normalized_user_embedded_unique =  normalized_unique(user_embeddings,user_id)
        normalized_item_embedded_unique = normalized_unique(item_embeddings,pos_item)
        normalized_user_embedded_edge = normalized_unique(user_embedded_edge,user_id)
        normalized_user_embedded_node = normalized_unique(user_embedded_node,user_id)
        normalized_item_embedded_edge = normalized_unique(item_embeddings_edge,pos_item)
        normalized_item_embedded_node = normalized_unique(item_embeddings_node,pos_item)

        score_user_edge = ssl_compute(normalized_user_embedded_edge, normalized_user_embedded_unique, opt) 
        score_item_edge = ssl_compute(normalized_item_embedded_edge, normalized_item_embedded_unique, opt)

        score_user_node = ssl_compute(normalized_user_embedded_node, normalized_user_embedded_unique, opt)
        score_item_node = ssl_compute(normalized_item_embedded_node, normalized_item_embedded_unique, opt)
        clloss=score_user_edge + score_item_edge + score_user_node + score_item_node

        loss = self.rec_loss_reg * (rec_loss_edge + rec_loss_node) + rec_loss + self.ssl_loss_reg *clloss + self.sparse_reg * (
                       node_reg + edge_reg)

     

        return loss

    def predict(self, user_id):
        # GNN agumentor
        cur_embedding = torch.cat([self.user_Embedding.weight, self.item_Embedding.weight], dim=0)
        all_embeddings = [cur_embedding]
        edge_mask_list, node_mask_list = generate_gumbel_masks(
            cur_embedding, self.edge_mask_learner, self.node_mask_learner, self.L, self.ssl_temp, self.device
        )

        for i in range(self.L):
            cur_embedding = torch.mm(self.joint_adjaceny_matrix_normal, cur_embedding) 
            all_embeddings.append(cur_embedding)
        all_embeddings = torch.stack(all_embeddings, dim=0)
        all_embeddings = torch.mean(all_embeddings, dim=0, keepdim=False)#
        user_embeddings, item_embeddings = torch.split(all_embeddings, [self.user_num, self.item_num])
        cur_embedding_edge = torch.cat([self.user_Embedding.weight, self.item_Embedding.weight], dim=0)
        all_embeddings_edge = [cur_embedding_edge]
        cur_embedding_node = torch.cat([self.user_Embedding.weight, self.item_Embedding.weight], dim=0)
        all_embeddings_node = [cur_embedding_node]
        edge_mask_list, node_mask_list = gumbel_process(
        cur_embedding, self.edge_mask_learner, self.node_mask_learner, self.L, self.ssl_temp, self.device
    )

        all_embeddings_edge, edge_reg = gnn_augmentation(
            cur_embedding, edge_mask_list, self.joint_adjaceny_matrix_normal, is_edge=True, L=self.L
        )
        all_embeddings_node, node_reg = gnn_augmentation(
            cur_embedding, node_mask_list, self.joint_adjaceny_matrix_normal, rw_adj=self.rw_adj, is_edge=False, L=self.L
        )
    
        all_embeddings_node = torch.stack(all_embeddings_node, dim=0)
        all_embeddings_node = torch.mean(all_embeddings_node, dim=0, keepdim=False)
        user_embeddings_node, item_embeddings_node = torch.split(all_embeddings_node,
                                                                           [self.user_num, self.item_num])
        all_embeddings_edge = torch.stack(all_embeddings_edge, dim=0)
        all_embeddings_edge = torch.mean(all_embeddings_edge, dim=0, keepdim=False)
        user_embeddings_edge, item_embeddings_edge = torch.split(all_embeddings_node,
                                                                           [self.user_num, self.item_num])
        
        # uuu * 3embedding_size
        user_embedded = torch.cat(
            (user_embeddings[user_id], user_embeddings_edge[user_id], user_embeddings_node[user_id]), dim=-1)
        # item_num * 3embedding_size
        pos_item_embedded = torch.cat((item_embeddings, item_embeddings_edge, item_embeddings_node), dim=-1)
        # uuu * item_num
        score = torch.mm(user_embedded, pos_item_embedded.t())

        return score
