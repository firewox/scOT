'''
Author: Kai Cao
Modified from SCALEX
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from itertools import cycle
import sys
import time
from .layer import *
from .loss import *
from sklearn.metrics import (average_precision_score, classification_report, roc_auc_score,precision_recall_curve)
import pandas as pd
import os
import psutil
from scipy.spatial import distance_matrix, minkowski_distance, distance
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
from igraph import *
import uniport.model.mmd as mmd


class VAE(nn.Module):
    """
    Variational Autoencoder framework
    """
    def __init__(self, enc, dec, ref_id, n_domain, mode,batch_size,lambda_recon,lambda_kl,lambda_ot,lambda_response):
        """
        Parameters
        ----------
        enc
            Encoder structure config
        dec
            Decoder structure config
        ref_id
            ID of reference dataset
        n_domain
            The number of different domains
        mode
            Choose from ['h', 'v', 'd']

        """
        super().__init__()


        x_dim = {}
        self.enc=enc #TODO new feature
        self.dec=dec #TODO new feature
        self.batch_size=batch_size #TODO new feature
        self.lambda_recon=lambda_recon #TODO new feature
        self.lambda_kl=lambda_kl #TODO new feature
        self.lambda_ot=lambda_ot #TODO new feature
        self.lambda_response=lambda_response #TODO new feature
        for key in dec.keys():
            if key!=3: #TODO,不把药物响应预测模型的维度，加到encoder里
                x_dim[key] = dec[key][-1][1]
        # self.z_dim = enc[-1][1]
        for key in enc.keys():
            if key == 0:
                self.z_dim = enc[0][-1][1]
            if key == 4:
                self.z_dim = enc[4][-1][1]
                
        self.encoder = Encoder(x_dim, enc, mode)
        self.decoder = Decoder(self.z_dim, dec)

        self.n_domain = n_domain
        self.ref_id = ref_id
  
    def load_model(self, path):
        """
        Load trained model parameters dictionary.
        Parameters
        ----------
        path
            file path that stores the model parameters
        """
        pretrained_dict = torch.load(path, map_location=lambda storage, loc: storage)                            
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict) 
        self.load_state_dict(model_dict)
        
    def encodeBatch( # 对源域bulk数据进行平衡采样时，使用这个
            self, 
            adata_cm, ###TODO,new feature
            adatas, ###TODO,new feature
            dataloader, 
            num_cell,#TODO new feature
            num_gene,
            mode='h',  
            out='latent', 
            batch_id=0,
            pred_id=1,
            device='cuda', 
            eval=False,
            DRUG='Gefitinib', ###TODO,new feature
            source_batch=0, ###TODO,new feature
            target_batch=0, ###TODO,new feature
            sampler="none", ###TODO,new feature
            unshared_encoder=False, ###TODO,new feature
        ):
        """
        Inference

        Parameters
        ----------
        dataloader
            An iterable over the given dataset for inference.
        num_gene
            List of number of genes in different datasets. 
        mode
            Choose from ['h', 'v', 'd']
            If 'h', integrate data with common genes
            If 'v', integrate data profiled from the same cells
            If 'd', inetrgate data without common genes
            Default: 'h'.
        out
            Output of uniPort. Choose from ['latent', 'project', 'predict']. 
            If out='latent', train the network and output cell embeddings. 
            If out='project', project data into the latent space and output cell embeddings. 
            If out='predict', project data into the latent space and output cell embeddings through a specified decoder. 
            Default: 'latent'.
        batch_id
            Choose which encoder to project data when mode='d'. Default: 0.
        pred_id
            Choose which decoder to reconstruct data when out='predict'.
        device
            'cuda' or 'cpu' for . Default: 'cuda'.
        eval
            If True, set the model to evaluation mode. If False, set the model to train mode. Default: False.
        
        Returns
        -------
        output
            Cell embeddings (if out='latent' or 'project') or Predicted data (if out='predict').
        """

        self.to(device)
        if eval:
            self.eval()
        else:
            self.train()
        AUC_bulk_test=0.0
        APR_bulk_test=0.0
        AUC_sc_test=0.0
        APR_sc_test=0.0
        # indices = np.zeros(dataloader.dataset.shape[0])
        output = []
        if out == 'latent':
            # output = np.zeros((dataloader.dataset.shape[0], self.z_dim))

            if mode == 'v':
                for x, idx in dataloader:
                    x = x.float().to(device)
                    z = self.encoder(x[:, 0:num_gene[0]], 0)[1]
                    output.append(z.detach().cpu())
                output = torch.cat(output).numpy()

            elif mode == 'd':
                for x, y,idx in dataloader:
                    x, y = x[:, 0:num_gene[batch_id]].float().to(device), y.long().to(device)
                    loc = torch.where(y==batch_id)[0]
                    x = x[loc]
                    z = self.encoder(x, batch_id)[1] # z, mu, var
                    output.append(z.detach().cpu())
                output = torch.cat(output).numpy()

            elif mode == 'h':
                count_a=0
                count_b=0
                bulk_best_threshold = 0.0
                bulk_best_f1_score = 0.0
                sc_best_threshold = 0.0
                sc_best_f1_score = 0.0
                #output = np.zeros((dataloader.dataset.shape[0], self.z_dim)) #output.shape=(22518, 16)
                #TODO new feature
                output = np.zeros((num_cell[0]+num_cell[1], self.z_dim)) #output.shape=(22518, 16)
                '''for x,y,idx in dataloader: #=x.shape=torch.Size([11259, 4000])
                    print(f'####vae.py#162rows########count_a={count_a},count_b={count_b},dataloader.dataset.shape={dataloader.dataset.shape}')
                    #TODO 这里对于x，y，idx是什么,疑问？
                    # print(f'vae.py#146行###########x,y,idx in dataloader:===x.shape={x.shape},y.shape={y.shape},idx.shape={idx.shape},output.shape={output.shape}')
                    x_c = x[:, 0:num_gene[self.n_domain]].float().to(device)
                    #TODO，@see：vae.py, 462rows, z,mu,var = self.encoder(x_c, 0)
                    z = self.encoder(x_c, 0)[1] #？？？这里竟然取得是mu，不重采样吗，
                    #z = self.encoder(x_c, 0)[0]
                    loc = {}
                    loc[0] = torch.where(y==0)[0]
                    loc[1] = torch.where(y==1)[0]
                    if adata_cm is not None:
                        if len(loc[0])>0: # 表示取数据集x的隐空间嵌入z（取bulk细胞系的隐空间嵌入z ）
                            count_a+=1
                            a = idx[y==0] #定位到数据集x（bulk细胞系）的索引，用来拿到其相应的response label
                            groundtruth_bulk_label = adata_cm.obs['response'].iloc[a,].values.reshape(-1,1)
                            groundtruth_bulk_label = torch.Tensor(groundtruth_bulk_label).to(device)
                            groundtruth_bulk_label = groundtruth_bulk_label.detach().cpu().numpy()
                            predicted_bulk_label = self.decoder(z[loc[0]], 0+2+1) # decoder(,0)表示所有共同数据的共同的解码器；decoder(,1)表示数据集X特异性高可变基因的解码器；decoder(,2)表示数据集Y特异性高可变基因的解码器；decoder(,3)表示所有共同数据里的数据X的药物响应解码器；
                            predicted_bulk_label = predicted_bulk_label.detach().cpu().numpy()
                            print(f'####vae.py#511rows#######predicted_bulk_label.shape={predicted_bulk_label.shape},predicted_bulk_label[:10]={predicted_bulk_label[:10]},groundtruth_bulk_label[:10]={groundtruth_bulk_label[:10]}')
                            tmp_1 = roc_auc_score(groundtruth_bulk_label, predicted_bulk_label)
                            print(f'####vae.py#174rows####AUC_bulk_test={tmp_1}')
                            AUC_bulk_test += tmp_1 # AUC
                            #
                            tmp_2 = average_precision_score(groundtruth_bulk_label, predicted_bulk_label)
                            print(f'####vae.py#177rows####APR_bulk_test={tmp_2}')
                            APR_bulk_test += tmp_2
                            # 计算每个阈值下的性能指标
                            precision, recall, thresholds_pr = precision_recall_curve(groundtruth_bulk_label, predicted_bulk_label)
                            # 计算每个阈值下的 F1 分数
                            f1_scores = 2 * (precision * recall) / (precision + recall)
                            # 找到达到最佳性能的阈值对应的索引
                            best_threshold_idx = np.argmax(f1_scores)
                            # 获取最佳阈值和对应的性能指标
                            bulk_best_threshold = thresholds_pr[best_threshold_idx]
                            bulk_best_f1_score = f1_scores[best_threshold_idx]
                        if len(loc[1])>0: # 表示取数据集x的隐空间嵌入z（取bulk细胞系的隐空间嵌入z ）
                            count_b+=1
                            b = idx[y==1] #定位到数据集x（bulk细胞系）的索引，用来拿到其相应的response label
                            groundtruth_sc_label = adata_cm.obs['response'].iloc[b,].values.reshape(-1,1)
                            groundtruth_sc_label = torch.Tensor(groundtruth_sc_label).to(device)
                            groundtruth_sc_label = groundtruth_sc_label.detach().cpu().numpy()
                            predicted_sc_label = self.decoder(z[loc[1]], 0+2+1) # decoder(,0)表示所有共同数据的共同的解码器；decoder(,1)表示数据集X特异性高可变基因的解码器；decoder(,2)表示数据集Y特异性高可变基因的解码器；decoder(,3)表示所有共同数据里的数据X的药物响应解码器；
                            predicted_sc_label = predicted_sc_label.detach().cpu().numpy()
                            print(f'####vae.py#511rows#######predicted_sc_label.shape={predicted_sc_label.shape},predicted_sc_label[:10]={predicted_sc_label[:10]},groundtruth_sc_label[:10]={groundtruth_sc_label[:10]}')
                            tmp_3 = roc_auc_score(groundtruth_sc_label, predicted_sc_label)
                            print(f'####vae.py#189rows####AUC_sc_test={tmp_3}')
                            AUC_sc_test += tmp_3 # AUC
                            #
                            tmp_4 = average_precision_score(groundtruth_sc_label, predicted_sc_label)
                            print(f'####vae.py#192rows####APR_sc_test={tmp_4}')
                            APR_sc_test += tmp_4
                            #AUC_sc_test += roc_auc_score(groundtruth_sc_label, predicted_sc_label) # AUC
                            #APR_sc_test += average_precision_score(groundtruth_sc_label, predicted_sc_label) # Precision Recall
                            precision, recall, thresholds_pr = precision_recall_curve(groundtruth_sc_label, predicted_sc_label)
                            # 计算每个阈值下的 F1 分数
                            f1_scores = 2 * (precision * recall) / (precision + recall)
                            # 找到达到最佳性能的阈值对应的索引
                            best_threshold_idx = np.argmax(f1_scores)
                            # 获取最佳阈值和对应的性能指标
                            sc_best_threshold = thresholds_pr[best_threshold_idx]
                            sc_best_f1_score = f1_scores[best_threshold_idx]
                    output[idx] = z.detach().cpu().numpy()
                '''
                for i,data in enumerate(dataloader): #=x.shape=torch.Size([11259, 4000])
                    output = np.zeros((data[0][0].shape[0]+data[1][0].shape[0], self.z_dim)) #output.shape=(22518, 16)
                    #print(f'####vae.py#162rows########count_a={count_a},count_b={count_b}')
                    #TODO 这里对于x，y，idx是什么,疑问？
                    # print(f'vae.py#146行###########x,y,idx in dataloader:===x.shape={x.shape},y.shape={y.shape},idx.shape={idx.shape},output.shape={output.shape}')
                    # 1. batch里 的bulk数据
                    x_bulk = data[0][0].float().to(device)
                    response_bulk = data[0][1].float().to(device)
                    y_bulk = torch.tensor([0] * data[0][0].shape[0]).float().to(device) # bulk数据的domain_id设置为0
                    idx_bulk = data[0][2]
                    # 2. batch里 的sc数据
                    x_sc = data[1][0].float().to(device)
                    response_sc = data[1][1].float().to(device)
                    y_sc = torch.tensor([1] * data[1][0].shape[0]).float().to(device) # sc数据的domain_id设置为1
                    idx_sc = (data[1][2] + num_cell[0])
                    #idx_sc = data[1][2] + len(idx_bulk) #TODO new feature
                    # 使用 torch.cat() 函数在第一个维度上进行连接
                    x = torch.cat((x_bulk, x_sc), dim=0)
                    y = torch.cat((y_bulk, y_sc), dim=0)
                    idx = torch.cat((idx_bulk, idx_sc), dim=0)
                    response = torch.cat((response_bulk, response_sc), dim=0)

                    x_c = x.float().to(device)
                    #TODO，@see：vae.py, 462rows, z,mu,var = self.encoder(x_c, 0)
                    if unshared_encoder: #TODO 分别使用两个不同的编码器
                        z_bulk = self.encoder(x_bulk, 3)[1]
                        z_sc = self.encoder(x_sc, 4)[1]
                        z = torch.cat((z_bulk, z_sc), dim=0)
                    else: #TODO 使用共享编码器
                        z = self.encoder(x_c, 0)[1] #？？？这里竟然取得是mu，不重采样吗，
                        
                    #TODO,这里对self.encoder(x_c, 0)的返回值有疑问？我觉得应该是（z, mu, var）的形式，为什么取[1]?，[0].shape===torch.Size([11259, 16])，[1].shape=torch.Size([11259, 16])，[2].shape=torch.Size([11259, 16])
                    adatas[0].obsm['latent'] = z[0:num_cell[0]].detach().cpu().numpy()
                    adatas[1].obsm['latent'] = z[num_cell[0]:num_cell[0]+num_cell[1]].detach().cpu().numpy()
                    adata_cm.obsm["latent"]=z.detach().cpu().numpy()
                    #print(f'####vae.py##266rows, adatas[0]=={adatas[0]}')
                    #print(f'####vae.py##267rows, adatas[0].obsm[latent]=={adatas[0].obsm["latent"].shape}')
                    #print(f'####vae.py##268rows, z[0:num_cell[0]]=={z[0:num_cell[0]].shape}')
                    #print(f'####vae.py##269rows, adatas[1]=={adatas[1]}')
                    #print(f'####vae.py##270rows, adatas[1].obsm[latent]=={adatas[1].obsm["latent"].shape}')
                    #print(f'####vae.py##271rows, z[num_cell[0]:num_cell[0]+num_cell[1]]=={z[num_cell[0]:num_cell[0]+num_cell[1]].shape}')
                    #print(f'####vae.py##272rows, adata_cm=={adata_cm}')
                    #print(f'####vae.py##273rows, z=={z.shape}')
                    #print(f'####vae.py##274rows, adata_cm.obsm[latent]=={adata_cm.obsm["latent"].shape}')
                    loc = {}
                    loc[0] = torch.where(y==0)[0]
                    loc[1] = torch.where(y==1)[0]
                    if adata_cm is not None:
                        if len(loc[0])>0: # 表示取数据集x的隐空间嵌入z（取bulk细胞系的隐空间嵌入z ）
                            count_a+=1
                            groundtruth_bulk_label = response_bulk.reshape(-1,1)
                            groundtruth_bulk_label = torch.Tensor(groundtruth_bulk_label).to(device)
                            groundtruth_bulk_label = groundtruth_bulk_label.detach().cpu().numpy()
                            if unshared_encoder: #TODO 分别使用两个不同的编码器
                                predicted_bulk_label = self.decoder(z_bulk, 0+2+1) # decoder(,0)表示所有共同数据的共同的解码器；decoder(,1)表示数据集X特异性高可变基因的解码器；decoder(,2)表示数据集Y特异性高可变基因的解码器；decoder(,3)表示所有共同数据里的数据X的药物响应解码器；
                            else:
                                predicted_bulk_label = self.decoder(z[loc[0]], 0+2+1) # decoder(,0)表示所有共同数据的共同的解码器；decoder(,1)表示数据集X特异性高可变基因的解码器；decoder(,2)表示数据集Y特异性高可变基因的解码器；decoder(,3)表示所有共同数据里的数据X的药物响应解码器；
                            predicted_bulk_label = predicted_bulk_label.detach().cpu().numpy()
                            adatas[0].obs["sensitivity_prediction"] = predicted_bulk_label
                            # print(f'####vae.py#511rows#######predicted_bulk_label.shape={predicted_bulk_label.shape},predicted_bulk_label[:10]={predicted_bulk_label[:10]},groundtruth_bulk_label[:10]={groundtruth_bulk_label[:10]}')
                            tmp_1 = roc_auc_score(groundtruth_bulk_label, predicted_bulk_label)
                            # print(f'####vae.py#174rows####AUC_bulk_test={tmp_1}')
                            AUC_bulk_test += tmp_1 # AUC
                            #
                            tmp_2 = average_precision_score(groundtruth_bulk_label, predicted_bulk_label)
                            # print(f'####vae.py#177rows####APR_bulk_test={tmp_2}')
                            APR_bulk_test += tmp_2
                            # 计算每个阈值下的性能指标
                            precision, recall, thresholds_pr = precision_recall_curve(groundtruth_bulk_label, predicted_bulk_label)
                            # 计算每个阈值下的 F1 分数
                            f1_scores = 2 * (precision * recall) / (precision + recall)
                            # 找到达到最佳性能的阈值对应的索引
                            best_threshold_idx = np.argmax(f1_scores)
                            # 获取最佳阈值和对应的性能指标
                            bulk_best_threshold = thresholds_pr[best_threshold_idx] #TODO,这里可以根据阈值的选择，大于阈值为标签1，小于阈值的为标签0，把二值化的预测标签存入adatas[0].obs["sensitivity_prediction_label"]
                            bulk_best_f1_score = f1_scores[best_threshold_idx]
                        if len(loc[1])>0: # 表示取数据集x的隐空间嵌入z（取sc细胞系的隐空间嵌入z ）
                            count_b+=1
                            groundtruth_sc_label = response_sc.reshape(-1,1)
                            groundtruth_sc_label = torch.Tensor(groundtruth_sc_label).to(device)
                            groundtruth_sc_label = groundtruth_sc_label.detach().cpu().numpy()                            
                            if unshared_encoder:
                                predicted_sc_label = self.decoder(z_sc, 0+2+1) # decoder(,0)表示所有共同数据的共同的解码器；decoder(,1)表示数据集X特异性高可变基因的解码器；decoder(,2)表示数据集Y特异性高可变基因的解码器；decoder(,3)表示所有共同数据里的数据X的药物响应解码器；
                            else:
                                predicted_sc_label = self.decoder(z[loc[1]], 0+2+1) # decoder(,0)表示所有共同数据的共同的解码器；decoder(,1)表示数据集X特异性高可变基因的解码器；decoder(,2)表示数据集Y特异性高可变基因的解码器；decoder(,3)表示所有共同数据里的数据X的药物响应解码器；
                            predicted_sc_label = predicted_sc_label.detach().cpu().numpy()
                            adatas[1].obs["sensitivity_prediction"] = predicted_sc_label #TODO new feature
                            # print(f'####vae.py#511rows#######predicted_sc_label.shape={predicted_sc_label.shape},predicted_sc_label[:10]={predicted_sc_label[:10]},groundtruth_sc_label[:10]={groundtruth_sc_label[:10]}')
                            tmp_3 = roc_auc_score(groundtruth_sc_label, predicted_sc_label)
                            # print(f'####vae.py#189rows####AUC_sc_test={tmp_3}')
                            AUC_sc_test += tmp_3 # AUC
                            #
                            tmp_4 = average_precision_score(groundtruth_sc_label, predicted_sc_label)
                            # print(f'####vae.py#192rows####APR_sc_test={tmp_4}')
                            APR_sc_test += tmp_4
                            # AUC_sc_test += roc_auc_score(groundtruth_sc_label, predicted_sc_label) # AUC
                            # APR_sc_test += average_precision_score(groundtruth_sc_label, predicted_sc_label) # Precision Recall
                            # 计算每个阈值下的性能指标
                            precision, recall, thresholds_pr = precision_recall_curve(groundtruth_sc_label, predicted_sc_label)
                            # 计算每个阈值下的 F1 分数
                            f1_scores = 2 * (precision * recall) / (precision + recall)
                            # 找到达到最佳性能的阈值对应的索引
                            best_threshold_idx = np.argmax(f1_scores)
                            # 获取最佳阈值和对应的性能指标
                            sc_best_threshold = thresholds_pr[best_threshold_idx] #TODO,这里可以根据阈值的选择，大于阈值为标签1，小于阈值的为标签0，把二值化的预测标签存入adatas[1].obs["sensitivity_prediction_label"]
                            sc_best_f1_score = f1_scores[best_threshold_idx]
                    #print(f'####315 rows## output.shape=={output.shape}, z.shape=={z.shape}, idx.length=={len(idx)}, idx=={idx}')
                    output[idx] = z.detach().cpu().numpy()
                    adata_cm.obsm["latent_output"] = output #TODO,new feature
                    ####TODO Save adata ####START
                    # Create directories if they do not exist
                    for path in ['./drug/'+str(DRUG)+'/']:
                        if not os.path.exists(path):
                            # Create a new directory because it does not exist
                            os.makedirs(path)
                            print("The new directory is created!")
                    para = str(DRUG)+"_batch_size_"+str(self.batch_size)+"_source_batch_"+str(source_batch)+"_target_batch_"+str(target_batch)+"_sam_"+str(sampler)+"_lambda_recon_"+str(self.lambda_recon)+"_lambda_kl_"+str(self.lambda_kl)+"_lambda_ot_"+str(self.lambda_ot)+"_lambda_response_"+str(self.lambda_response)
                    # TODO 搜索最优的过程中，先把这段代码注释
                    # adatas[0].write("./drug/"+str(DRUG)+'/'+"bulk_"+para+".h5ad")
                    # adatas[1].write("./drug/"+str(DRUG)+'/'+"sc_"+para+".h5ad")
                    # adata_cm.write("./drug/"+str(DRUG)+'/'+"adata_cm_"+para+".h5ad")
                    ####TODOq Save adata ####END
                    # output.append(z.detach().cpu())
                # output = torch.cat(output).numpy()
                AUC_bulk_test_avg = AUC_bulk_test/count_a
                APR_bulk_test_avg = APR_bulk_test/count_a

                AUC_sc_test_avg = AUC_sc_test/count_b
                APR_sc_test_avg = APR_sc_test/count_b
                #bulk预测
                print(f'AUC_bulk_test_avg==={AUC_bulk_test_avg},APR_bulk_test_avg==={APR_bulk_test_avg}')
                print(f'bulk_best_threshold==={bulk_best_threshold},bulk_best_f1_score==={bulk_best_f1_score}')
                #sc预测
                print(f'AUC_sc_test_avg==={AUC_sc_test_avg},APR_sc_test_avg==={APR_sc_test_avg}')
                print(f'sc_best_threshold==={sc_best_threshold},sc_best_f1_score==={sc_best_f1_score}')
                
                
                # 保存性能指标
                # 获取当前进程的 PID
                current_pid = os.getpid()
                # 使用 psutil 获取当前进程的信息
                current_process = psutil.Process(current_pid)
                # 获取命令行参数
                command_line = " ".join(current_process.cmdline())
                now=time.strftime("%Y-%m-%d-%H-%M-%S")
                file = './drug/'+str(DRUG)+'/'+str(DRUG)+'_auc_apr.txt'
                with open(file, 'a+') as f:
                    f.write('AUC_bulk_test_avg==='+str(AUC_bulk_test_avg)+'\t'+
                            'APR_bulk_test_avg==='+str(APR_bulk_test_avg)+'\t'+
                            'bulk_best_threshold==='+str(bulk_best_threshold)+'\t'+
                            'bulk_best_f1_score==='+str(bulk_best_f1_score)+'\t'+
                            'AUC_sc_test_avg='+str(AUC_sc_test_avg)+'\t'+
                            'APR_sc_test_avg='+str(APR_sc_test_avg)+'\t'
                            'sc_best_threshold='+str(sc_best_threshold)+'\t'+
                            'sc_best_f1_score='+str(sc_best_f1_score)+'\t'+str(now)+'\t'+'\t'+str(command_line)+'\t'+'\n')
                
                df = pd.DataFrame(columns=['drug', 'encoder_dim', 'decoder_dim', 'pdim','batch_size','source_batch','target_batch','sam','lambda_recon','lambda_kl','lambda_ot','lambda_response','sc_AUC', 'sc_AUPR','bulk_AUC', 'bulk_AUPR'])
                file2 = './drug/'+str(DRUG)+'/'+str(DRUG)+'_auc_aupr'+'_result.xlsx'
                df_tmp=pd.DataFrame({
                    'drug': [str(DRUG)],
                    'encoder_dim': [str([str(i[1])+"," for i in self.enc[0]])],
                    'decoder_dim': [str([str(i[1])+"," for i in self.dec[0]])],
                    'pdim': [str([str(i[1])+"," for i in self.dec[3]])],
                    'batch_size': [str(self.batch_size)],
                    'source_batch': [str(source_batch)],
                    'target_batch': [str(target_batch)],
                    'sam': [str(sampler)],
                    'lambda_recon': [str(self.lambda_recon)],
                    'lambda_kl': [str(self.lambda_kl)],
                    'lambda_ot': [str(self.lambda_ot)],
                    'lambda_response': [str(self.lambda_response)],
                    'sc_AUC': [str(AUC_sc_test_avg)],
                    'sc_AUPR': [str(APR_sc_test_avg)],
                    'bulk_AUC': [str(AUC_bulk_test_avg)],#TO DO
                    'bulk_AUPR': [str(APR_bulk_test_avg)],#TO DO
                })
                if not os.path.isfile(file2):
                    # 将数据添加到DataFrame的一行中
                    #df = df.append(df_tmp, ignore_inde=True)
                    df = pd.concat([df, df_tmp], axis=0)
                    #保存
                    #df.to_csv(file2,sep='\t',index=False)
                    df.to_excel(file2,index=False)
                else:
                    df = pd.read_excel(file2)
                    #df = df.append(df_tmp, ignore_index=True)
                    df = pd.concat([df, df_tmp], axis=0)
                    #保存
                    #df.to_csv(file2,mode='a',sep='\t',index=False,header=False)
                    df.to_excel(file2,index=False)
        elif out == 'predict':
            count_a=0
            count_b=0
            bulk_best_threshold = 0.0
            bulk_best_f1_score = 0.0
            sc_best_threshold = 0.0
            sc_best_f1_score = 0.0
            output = np.zeros((num_cell[0]+num_cell[1], self.z_dim)) #output.shape=(22518, 16)
            for i,data in enumerate(dataloader): #=x.shape=torch.Size([11259, 4000])
                output = np.zeros((data[0][0].shape[0]+data[1][0].shape[0], self.z_dim)) #output.shape=(22518, 16)
                #print(f'####vae.py#449rows########count_a={count_a},count_b={count_b}')
                # 1. batch里 的bulk数据
                x_bulk = data[0][0].float().to(device)
                response_bulk = data[0][1].float().to(device)
                y_bulk = torch.tensor([0] * data[0][0].shape[0]).float().to(device) # bulk数据的domain_id设置为0
                idx_bulk = data[0][2]
                # 2. batch里 的sc数据
                x_sc = data[1][0].float().to(device)
                response_sc = data[1][1].float().to(device)
                y_sc = torch.tensor([1] * data[1][0].shape[0]).float().to(device) # sc数据的domain_id设置为1
                idx_sc = (data[1][2] + num_cell[0])
                # 使用 torch.cat() 函数在第一个维度上进行连接
                x = torch.cat((x_bulk, x_sc), dim=0)
                y = torch.cat((y_bulk, y_sc), dim=0)
                idx = torch.cat((idx_bulk, idx_sc), dim=0)
                response = torch.cat((response_bulk, response_sc), dim=0)

                x_c = x.float().to(device)
                #TODOq，@see：vae.py, 462rows, z,mu,var = self.encoder(x_c, 0)
                if unshared_encoder: #TODOq 分别使用两个不同的编码器
                    z_bulk = self.encoder(x_bulk, 3)[1]
                    z_sc = self.encoder(x_sc, 4)[1]
                    z = torch.cat((z_bulk, z_sc), dim=0)
                else: #TODOq 使用共享编码器
                    z = self.encoder(x_c, 0)[1] #？？？这里竟然取得是mu，不重采样吗，
                    
                adatas[0].obsm['latent'] = z[0:num_cell[0]].detach().cpu().numpy()
                adatas[1].obsm['latent'] = z[num_cell[0]:num_cell[0]+num_cell[1]].detach().cpu().numpy()
                adata_cm.obsm["latent"]=z.detach().cpu().numpy()
                loc = {}
                loc[0] = torch.where(y==0)[0]
                loc[1] = torch.where(y==1)[0]
                if adata_cm is not None:
                    if len(loc[0])>0: # 表示取数据集x的隐空间嵌入z（取bulk细胞系的隐空间嵌入z ）
                        count_a+=1
                        groundtruth_bulk_label = response_bulk.reshape(-1,1)
                        groundtruth_bulk_label = torch.Tensor(groundtruth_bulk_label).to(device)
                        groundtruth_bulk_label = groundtruth_bulk_label.detach().cpu().numpy()
                        if unshared_encoder: #TODOq 分别使用两个不同的编码器
                            predicted_bulk_label = self.decoder(z_bulk, 0+2+1) # decoder(,0)表示所有共同数据的共同的解码器；decoder(,1)表示数据集X特异性高可变基因的解码器；decoder(,2)表示数据集Y特异性高可变基因的解码器；decoder(,3)表示所有共同数据里的数据X的药物响应解码器；
                        else:
                            predicted_bulk_label = self.decoder(z[loc[0]], 0+2+1) # decoder(,0)表示所有共同数据的共同的解码器；decoder(,1)表示数据集X特异性高可变基因的解码器；decoder(,2)表示数据集Y特异性高可变基因的解码器；decoder(,3)表示所有共同数据里的数据X的药物响应解码器；
                        predicted_bulk_label = predicted_bulk_label.detach().cpu().numpy()
                        adatas[0].obs["sensitivity_prediction"] = predicted_bulk_label
                        tmp_1 = roc_auc_score(groundtruth_bulk_label, predicted_bulk_label)
                        AUC_bulk_test += tmp_1 # AUC
                        #
                        tmp_2 = average_precision_score(groundtruth_bulk_label, predicted_bulk_label)
                        APR_bulk_test += tmp_2
                        # 计算每个阈值下的性能指标
                        precision, recall, thresholds_pr = precision_recall_curve(groundtruth_bulk_label, predicted_bulk_label)
                        # 计算每个阈值下的 F1 分数
                        f1_scores = 2 * (precision * recall) / (precision + recall)
                        # 找到达到最佳性能的阈值对应的索引
                        best_threshold_idx = np.argmax(f1_scores)
                        # 获取最佳阈值和对应的性能指标
                        bulk_best_threshold = thresholds_pr[best_threshold_idx] #TODOa,这里可以根据阈值的选择，大于阈值为标签1，小于阈值的为标签0，把二值化的预测标签存入adatas[0].obs["sensitivity_prediction_label"]
                        bulk_best_f1_score = f1_scores[best_threshold_idx]
                    if len(loc[1])>0: # 表示取数据集x的隐空间嵌入z（取sc细胞系的隐空间嵌入z ）
                        count_b+=1
                        groundtruth_sc_label = response_sc.reshape(-1,1)
                        groundtruth_sc_label = torch.Tensor(groundtruth_sc_label).to(device)
                        groundtruth_sc_label = groundtruth_sc_label.detach().cpu().numpy()                            
                        if unshared_encoder:
                            predicted_sc_label = self.decoder(z_sc, 0+2+1) # decoder(,0)表示所有共同数据的共同的解码器；decoder(,1)表示数据集X特异性高可变基因的解码器；decoder(,2)表示数据集Y特异性高可变基因的解码器；decoder(,3)表示所有共同数据里的数据X的药物响应解码器；
                        else:
                            predicted_sc_label = self.decoder(z[loc[1]], 0+2+1) # decoder(,0)表示所有共同数据的共同的解码器；decoder(,1)表示数据集X特异性高可变基因的解码器；decoder(,2)表示数据集Y特异性高可变基因的解码器；decoder(,3)表示所有共同数据里的数据X的药物响应解码器；
                        predicted_sc_label = predicted_sc_label.detach().cpu().numpy()
                        adatas[1].obs["sensitivity_prediction"] = predicted_sc_label #TODO new feature
                        tmp_3 = roc_auc_score(groundtruth_sc_label, predicted_sc_label)
                        AUC_sc_test += tmp_3 # AUC
                        #
                        tmp_4 = average_precision_score(groundtruth_sc_label, predicted_sc_label)
                        APR_sc_test += tmp_4
                        # 计算每个阈值下的性能指标
                        precision, recall, thresholds_pr = precision_recall_curve(groundtruth_sc_label, predicted_sc_label)
                        # 计算每个阈值下的 F1 分数
                        f1_scores = 2 * (precision * recall) / (precision + recall)
                        # 找到达到最佳性能的阈值对应的索引
                        best_threshold_idx = np.argmax(f1_scores)
                        # 获取最佳阈值和对应的性能指标
                        sc_best_threshold = thresholds_pr[best_threshold_idx] #TODOa,这里可以根据阈值的选择，大于阈值为标签1，小于阈值的为标签0，把二值化的预测标签存入adatas[1].obs["sensitivity_prediction_label"]
                        sc_best_f1_score = f1_scores[best_threshold_idx]
                output[idx] = z.detach().cpu().numpy()
                adata_cm.obsm["latent_output"] = output #TODO,new feature
                ############ Save adata ############START
                # Create directories if they do not exist
                for path in ['./drug/'+str(DRUG)+'/']:
                    if not os.path.exists(path):
                        # Create a new directory because it does not exist
                        os.makedirs(path)
                        print("The new directory is created!")
                para = str(DRUG)+"_batch_size_"+str(self.batch_size)+"_source_batch_"+str(source_batch)+"_target_batch_"+str(target_batch)+"_sam_"+str(sampler)+"_lambda_recon_"+str(self.lambda_recon)+"_lambda_kl_"+str(self.lambda_kl)+"_lambda_ot_"+str(self.lambda_ot)+"_lambda_response_"+str(self.lambda_response)
                # TODO 搜索最优的过程中，先把这段代码注释
                adatas[0].write("./drug/"+str(DRUG)+'/'+"bulk_"+para+".h5ad")
                adatas[1].write("./drug/"+str(DRUG)+'/'+"sc_"+para+".h5ad")
                adata_cm.write("./drug/"+str(DRUG)+'/'+"adata_cm_"+para+".h5ad")
                ############TODOq Save adata ############END
                # output.append(z.detach().cpu())
            # output = torch.cat(output).numpy()
            AUC_bulk_test_avg = AUC_bulk_test/count_a
            APR_bulk_test_avg = APR_bulk_test/count_a

            AUC_sc_test_avg = AUC_sc_test/count_b
            APR_sc_test_avg = APR_sc_test/count_b
            #bulk预测
            print(f'AUC_bulk_test_avg==={AUC_bulk_test_avg},APR_bulk_test_avg==={APR_bulk_test_avg}')
            print(f'bulk_best_threshold==={bulk_best_threshold},bulk_best_f1_score==={bulk_best_f1_score}')
            #sc预测
            print(f'AUC_sc_test_avg==={AUC_sc_test_avg},APR_sc_test_avg==={APR_sc_test_avg}')
            print(f'sc_best_threshold==={sc_best_threshold},sc_best_f1_score==={sc_best_f1_score}')
            
            # 保存性能指标
            # 获取当前进程的 PID
            current_pid = os.getpid()
            # 使用 psutil 获取当前进程的信息
            current_process = psutil.Process(current_pid)
            # 获取命令行参数
            command_line = " ".join(current_process.cmdline())
            now=time.strftime("%Y-%m-%d-%H-%M-%S")
            file = './drug/'+str(DRUG)+'/'+str(DRUG)+'_auc_apr.txt'
            with open(file, 'a+') as f:
                f.write('AUC_bulk_test_avg==='+str(AUC_bulk_test_avg)+'\t'+
                        'APR_bulk_test_avg==='+str(APR_bulk_test_avg)+'\t'+
                        'bulk_best_threshold==='+str(bulk_best_threshold)+'\t'+
                        'bulk_best_f1_score==='+str(bulk_best_f1_score)+'\t'+
                        'AUC_sc_test_avg='+str(AUC_sc_test_avg)+'\t'+
                        'APR_sc_test_avg='+str(APR_sc_test_avg)+'\t'
                        'sc_best_threshold='+str(sc_best_threshold)+'\t'+
                        'sc_best_f1_score='+str(sc_best_f1_score)+'\t'+str(now)+'\t'+'\t'+str(command_line)+'\t'+'\n')
            
            df = pd.DataFrame(columns=['drug', 'encoder_dim', 'decoder_dim', 'pdim','batch_size','source_batch','target_batch','sam','lambda_recon','lambda_kl','lambda_ot','lambda_response','sc_AUC', 'sc_AUPR','bulk_AUC', 'bulk_AUPR'])
            file2 = './drug/'+str(DRUG)+'/'+str(DRUG)+'_auc_aupr'+'_result.xlsx'
            df_tmp=pd.DataFrame({
                'drug': [str(DRUG)],
                'encoder_dim': [str([str(i[1])+"," for i in self.enc[0]])],
                'decoder_dim': [str([str(i[1])+"," for i in self.dec[0]])],
                'pdim': [str([str(i[1])+"," for i in self.dec[3]])],
                'batch_size': [str(self.batch_size)],
                'source_batch': [str(source_batch)],
                'target_batch': [str(target_batch)],
                'sam': [str(sampler)],
                'lambda_recon': [str(self.lambda_recon)],
                'lambda_kl': [str(self.lambda_kl)],
                'lambda_ot': [str(self.lambda_ot)],
                'lambda_response': [str(self.lambda_response)],
                'sc_AUC': [str(AUC_sc_test_avg)],
                'sc_AUPR': [str(APR_sc_test_avg)],
                'bulk_AUC': [str(AUC_bulk_test_avg)],#TO DO
                'bulk_AUPR': [str(APR_bulk_test_avg)],#TO DO
            })
            if not os.path.isfile(file2):
                # 将数据添加到DataFrame的一行中
                df = pd.concat([df, df_tmp], axis=0)
                #保存
                df.to_excel(file2,index=False)
            else:
                df = pd.read_excel(file2)
                df = pd.concat([df, df_tmp], axis=0)
                #保存
                df.to_excel(file2,index=False)
            
        torch.cuda.empty_cache()
        return 

    def encodeBatch_1( # 未对源域bulk数据平衡采样时，使用这个
            self, 
            adata_cm, ###TODO,new feature
            adatas, ###TODO,new feature
            dataloader, 
            num_gene,
            num_cell,#TODO new feature
            mode='h',  
            out='latent', 
            batch_id=0,
            pred_id=1,
            device='cuda', 
            eval=False,
            DRUG='Gefitinib', ###TODO,new feature
            source_batch=0, ###TODO,new feature
            target_batch=0, ###TODO,new feature
            sampler="none", ###TODO,new feature
            unshared_encoder=False, ###TODO,new feature
        ):
        """
        Inference

        Parameters
        ----------
        dataloader
            An iterable over the given dataset for inference.
        num_gene
            List of number of genes in different datasets. 
        mode
            Choose from ['h', 'v', 'd']
            If 'h', integrate data with common genes
            If 'v', integrate data profiled from the same cells
            If 'd', inetrgate data without common genes
            Default: 'h'.
        out
            Output of uniPort. Choose from ['latent', 'project', 'predict']. 
            If out='latent', train the network and output cell embeddings. 
            If out='project', project data into the latent space and output cell embeddings. 
            If out='predict', project data into the latent space and output cell embeddings through a specified decoder. 
            Default: 'latent'.
        batch_id
            Choose which encoder to project data when mode='d'. Default: 0.
        pred_id
            Choose which decoder to reconstruct data when out='predict'.
        device
            'cuda' or 'cpu' for . Default: 'cuda'.
        eval
            If True, set the model to evaluation mode. If False, set the model to train mode. Default: False.
        
        Returns
        -------
        output
            Cell embeddings (if out='latent' or 'project') or Predicted data (if out='predict').
        """

        self.to(device)
        if eval:
            self.eval()
        else:
            self.train()
        AUC_bulk_test=0.0
        APR_bulk_test=0.0
        AUC_sc_test=0.0
        APR_sc_test=0.0
        # indices = np.zeros(dataloader.dataset.shape[0])
        output = []
        if out == 'latent':
            output = np.zeros((dataloader.dataset.shape[0], self.z_dim))

            if mode == 'v':
                for x, idx in dataloader:
                    x = x.float().to(device)
                    z = self.encoder(x[:, 0:num_gene[0]], 0)[1]
                    output.append(z.detach().cpu())
                output = torch.cat(output).numpy()

            elif mode == 'd':
                for x, y,idx in dataloader:
                    x, y = x[:, 0:num_gene[batch_id]].float().to(device), y.long().to(device)
                    loc = torch.where(y==batch_id)[0]
                    x = x[loc]
                    z = self.encoder(x, batch_id)[1] # z, mu, var
                    output.append(z.detach().cpu())
                output = torch.cat(output).numpy()

            elif mode == 'h':
                count_a=0
                count_b=0
                bulk_best_threshold = 0.0
                bulk_best_f1_score = 0.0
                sc_best_threshold = 0.0
                sc_best_f1_score = 0.0
                output = np.zeros((dataloader.dataset.shape[0], self.z_dim)) #output.shape=(22518, 16)
                
                for x,y,idx in dataloader: #=x.shape=torch.Size([11259, 4000])
                    print(f'####vae.py#162rows########count_a={count_a},count_b={count_b},dataloader.dataset.shape={dataloader.dataset.shape}')
                    #TODO 这里对于x，y，idx是什么,疑问？
                    # print(f'vae.py#146行###########x,y,idx in dataloader:===x.shape={x.shape},y.shape={y.shape},idx.shape={idx.shape},output.shape={output.shape}')
                    x_c = x[:, 0:num_gene[self.n_domain]].float().to(device)
                    #TODO，@see：vae.py, 462rows, z,mu,var = self.encoder(x_c, 0)
                    z = self.encoder(x_c, 0)[1] #？？？这里竟然取得是mu，不重采样吗，
                    # z = self.encoder(x_c, 0)[0] #？？？这里竟然取得是mu，不重采样吗，
                    # print(f'####vae.py 558row##, z,mu,var = self.encoder(x_c, 0) 测试隐空间数据取z')
                    # adatas= [data_bulk_adata, data_sc_adata]，
                    # bulk数据的隐空间嵌入z[0:num_cell[0]]放到data_bulk_adata.obsm['latent'],
                    # sc数据的隐空间嵌入z[num_cell[0]+1:-1]放到data_sc_adata.obsm['latent']
                    if unshared_encoder: #TODO, 分别使用两个编码器
                        x_c = x[:, :].float().to(device)
                        loc_bulk = torch.where(y==0)[0]
                        loc_sc = torch.where(y==1)[0]
                        # 把x拆包分成x_bulk和x_sc数据
                        x_bulk = x_c[loc_bulk, 0:num_gene[0]]
                        x_sc = x_c[loc_sc, 0:num_gene[1]]
                        # 分别使用两个编码器
                        z_bulk, mu_bulk, var_bulk = self.encoder(x_bulk, 3)
                        z_sc, mu_sc, var_sc = self.encoder(x_sc, 4)
                        # 把分别得到的两种隐空间数据部分，拼到一起
                        z = torch.zeros(size=(z_bulk.shape[0]+z_sc.shape[0], z_bulk.shape[1]), device=device)
                        mu = torch.zeros(size=(mu_bulk.shape[0]+mu_sc.shape[0], mu_bulk.shape[1]), device=device)
                        var = torch.zeros(size=(var_bulk.shape[0]+var_sc.shape[0], var_bulk.shape[1]), device=device)
                        # 使用 loc_bulk 索引将 z_bulk 填充到 z
                        z[loc_bulk] = z_bulk
                        mu[loc_bulk] = mu_bulk
                        var[loc_bulk] = var_bulk
                        # 使用 loc_sc 索引将 z_sc 填充到 z
                        z[loc_sc] = z_sc
                        mu[loc_sc] = mu_sc
                        var[loc_sc] = var_sc
                    
                    adatas[0].obsm['latent'] = z[0:num_cell[0]].detach().cpu().numpy()
                    adatas[1].obsm['latent'] = z[num_cell[0]:num_cell[0]+num_cell[1]].detach().cpu().numpy()
                    adata_cm.obsm["latent"]=z.detach().cpu().numpy()
                    #z = self.encoder(x_c, 0)[0]
                    loc = {}
                    loc[0] = torch.where(y==0)[0]
                    loc[1] = torch.where(y==1)[0]
                    if adata_cm is not None:
                        if len(loc[0])>0: # 表示取数据集x的隐空间嵌入z（取bulk细胞系的隐空间嵌入z ）
                            count_a+=1
                            a = idx[y==0].tolist() #定位到数据集x（bulk细胞系）的索引，用来拿到其相应的response label
                            groundtruth_bulk_label = adata_cm.obs['response'].iloc[a,].values.reshape(-1,1)
                            groundtruth_bulk_label = torch.Tensor(groundtruth_bulk_label).to(device)
                            groundtruth_bulk_label = groundtruth_bulk_label.detach().cpu().numpy()
                            predicted_bulk_label = self.decoder(z[loc[0]], 0+2+1) # decoder(,0)表示所有共同数据的共同的解码器；decoder(,1)表示数据集X特异性高可变基因的解码器；decoder(,2)表示数据集Y特异性高可变基因的解码器；decoder(,3)表示所有共同数据里的数据X的药物响应解码器；
                            predicted_bulk_label = predicted_bulk_label.detach().cpu().numpy()
                            adatas[0].obs["sensitivity_prediction"] = predicted_bulk_label #TODO new feature
                            #print(f'####vae.py#511rows#######predicted_bulk_label.shape={predicted_bulk_label.shape},predicted_bulk_label[:10]={predicted_bulk_label[:10]},groundtruth_bulk_label[:10]={groundtruth_bulk_label[:10]}')
                            tmp_1 = roc_auc_score(groundtruth_bulk_label, predicted_bulk_label)
                            #print(f'####vae.py#174rows####AUC_bulk_test={tmp_1}')
                            AUC_bulk_test += tmp_1 # AUC
                            #
                            tmp_2 = average_precision_score(groundtruth_bulk_label, predicted_bulk_label)
                            #print(f'####vae.py#177rows####APR_bulk_test={tmp_2}')
                            APR_bulk_test += tmp_2
                            # 计算每个阈值下的性能指标
                            precision, recall, thresholds_pr = precision_recall_curve(groundtruth_bulk_label, predicted_bulk_label)
                            # 计算每个阈值下的 F1 分数
                            f1_scores = 2 * (precision * recall) / (precision + recall)
                            # 找到达到最佳性能的阈值对应的索引
                            best_threshold_idx = np.argmax(f1_scores)
                            # 获取最佳阈值和对应的性能指标
                            bulk_best_threshold = thresholds_pr[best_threshold_idx] #TODO,这里可以根据阈值的选择，大于阈值为标签1，小于阈值的为标签0，把二值化的预测标签存入adatas[0].obs["sensitivity_prediction_label"]
                            bulk_best_f1_score = f1_scores[best_threshold_idx]
                        if len(loc[1])>0: # 表示取数据集x的隐空间嵌入z（取bulk细胞系的隐空间嵌入z ）
                            count_b+=1
                            b = idx[y==1].tolist() #定位到数据集x（bulk细胞系）的索引，用来拿到其相应的response label
                            groundtruth_sc_label = adata_cm.obs['response'].iloc[b,].values.reshape(-1,1)
                            groundtruth_sc_label = torch.Tensor(groundtruth_sc_label).to(device)
                            groundtruth_sc_label = groundtruth_sc_label.detach().cpu().numpy()
                            predicted_sc_label = self.decoder(z[loc[1]], 0+2+1) # decoder(,0)表示所有共同数据的共同的解码器；decoder(,1)表示数据集X特异性高可变基因的解码器；decoder(,2)表示数据集Y特异性高可变基因的解码器；decoder(,3)表示所有共同数据里的数据X的药物响应解码器；
                            predicted_sc_label = predicted_sc_label.detach().cpu().numpy()
                            adatas[1].obs["sensitivity_prediction"] = predicted_sc_label #TODO new feature
                            #print(f'####vae.py#511rows#######predicted_sc_label.shape={predicted_sc_label.shape},predicted_sc_label[:10]={predicted_sc_label[:10]},groundtruth_sc_label[:10]={groundtruth_sc_label[:10]}')
                            tmp_3 = roc_auc_score(groundtruth_sc_label, predicted_sc_label)
                            #print(f'####vae.py#189rows####AUC_sc_test={tmp_3}')
                            AUC_sc_test += tmp_3 # AUC
                            #
                            tmp_4 = average_precision_score(groundtruth_sc_label, predicted_sc_label)
                            #print(f'####vae.py#192rows####APR_sc_test={tmp_4}')
                            APR_sc_test += tmp_4
                            #AUC_sc_test += roc_auc_score(groundtruth_sc_label, predicted_sc_label) # AUC
                            #APR_sc_test += average_precision_score(groundtruth_sc_label, predicted_sc_label) # Precision Recall
                            precision, recall, thresholds_pr = precision_recall_curve(groundtruth_sc_label, predicted_sc_label)
                            # 计算每个阈值下的 F1 分数
                            f1_scores = 2 * (precision * recall) / (precision + recall)
                            # 找到达到最佳性能的阈值对应的索引
                            best_threshold_idx = np.argmax(f1_scores)
                            # 获取最佳阈值和对应的性能指标
                            sc_best_threshold = thresholds_pr[best_threshold_idx] #TODO,这里可以根据阈值的选择，大于阈值为标签1，小于阈值的为标签0，把二值化的预测标签存入adatas[1].obs["sensitivity_prediction_label"]
                            sc_best_f1_score = f1_scores[best_threshold_idx]
                    output[idx] = z.detach().cpu().numpy()
                    adata_cm.obsm["latent_output"] = output #TODO,new feature
                    ####TODO Save adata ####START
                    # Create directories if they do not exist
                    for path in ['./drug/'+str(DRUG)+'/']:
                        if not os.path.exists(path):
                            # Create a new directory because it does not exist
                            os.makedirs(path)
                            print("The new directory is created!")
                    # para = str(DRUG)+"_encoder_dim_"+str([str(i[1])+"," for i in self.enc])+"_batch_size_"+str(self.batch_size)+"_source_batch_"+str(source_batch)+"_target_batch_"+str(target_batch)+"_sam_"+str(sampler)+"_lambda_recon_"+str(self.lambda_recon)+"_lambda_kl_"+str(self.lambda_kl)+"_lambda_ot_"+str(self.lambda_ot)+"_lambda_response_"+str(self.lambda_response)
                    para = str(DRUG)+"_batch_size_"+str(self.batch_size)+"_source_batch_"+str(source_batch)+"_target_batch_"+str(target_batch)+"_sam_"+str(sampler)+"_lambda_recon_"+str(self.lambda_recon)+"_lambda_kl_"+str(self.lambda_kl)+"_lambda_ot_"+str(self.lambda_ot)+"_lambda_response_"+str(self.lambda_response)
                    # TODO 搜索最优的过程中，先把这段代码注释
                    # adatas[0].write("./drug/"+str(DRUG)+'/'+"bulk_"+para+".h5ad")
                    # adatas[1].write("./drug/"+str(DRUG)+'/'+"sc_"+para+".h5ad")
                    # adata_cm.write("./drug/"+str(DRUG)+'/'+"adata_cm_"+para+".h5ad")
                    ####TODOq Save adata ####END
                # output = torch.cat(output).numpy()
                AUC_bulk_test_avg = AUC_bulk_test/count_a
                APR_bulk_test_avg = APR_bulk_test/count_a

                AUC_sc_test_avg = AUC_sc_test/count_b
                APR_sc_test_avg = APR_sc_test/count_b
                #bulk预测
                print(f'AUC_bulk_test_avg==={AUC_bulk_test_avg},APR_bulk_test_avg==={APR_bulk_test_avg}')
                print(f'bulk_best_threshold==={bulk_best_threshold},bulk_best_f1_score==={bulk_best_f1_score}')
                #sc预测
                print(f'AUC_sc_test_avg==={AUC_sc_test_avg},APR_sc_test_avg==={APR_sc_test_avg}')
                print(f'sc_best_threshold==={sc_best_threshold},sc_best_f1_score==={sc_best_f1_score}')
                
                
                # 保存性能指标
                # 获取当前进程的 PID
                current_pid = os.getpid()
                # 使用 psutil 获取当前进程的信息
                current_process = psutil.Process(current_pid)
                # 获取命令行参数
                command_line = " ".join(current_process.cmdline())
                now=time.strftime("%Y-%m-%d-%H-%M-%S")
                file = './drug/'+str(DRUG)+'/'+str(DRUG)+'_auc_apr.txt'
                with open(file, 'a+') as f:
                    f.write('AUC_bulk_test_avg==='+str(AUC_bulk_test_avg)+'\t'+
                            'APR_bulk_test_avg==='+str(APR_bulk_test_avg)+'\t'+
                            'bulk_best_threshold==='+str(bulk_best_threshold)+'\t'+
                            'bulk_best_f1_score==='+str(bulk_best_f1_score)+'\t'+
                            'AUC_sc_test_avg='+str(AUC_sc_test_avg)+'\t'+
                            'APR_sc_test_avg='+str(APR_sc_test_avg)+'\t'
                            'sc_best_threshold='+str(sc_best_threshold)+'\t'+
                            'sc_best_f1_score='+str(sc_best_f1_score)+'\t'+str(now)+'\t'+str(command_line)+'\n')
                
                df = pd.DataFrame(columns=['drug', 'encoder_dim', 'decoder_dim', 'pdim','batch_size','source_batch','target_batch','sam','lambda_recon','lambda_kl','lambda_ot','lambda_response','sc_AUC', 'sc_AUPR','bulk_AUC', 'bulk_AUPR'])
                file2 = './drug/'+str(DRUG)+'/'+str(DRUG)+'_auc_aupr'+'_result.xlsx'
                df_tmp=pd.DataFrame({
                    'drug': [str(DRUG)],
                    'encoder_dim': [str([str(i[1])+"," for i in self.enc[0]])],
                    'decoder_dim': [str([str(i[1])+"," for i in self.dec[0]])],
                    'pdim': [str([str(i[1])+"," for i in self.dec[3]])],
                    'batch_size': [str(self.batch_size)],
                    'source_batch': [str(source_batch)],
                    'target_batch': [str(target_batch)],
                    'sam': [str(sampler)],
                    'lambda_recon': [str(self.lambda_recon)],
                    'lambda_kl': [str(self.lambda_kl)],
                    'lambda_ot': [str(self.lambda_ot)],
                    'lambda_response': [str(self.lambda_response)],
                    'sc_AUC': [str(AUC_sc_test_avg)],
                    'sc_AUPR': [str(APR_sc_test_avg)],
                    'bulk_AUC': [str(AUC_bulk_test_avg)],#TO DO
                    'bulk_AUPR': [str(APR_bulk_test_avg)],#TO DO
                })
                if not os.path.isfile(file2):
                    # 将数据添加到DataFrame的一行中
                    #df = df.append(df_tmp, ignore_index=True)
                    df = pd.concat([df, df_tmp], axis=0)
                    #保存
                    #df.to_csv(file2,sep='\t',index=False)
                    df.to_excel(file2,index=False)
                else:
                    df = pd.read_excel(file2)
                    #df = df.append(df_tmp, ignore_index=True)
                    df = pd.concat([df, df_tmp], axis=0)
                    #保存
                    #df.to_csv(file2,mode='a',sep='\t',index=False,header=False)
                    df.to_excel(file2,index=False)
        elif out == 'predict':
            count_a=0
            count_b=0
            bulk_best_threshold = 0.0
            bulk_best_f1_score = 0.0
            sc_best_threshold = 0.0
            sc_best_f1_score = 0.0
            output = np.zeros((dataloader.dataset.shape[0], self.z_dim)) #output.shape=(22518, 16)
            for x,y,idx in dataloader: #=x.shape=torch.Size([11259, 4000])
                print(f'####vae.py#896rows########count_a={count_a},count_b={count_b},dataloader.dataset.shape={dataloader.dataset.shape}')
                x_c = x[:, 0:num_gene[self.n_domain]].float().to(device)
                #TODOa，@see：vae.py, 462rows, z,mu,var = self.encoder(x_c, 0)
                z = self.encoder(x_c, 0)[1] #？？？这里竟然取得是mu，不重采样吗，
                # bulk数据的隐空间嵌入z[0:num_cell[0]]放到data_bulk_adata.obsm['latent'],
                # sc数据的隐空间嵌入z[num_cell[0]+1:-1]放到data_sc_adata.obsm['latent']
                if unshared_encoder: #TODOa, 分别使用两个编码器
                    x_c = x[:, :].float().to(device)
                    loc_bulk = torch.where(y==0)[0]
                    loc_sc = torch.where(y==1)[0]
                    # 把x拆包分成x_bulk和x_sc数据
                    x_bulk = x_c[loc_bulk, 0:num_gene[0]]
                    x_sc = x_c[loc_sc, 0:num_gene[1]]
                    # 分别使用两个编码器
                    z_bulk, mu_bulk, var_bulk = self.encoder(x_bulk, 3)
                    z_sc, mu_sc, var_sc = self.encoder(x_sc, 4)
                    # 把分别得到的两种隐空间数据部分，拼到一起
                    z = torch.zeros(size=(z_bulk.shape[0]+z_sc.shape[0], z_bulk.shape[1]), device=device)
                    mu = torch.zeros(size=(mu_bulk.shape[0]+mu_sc.shape[0], mu_bulk.shape[1]), device=device)
                    var = torch.zeros(size=(var_bulk.shape[0]+var_sc.shape[0], var_bulk.shape[1]), device=device)
                    # 使用 loc_bulk 索引将 z_bulk 填充到 z
                    z[loc_bulk] = z_bulk
                    mu[loc_bulk] = mu_bulk
                    var[loc_bulk] = var_bulk
                    # 使用 loc_sc 索引将 z_sc 填充到 z
                    z[loc_sc] = z_sc
                    mu[loc_sc] = mu_sc
                    var[loc_sc] = var_sc
                
                adatas[0].obsm['latent'] = z[0:num_cell[0]].detach().cpu().numpy()
                adatas[1].obsm['latent'] = z[num_cell[0]:num_cell[0]+num_cell[1]].detach().cpu().numpy()
                adata_cm.obsm["latent"]=z.detach().cpu().numpy()
                loc = {}
                loc[0] = torch.where(y==0)[0]
                loc[1] = torch.where(y==1)[0]
                if adata_cm is not None:
                    if len(loc[0])>0: # 表示取数据集x的隐空间嵌入z（取bulk细胞系的隐空间嵌入z ）
                        count_a+=1
                        a = idx[y==0].tolist() #定位到数据集x（bulk细胞系）的索引，用来拿到其相应的response label
                        groundtruth_bulk_label = adata_cm.obs['response'].iloc[a,].values.reshape(-1,1)
                        groundtruth_bulk_label = torch.Tensor(groundtruth_bulk_label).to(device)
                        groundtruth_bulk_label = groundtruth_bulk_label.detach().cpu().numpy()
                        predicted_bulk_label = self.decoder(z[loc[0]], 0+2+1) # decoder(,0)表示所有共同数据的共同的解码器；decoder(,1)表示数据集X特异性高可变基因的解码器；decoder(,2)表示数据集Y特异性高可变基因的解码器；decoder(,3)表示所有共同数据里的数据X的药物响应解码器；
                        predicted_bulk_label = predicted_bulk_label.detach().cpu().numpy()
                        adatas[0].obs["sensitivity_prediction"] = predicted_bulk_label
                        tmp_1 = roc_auc_score(groundtruth_bulk_label, predicted_bulk_label)
                        AUC_bulk_test += tmp_1 # AUC
                        #
                        tmp_2 = average_precision_score(groundtruth_bulk_label, predicted_bulk_label)
                        APR_bulk_test += tmp_2
                        # 计算每个阈值下的性能指标
                        precision, recall, thresholds_pr = precision_recall_curve(groundtruth_bulk_label, predicted_bulk_label)
                        # 计算每个阈值下的 F1 分数
                        f1_scores = 2 * (precision * recall) / (precision + recall)
                        # 找到达到最佳性能的阈值对应的索引
                        best_threshold_idx = np.argmax(f1_scores)
                        # 获取最佳阈值和对应的性能指标
                        bulk_best_threshold = thresholds_pr[best_threshold_idx] #TODOa,这里可以根据阈值的选择，大于阈值为标签1，小于阈值的为标签0，把二值化的预测标签存入adatas[0].obs["sensitivity_prediction_label"]
                        bulk_best_f1_score = f1_scores[best_threshold_idx]
                    if len(loc[1])>0: # 表示取数据集x的隐空间嵌入z（取bulk细胞系的隐空间嵌入z ）
                        count_b+=1
                        b = idx[y==1].tolist() #定位到数据集x（bulk细胞系）的索引，用来拿到其相应的response label
                        groundtruth_sc_label = adata_cm.obs['response'].iloc[b,].values.reshape(-1,1)
                        groundtruth_sc_label = torch.Tensor(groundtruth_sc_label).to(device)
                        groundtruth_sc_label = groundtruth_sc_label.detach().cpu().numpy()
                        predicted_sc_label = self.decoder(z[loc[1]], 0+2+1) # decoder(,0)表示所有共同数据的共同的解码器；decoder(,1)表示数据集X特异性高可变基因的解码器；decoder(,2)表示数据集Y特异性高可变基因的解码器；decoder(,3)表示所有共同数据里的数据X的药物响应解码器；
                        predicted_sc_label = predicted_sc_label.detach().cpu().numpy()
                        adatas[1].obs["sensitivity_prediction"] = predicted_sc_label
                        tmp_3 = roc_auc_score(groundtruth_sc_label, predicted_sc_label)
                        AUC_sc_test += tmp_3 # AUC
                        #
                        tmp_4 = average_precision_score(groundtruth_sc_label, predicted_sc_label)
                        APR_sc_test += tmp_4
                        precision, recall, thresholds_pr = precision_recall_curve(groundtruth_sc_label, predicted_sc_label)
                        # 计算每个阈值下的 F1 分数
                        f1_scores = 2 * (precision * recall) / (precision + recall)
                        # 找到达到最佳性能的阈值对应的索引
                        best_threshold_idx = np.argmax(f1_scores)
                        # 获取最佳阈值和对应的性能指标
                        sc_best_threshold = thresholds_pr[best_threshold_idx] #TODOa,这里可以根据阈值的选择，大于阈值为标签1，小于阈值的为标签0，把二值化的预测标签存入adatas[1].obs["sensitivity_prediction_label"]
                        sc_best_f1_score = f1_scores[best_threshold_idx]
                output[idx] = z.detach().cpu().numpy()
                adata_cm.obsm["latent_output"] = output #TODO,new feature
                ############TODOa Save adata ############START
                # Create directories if they do not exist
                for path in ['./drug/'+str(DRUG)+'/']:
                    if not os.path.exists(path):
                        # Create a new directory because it does not exist
                        os.makedirs(path)
                        print("The new directory is created!")
                para = str(DRUG)+"_batch_size_"+str(self.batch_size)+"_source_batch_"+str(source_batch)+"_target_batch_"+str(target_batch)+"_sam_"+str(sampler)+"_lambda_recon_"+str(self.lambda_recon)+"_lambda_kl_"+str(self.lambda_kl)+"_lambda_ot_"+str(self.lambda_ot)+"_lambda_response_"+str(self.lambda_response)
                # TODO 搜索最优的过程中，先把这段代码注释
                adatas[0].write("./drug/"+str(DRUG)+'/'+"bulk_"+para+".h5ad")
                adatas[1].write("./drug/"+str(DRUG)+'/'+"sc_"+para+".h5ad")
                adata_cm.write("./drug/"+str(DRUG)+'/'+"adata_cm_"+para+".h5ad")
                ############TODOq Save adata ############END
            AUC_bulk_test_avg = AUC_bulk_test/count_a
            APR_bulk_test_avg = APR_bulk_test/count_a

            AUC_sc_test_avg = AUC_sc_test/count_b
            APR_sc_test_avg = APR_sc_test/count_b
            #bulk预测
            print(f'AUC_bulk_test_avg==={AUC_bulk_test_avg},APR_bulk_test_avg==={APR_bulk_test_avg}')
            print(f'bulk_best_threshold==={bulk_best_threshold},bulk_best_f1_score==={bulk_best_f1_score}')
            #sc预测
            print(f'AUC_sc_test_avg==={AUC_sc_test_avg},APR_sc_test_avg==={APR_sc_test_avg}')
            print(f'sc_best_threshold==={sc_best_threshold},sc_best_f1_score==={sc_best_f1_score}')
            
            # 保存性能指标
            # 获取当前进程的 PID
            current_pid = os.getpid()
            # 使用 psutil 获取当前进程的信息
            current_process = psutil.Process(current_pid)
            # 获取命令行参数
            command_line = " ".join(current_process.cmdline())
            now=time.strftime("%Y-%m-%d-%H-%M-%S")
            file = './drug/'+str(DRUG)+'/'+str(DRUG)+'_auc_apr.txt'
            with open(file, 'a+') as f:
                f.write('AUC_bulk_test_avg==='+str(AUC_bulk_test_avg)+'\t'+
                        'APR_bulk_test_avg==='+str(APR_bulk_test_avg)+'\t'+
                        'bulk_best_threshold==='+str(bulk_best_threshold)+'\t'+
                        'bulk_best_f1_score==='+str(bulk_best_f1_score)+'\t'+
                        'AUC_sc_test_avg='+str(AUC_sc_test_avg)+'\t'+
                        'APR_sc_test_avg='+str(APR_sc_test_avg)+'\t'
                        'sc_best_threshold='+str(sc_best_threshold)+'\t'+
                        'sc_best_f1_score='+str(sc_best_f1_score)+'\t'+str(now)+'\t'+str(command_line)+'\n')
            
            df = pd.DataFrame(columns=['drug', 'encoder_dim', 'decoder_dim', 'pdim','batch_size','source_batch','target_batch','sam','lambda_recon','lambda_kl','lambda_ot','lambda_response','sc_AUC', 'sc_AUPR','bulk_AUC', 'bulk_AUPR'])
            file2 = './drug/'+str(DRUG)+'/'+str(DRUG)+'_auc_aupr'+'_result.xlsx'
            df_tmp=pd.DataFrame({
                'drug': [str(DRUG)],
                'encoder_dim': [str([str(i[1])+"," for i in self.enc[0]])],
                'decoder_dim': [str([str(i[1])+"," for i in self.dec[0]])],
                'pdim': [str([str(i[1])+"," for i in self.dec[3]])],
                'batch_size': [str(self.batch_size)],
                'source_batch': [str(source_batch)],
                'target_batch': [str(target_batch)],
                'sam': [str(sampler)],
                'lambda_recon': [str(self.lambda_recon)],
                'lambda_kl': [str(self.lambda_kl)],
                'lambda_ot': [str(self.lambda_ot)],
                'lambda_response': [str(self.lambda_response)],
                'sc_AUC': [str(AUC_sc_test_avg)],
                'sc_AUPR': [str(APR_sc_test_avg)],
                'bulk_AUC': [str(AUC_bulk_test_avg)],#TO DO
                'bulk_AUPR': [str(APR_bulk_test_avg)],#TO DO
            })
            if not os.path.isfile(file2):
                # 将数据添加到DataFrame的一行中
                df = pd.concat([df, df_tmp], axis=0)
                #保存
                df.to_excel(file2,index=False)
            else:
                df = pd.read_excel(file2)
                df = pd.concat([df, df_tmp], axis=0)
                #保存
                df.to_excel(file2,index=False)
        torch.cuda.empty_cache()
        return 



    def fit( # 使用EarlyStopping早停机制的时候（epoch不固定，由训练过程中损失函数不再下降为止），且使用共享解码器，且对源域bulk数据进行平衡采样，使用这个函数做训练
            self, 
            # dataloader,
            Ctrainloader, #TODO new feature
            Ptrainloader, #TODO new feature
            tran,
            num_cell,
            num_gene,
            mode='h',
            loss_type='BCE',
            label_weight=None,
            Prior=None,
            save_OT=False,
            use_specific=True,
            lambda_s=0.5,
            lambda_kl=0.5,
            lambda_recon=1.0,
            lambda_ot=1.0,
            lambda_response=1.0, #TODO,药物响应预测模型的权重参数
            reg=0.1,
            reg_m=1.0,
            lr=2e-4,
            max_iteration=30000,
            early_stopping=None,
            device='cuda:0',  
            verbose=False,
            drug_response=True, ###TODO,new feature
            adata_cm=None ###TODO,new feature
        ):
        """
        train VAE

        Parameters
        ----------
        dataloader
            An iterable over the given dataset for training.
        tran
            A global OT plan. tran={} if save_OT=False in function.py.
        num_cell
            List of number of cells in different datasets. 
        num_gene
            List of number of genes in different datasets. 
        mode
            Choose from ['h', 'v', 'd']
            If 'h', integrate data with common genes
            If 'v', integrate data profiled from the same cells
            If 'd', inetrgate data without common genes
            Default: 'h'
        loss_type
            type of loss. Choose between ['BCE', 'MSE', 'L1']. Default: 'BCE'
        label_weight
            Prior-guided weighted vectors. Default: None
        Prior
            Prior correspondence matrix.
        save_OT
            If True, output a global OT plan. Default: False
        use_specific
            If True, specific genes in each dataset will be considered. Default: True
        lambda_s
            Balanced parameter for specific genes. Default: 0.5
        lambda_kl: 
            Balanced parameter for KL divergence. Default: 0.5
        lambda_recon:
            Balanced parameter for reconstruction. Default: 1.0
        lambda_ot:
            Balanced parameter for OT. Default: 1.0
        reg:
            Entropy regularization parameter in OT. Default: 0.1
        reg_m:
            Unbalanced OT parameter. Larger values means more balanced OT. Default: 1.0
        lr
            Learning rate. Default: 2e-4
        max_iteration
            Max iterations for training. Training one batch_size samples is one iteration. Default: 60000
        early_stopping
            EarlyStopping class (definite in utils.py) for stoping the training if loss doesn't improve after a given patience. Default: None
        device
            'cuda' or 'cpu' for training. Default: 'cuda'
        verbose
            Verbosity, True or False. Default: False
        drug_response
            if True, use drug_response decoder to predict drug response label. Default: True
        """

        self.to(device)

        optim = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=5e-4)

        #n_epoch = int(np.ceil(max_iteration/len(dataloader))) #用于向上取整，即返回不小于输入值的最小整数,len(dataloader)=1
        n_epoch = int(np.ceil(max_iteration/len(Ctrainloader))) #用于向上取整，即返回不小于输入值的最小整数,len(dataloader)=1    
        if loss_type == 'BCE':
            loss_func = nn.BCELoss()
        elif loss_type == 'MSE':
            loss_func = nn.MSELoss()
        elif loss_type == 'L1':
            loss_func = nn.L1Loss()
        
        with tqdm(range(n_epoch), total=n_epoch, desc='Epochs') as tq: # 用于在循环或迭代过程中显示进度条。它可以帮助你直观地了解代码的执行进度。
            for epoch in tq:
                #TODO，dataloader.dataset是什么？，dataloader.dataset.shape=(22518, 4000)
                #print(f'####vae.py#279行####################dataloader.dataset.shape={dataloader.dataset.shape}')
                #tk0 = tqdm(enumerate(dataloader), total=len(dataloader), leave=False, desc='Iterations', disable=(not verbose))
                tk0 = tqdm(enumerate(zip(Ctrainloader,cycle(Ptrainloader))), total=len(Ctrainloader), leave=False, desc='Iterations', disable=(not verbose))
                epoch_loss = defaultdict(float)

                if mode == 'v':

                    for i, (x, idx) in tk0:
                        x = x.float().to(device)
                        idx = idx.to(device)

                        x_list = []
                        num_sum = []
                        num_sum.append(num_gene[0])
                        x_list.append(x[:, 0:num_sum[0]])

                        for j in range(1, self.n_domain):
                            num_sum.append(num_sum[-1] + num_gene[j])
                            x_list.append(x[:, num_sum[-2]:num_sum[-1]])

                        recon_loss = torch.tensor(0.0).to(device)
                        kl_loss = torch.tensor(0.0).to(device)

                        z, mu, var = self.encoder(x_list[0], 0)
                        kl_loss += kl_div(mu, var) 
                        recon = self.decoder(z, 0)
                        recon_loss = loss_func(recon, x_list[0]) * 2000

                        for j in range(1, self.n_domain):

                            recon = self.decoder(z, j)
                            recon_loss += lambda_s * loss_func(recon, x_list[j]) * 2000   ## TO DO
                    
                        loss = {'recon_loss':lambda_recon*recon_loss, 'kl_loss':lambda_kl*kl_loss} 

                        optim.zero_grad()
                        sum(loss.values()).backward()
                        optim.step()
                        
                        for k,v in loss.items():
                            epoch_loss[k] += loss[k].item()
                            
                        info = ','.join(['{}={:.3f}'.format(k, v) for k,v in loss.items()])
                        tk0.set_postfix_str(info)
                    

                    epoch_loss = {k:v/(i+1) for k, v in epoch_loss.items()}
                    epoch_info = ','.join(['{}={:.3f}'.format(k, v) for k,v in epoch_loss.items()])
                    tq.set_postfix_str(epoch_info) 


                elif mode == 'd':

                    for i, (x,y,idx) in tk0:

                        x, y = x.float().to(device), y.long().to(device)    
                        idx = idx.to(device)

                        if len(torch.unique(y)) < self.n_domain:
                            continue

                        mu_dict = {}
                        var_dict = {}
                               
                        loc_ref = torch.where(y==self.ref_id)[0]
                        idx_ref = idx[loc_ref] - sum(num_cell[0:self.ref_id])

                        loc_query = {}
                        idx_query = {}
                        tran_batch = {}
                        Prior_batch = None

                        query_id = list(range(self.n_domain))
                        query_id.remove(self.ref_id)

                        for j in query_id:

                            loc_query[j] = torch.where(y==j)[0]
                            idx_query[j] = idx[loc_query[j]] - sum(num_cell[0:j])

                            if save_OT:
                                tran_batch[j] = torch.from_numpy(tran[j]).to(device)[idx_query[j]][:,idx_ref]
                            else:
                                tran_batch[j] = None

                            if Prior is not None:
                                Prior_batch = Prior[j][idx_query[j]][:,idx_ref].to(device)

                        recon_loss = torch.tensor(0.0).to(device)
                        kl_loss = torch.tensor(0.0).to(device)
                        ot_loss = torch.tensor(0.0).to(device)

                        loc = loc_query
                        loc[self.ref_id] = loc_ref

                        for j in range(self.n_domain):

                            z_j, mu_j, var_j = self.encoder(x[loc[j]][:, 0:num_gene[j]], j)
                            mu_dict[j] = mu_j
                            var_dict[j] = var_j
                            recon_j = self.decoder(z_j, j)

                            recon_loss += loss_func(recon_j, x[loc[j]][:, 0:num_gene[j]]) * x[loc[j]].size(-1)  ## TO DO
                            kl_loss += kl_div(mu_j, var_j) 

                        for j in query_id:

                            ot_loss_tmp, tran_batch[j] = unbalanced_ot(tran_batch[j], mu_dict[j], var_dict[j], \
                                mu_dict[self.ref_id].detach(), var_dict[self.ref_id].detach(), Couple=Prior_batch, device=device)

                            if save_OT:
                                t0 = np.repeat(idx_query[j].cpu().numpy(), len(idx_ref)).reshape(len(idx_query[j]),len(idx_ref))
                                t1 = np.tile(idx_ref.cpu().numpy(), (len(idx_query[j]), 1))
                                tran[j][t0,t1] = tran_batch[j].cpu().numpy()

                            ot_loss += ot_loss_tmp

                        loss = {'recon_loss':lambda_recon*recon_loss, 'kl_loss':lambda_kl*kl_loss, 'ot_loss':lambda_ot*ot_loss} 

                        optim.zero_grad()
                        sum(loss.values()).backward()
                        optim.step()
          
                        for k,v in loss.items():
                            epoch_loss[k] += loss[k].item()
                            
                        info = ','.join(['{}={:.3f}'.format(k, v) for k,v in loss.items()])
                        tk0.set_postfix_str(info)

                    epoch_loss = {k:v/(i+1) for k, v in epoch_loss.items()}
                    epoch_info = ','.join(['{}={:.3f}'.format(k, v) for k,v in epoch_loss.items()])
                    tq.set_postfix_str(epoch_info) 


                elif mode == 'h':

                    '''for i, (x, y, idx) in tk0:#①y表示领域的id，0表示该条数据来自第一个数据集，1表示该条数据来自第二个数据集y=tensor([0, 1, 0, 1...])。   ②idx表示该条数据的索引号idx=tensor([ 9371,  6514,  9710]).
                        idx_origin = idx
                        #TODO 对x，y，idx的值有疑问,@see:data_loader.py, 143rows, x.shape=torch.Size([22517, 4000]),y.shape=torch.Size([22517]),idx.shape=torch.Size([22517])
                        # (x, y, idx) in tk0:,  @see:data_loader.py,143rows.
                        #print(f'###vae.py#415rows#################(x.shape, y.shape, idx.shape)===x.shape={x.shape},y.shape={y.shape},idx.shape={idx.shape}')
                        #print(f'###vae.py#416行################self.n_domain==={self.n_domain}') #self.n_domain===2
                        #print(f'###vae.py#427行################num_gene==={num_gene}') #num_gene===[2000, 2000, 2000]
                        x_c, y = x[:, 0:num_gene[self.n_domain]].float().to(device), y.long().to(device)  
                        idx = idx.to(device)
                                                    
                        loc_ref = torch.where(y==self.ref_id)[0] #用于根据给定条件选择元素
                        #TODO num_cell[0:self.ref_id]是什么, num_cell[0:self.ref_id]===[]，num_cell=[11259, 11259]，loc_ref=tensor([    0,     1,     2,  ..., 22510, 22512, 22514]
                        #print(f'###vae.py#423rows############num_cell[0:self.ref_id]==={num_cell[0:self.ref_id]}，num_cell={num_cell}，loc_ref.shape={loc_ref.shape}')
                        idx_ref = idx[loc_ref] - sum(num_cell[0:self.ref_id])#TODO，这个代码是什么意思？loc_ref是索引,loc_ref=tensor([  0,   1,   2,   3...])
                        #print(f'###vae.py#425rows####################idx_ref.shape={idx_ref.shape}')#idx_ref是找到self.ref_id的索引号，idx_ref=tensor([ 9371,  6514,  9710,...])
                        loc_query = {}
                        idx_query = {}
                        tran_batch = {}
                        Prior_batch = None

                        query_id = list(range(self.n_domain))
                        query_id.remove(self.ref_id)

                        if len(loc_ref) > 0:
                            for j in query_id:

                                loc_query[j] = torch.where(y==j)[0]
                                #print(f'####vae.py#440rows############j={j},loc_query[j]={loc_query[j]}')
                                idx_query[j] = idx[loc_query[j]] - sum(num_cell[0:j])

                                if save_OT:
                                    if len(idx_query[j]) != 0:
                                        if (len(idx_query[j])) == 1:
                                            tran_batch[j] = torch.from_numpy(tran[j]).to(device)[idx_query[j]][idx_ref]
                                        else:
                                            tran_batch[j] = torch.from_numpy(tran[j]).to(device)[idx_query[j]][:,idx_ref]
                                else:
                                    tran_batch[j] = None

                                if Prior is not None:
                                    Prior_batch = Prior[j][idx_query[j]][:,idx_ref].to(device)

                        ot_loss = torch.tensor(0.0).to(device)
                        recon_loss = torch.tensor(0.0).to(device)
                        kl_loss = torch.tensor(0.0).to(device)
                        drug_response_loss = torch.tensor(0.0).to(device)

                        loc = loc_query
                        loc[self.ref_id] = loc_ref
                        idx = idx_query
                        idx[self.ref_id] = idx_ref #x_c.shape=torch.Size([256, 2000])
                        #TODO，@see：vae.py, 148rows
                        z, mu, var = self.encoder(x_c, 0) #z.shape=torch.Size([256, 16]),mu.shape=torch.Size([256, 16]),var.shape=torch.Size([256, 16])
                        #TODO,疑问，self.encoder(x_c, 0)==x_c.shape=torch.Size([22517, 2000])，z.shape=torch.Size([22517, 16])，mu.shape=torch.Size([22517, 16])，var.shape=torch.Size([22517, 16])
                        #print(f'###vae.y#474rows##################self.encoder(x_c, 0)==x_c.shape={x_c.shape},z.shape={z.shape},mu.shape={mu.shape},var.shape={var.shape}')
                        recon_x_c = self.decoder(z, 0, y)        
   
                        if label_weight is None: # default, label_weight is None
                            #recon_loss = loss_func(recon_x_c, x_c) * 2000 # default, loss_func = nn.BCELoss()#TODO，这里为什么要乘2000，为什么BCEloss损失为负数
                            #recon_loss = loss_func(recon_x_c, x_c) * num_gene[self.n_domain]
                            recon_loss = loss_func(recon_x_c, x_c) 
                            #recon_loss = abs(loss_func(recon_x_c, x_c)) * 2000 # default, loss_func = nn.BCELoss()#TODO，这里为什么要乘2000，为什么BCEloss损失为负数
                        else:
                            for j, weight in enumerate(label_weight):

                                if len(loc[j])>0:
                                    if weight is None:
                                        recon_loss += 1/self.n_domain * loss_func(recon_x_c[loc[j]], x_c[loc[j]]) * \
                                        2000
                                        # kl_loss += kl_div(mu[loc[j]], var[loc[j]])                               
                                    else:
                                        weight = weight.to(device)
                                        recon_loss += 1/self.n_domain * F.binary_cross_entropy(recon_x_c[loc[j]], x_c[loc[j]], weight=weight[idx[j]]) * \
                                        2000
                                        # kl_loss += kl_div(mu[loc[j]], var[loc[j]], weight[idx[j]])

                        kl_loss = kl_div(mu, var) 
                        if use_specific:

                            x_s = x[:, num_gene[self.n_domain]:].float().to(device) #x.shape=torch.Size([22517, 4000])

                            for j in range(self.n_domain):
                                if len(loc[j])>0:
                                    #TODO，loc[j]是什么？,loc[j]应该是索引loc[j]=tensor([  0,   1,   2,   3,...]，loc[j].shape=torch.Size([11258])， z[loc[j]].shape=torch.Size([11258, 16])，z.shape=torch.Size([22517, 16])
                                    #print(f'###vae.py#489rows###################loc[j].shape={loc[j].shape}，z[loc[j]].shape={z[loc[j]].shape}，z.shape={z.shape}，loc[j]={loc[j]}，loc={loc}')
                                    recon_x_s = self.decoder(z[loc[j]], j+1)
                                    recon_loss += lambda_s * loss_func(recon_x_s, x_s[loc[j]][:, 0:num_gene[j]]) * 2000 #计算特异性高表达基因的重建数据和原数据之间的损失函数

                        if drug_response:#TODO drug_response_decoder的bulk模型训练
                            for j in range(self.n_domain):
                                if len(loc[j])>0 and j==0: # 表示取数据集x的隐空间嵌入z（取bulk细胞系的隐空间嵌入z ）
                                    if adata_cm is not None:
                                        a = idx_origin[y==j] #定位到数据集x（bulk细胞系）的索引，用来拿到其相应的response label
                                        groundtruth_bulk_label = adata_cm.obs['response'].iloc[a,].values.reshape(-1,1)
                                        groundtruth_bulk_label = torch.Tensor(groundtruth_bulk_label).to(device)
                                        #print(f'####vae.py#509rows##################groundtruth_bulk_label.shape={groundtruth_bulk_label.shape},groundtruth_bulk_label[:5]={groundtruth_bulk_label[:5]}')
                                        predicted_bulk_label = self.decoder(z[loc[j]], j+2+1) # decoder(,0)表示所有共同数据的共同的解码器；decoder(,1)表示数据集X特异性高可变基因的解码器；decoder(,2)表示数据集Y特异性高可变基因的解码器；decoder(,3)表示所有共同数据里的数据X的药物响应解码器；
                                        #print(f'####vae.py#511rows##################predicted_bulk_label.shape={predicted_bulk_label.shape},predicted_bulk_label[:5]={predicted_bulk_label[:5]}')
                                        drug_response_loss = drug_response_loss = loss_func(predicted_bulk_label, groundtruth_bulk_label)#计算bulk细胞系药物响应真实标签和预测标签之间的损失函数
                        
                        if len(torch.unique(y))>1 and len(loc[self.ref_id])!=0:
                            
                            mu_dict = {}
                            var_dict = {}

                            for j in range(self.n_domain):
                                if len(loc[j])>0:
                                    mu_dict[j] = mu[loc[j]]
                                    var_dict[j] = var[loc[j]]

                            for j in query_id:
                                if len(loc[j])>0:

                                    if label_weight is None:

                                        ot_loss_tmp, tran_batch[j] = unbalanced_ot(
                                            tran_batch[j],
                                            mu_dict[j], 
                                            var_dict[j], 
                                            mu_dict[self.ref_id].detach(), 
                                            var_dict[self.ref_id].detach(), 
                                            reg=reg,
                                            reg_m=reg_m,
                                            idx_q=idx_query[j],
                                            idx_r=idx_ref,
                                            Couple=Prior_batch, 
                                            device=device,
                                        )

                                    else:

                                        ot_loss_tmp, tran_batch[j] = unbalanced_ot(
                                            tran_batch[j],
                                            mu_dict[j], 
                                            var_dict[j], 
                                            mu_dict[self.ref_id].detach(), 
                                            var_dict[self.ref_id].detach(), 
                                            reg=reg,
                                            reg_m=reg_m,
                                            idx_q=idx_query[j],
                                            idx_r=idx_ref,
                                            Couple=Prior_batch, 
                                            device=device,
                                            query_weight=label_weight[j],
                                            ref_weight=label_weight[self.ref_id],
                                        )

                                    if save_OT:
                                        t0 = np.repeat(idx_query[j].cpu().numpy(), len(idx_ref)).reshape(len(idx_query[j]),len(idx_ref))
                                        t1 = np.tile(idx_ref.cpu().numpy(), (len(idx_query[j]), 1))
                                        tran[j][t0,t1] = tran_batch[j].cpu().numpy()

                                    ot_loss += ot_loss_tmp
   
                        loss = {'recloss':lambda_recon*recon_loss, 'klloss':lambda_kl*kl_loss, 'otloss':lambda_ot*ot_loss, 'drug_response_loss':lambda_response*drug_response_loss} #
                        
                        optim.zero_grad()
                        sum(loss.values()).backward()#TODO，疑问，可以这样子后向传播吗？
                        optim.step()

                        for k,v in loss.items():
                            epoch_loss[k] += loss[k].item() #TODO 疑问，loss[k].item()是什么, k=recloss,v=2528.068359375,loss.items()=dict_items([('recloss', tensor(...)), ('klloss', tensor(...)), ('otloss', tensor(...))])
                            #print(f'#######vae.py#555rows#############(k,v) in loss.items()====k={k},v={v}')

                        info = ','.join(['{}={:.2f}'.format(k, v) for k,v in loss.items()])
                        tk0.set_postfix_str(info)
                    '''
                    for i, data in tk0:#①y表示领域的id，0表示该条数据来自第一个数据集bulk，1表示该条数据来自第二个数据集sc,y=tensor([0, 1, 0, 1...])。   ②idx表示该条数据的索引号idx=tensor([ 9371,  6514,  9710]).
                        # 1. batch里 的bulk数据
                        x_bulk = data[0][0]
                        response_bulk = data[0][1].float()
                        y_bulk = torch.tensor([0] * data[0][0].shape[0]) # bulk数据的domain_id设置为0
                        idx_bulk = data[0][2]
                        # 2. batch里 的sc数据
                        x_sc = data[1][0]
                        response_sc = data[1][1].float()
                        y_sc = torch.tensor([1] * data[1][0].shape[0]) # sc数据的domain_id设置为1
                        idx_sc = data[1][2] + num_cell[0]
                        # 使用 torch.cat() 函数在第一个维度上进行连接
                        x = torch.cat((x_bulk, x_sc), dim=0)
                        y = torch.cat((y_bulk, y_sc), dim=0)
                        idx = torch.cat((idx_bulk, idx_sc), dim=0)
                        response = torch.cat((response_bulk, response_sc), dim=0)

                        idx_origin = idx
                        #TODO 对x，y，idx的值有疑问,@see:data_loader.py, 143rows, x.shape=torch.Size([22517, 4000]),y.shape=torch.Size([22517]),idx.shape=torch.Size([22517])
                        # (x, y, idx) in tk0:,  @see:data_loader.py,143rows.
                        # print(f'###vae.py#415rows#################(x.shape, y.shape, idx.shape)===x.shape={x.shape},y.shape={y.shape},idx.shape={idx.shape}')
                        # print(f'###vae.py#416行################self.n_domain==={self.n_domain}') #self.n_domain===2
                        # print(f'###vae.py#417行################num_gene==={num_gene}') #num_gene===[2000, 2000, 2000]
                        # x_c, y = x[:, 0:num_gene[self.n_domain]].float().to(device), y.long().to(device)
                        # x_c, y = x[:,:].float().to(device), y.long().to(device) #TODO 使用数据集x（bulk细胞系）所有的基因
                        x_c, y = x.float().to(device), y.long().to(device)
                        idx = idx.to(device)

                        loc_ref = torch.where(y==self.ref_id)[0] #用于根据给定条件选择元素
                        #TODO num_cell[0:self.ref_id]是什么, num_cell[0:self.ref_id]===[]，num_cell=[11259, 11259]，loc_ref=tensor([    0,     1,     2,  ..., 22510, 22512, 22514]
                        # print(f'###vae.py#423rows############num_cell[0:self.ref_id]==={num_cell[0:self.ref_id]}，num_cell={num_cell}，loc_ref.shape={loc_ref.shape}')
                        idx_ref = idx[loc_ref] - sum(num_cell[0:self.ref_id])#TODO，这个代码是什么意思？loc_ref是索引,loc_ref=tensor([  0,   1,   2,   3...])
                        # print(f'###vae.py#425rows####################idx_ref.shape={idx_ref.shape}')#idx_ref是找到self.ref_id的索引号，idx_ref=tensor([ 9371,  6514,  9710,...])
                        loc_query = {}
                        idx_query = {}
                        tran_batch = {}
                        Prior_batch = None

                        query_id = list(range(self.n_domain))
                        query_id.remove(self.ref_id)

                        if len(loc_ref) > 0:
                            for j in query_id:

                                loc_query[j] = torch.where(y==j)[0]
                                # print(f'####vae.py#440rows############j={j},loc_query[j]={loc_query[j]}')
                                idx_query[j] = idx[loc_query[j]] - sum(num_cell[0:j])

                                if save_OT:
                                    if len(idx_query[j]) != 0:
                                        if (len(idx_query[j])) == 1:
                                            tran_batch[j] = torch.from_numpy(tran[j]).to(device)[idx_query[j]][idx_ref]
                                        else:
                                            tran_batch[j] = torch.from_numpy(tran[j]).to(device)[idx_query[j]][:,idx_ref]
                                else:
                                    tran_batch[j] = None

                                if Prior is not None:
                                    Prior_batch = Prior[j][idx_query[j]][:,idx_ref].to(device)

                        ot_loss = torch.tensor(0.0).to(device)
                        recon_loss = torch.tensor(0.0).to(device)
                        kl_loss = torch.tensor(0.0).to(device)
                        drug_response_loss = torch.tensor(0.0).to(device)

                        loc = loc_query
                        loc[self.ref_id] = loc_ref
                        idx = idx_query
                        idx[self.ref_id] = idx_ref #x_c.shape=torch.Size([256, 2000])
                        #TODO，@see：vae.py, 148rows
                        z, mu, var = self.encoder(x_c, 0) #z.shape=torch.Size([256, 16]),mu.shape=torch.Size([256, 16]),var.shape=torch.Size([256, 16])
                        #TODO,疑问，self.encoder(x_c, 0)==x_c.shape=torch.Size([22517, 2000])，z.shape=torch.Size([22517, 16])，mu.shape=torch.Size([22517, 16])，var.shape=torch.Size([22517, 16])
                        # print(f'###vae.y#474rows##################self.encoder(x_c, 0)==x_c.shape={x_c.shape},z.shape={z.shape},mu.shape={mu.shape},var.shape={var.shape}')
                        recon_x_c = self.decoder(z, 0, y)

                        if label_weight is None: # default, label_weight is None
                            # print(f'####vae.py#482rows###########recon_x_c={recon_x_c},x_c={x_c}')
                            # recon_loss = loss_func(recon_x_c, x_c) * 2000 # default, loss_func = nn.BCELoss()#TODO，这里为什么要乘2000，为什么BCEloss损失为负数
                            # recon_loss = loss_func(recon_x_c, x_c) * num_gene[self.n_domain] # default, loss_func = nn.BCELoss()#TODO，这里为什么要乘2000，为什么BCEloss损失为负数
                            recon_loss = loss_func(recon_x_c, x_c)
                            # recon_loss = abs(loss_func(recon_x_c, x_c)) * 2000 # default, loss_func = nn.BCELoss()#TODO，这里为什么要乘2000，为什么BCEloss损失为负数
                        else:
                            for j, weight in enumerate(label_weight):

                                if len(loc[j])>0:
                                    if weight is None:
                                        recon_loss += 1/self.n_domain * loss_func(recon_x_c[loc[j]], x_c[loc[j]]) * \
                                                      2000
                                        # kl_loss += kl_div(mu[loc[j]], var[loc[j]])
                                    else:
                                        weight = weight.to(device)
                                        recon_loss += 1/self.n_domain * F.binary_cross_entropy(recon_x_c[loc[j]], x_c[loc[j]], weight=weight[idx[j]]) * \
                                                      2000
                                        # kl_loss += kl_div(mu[loc[j]], var[loc[j]], weight[idx[j]])

                        kl_loss = kl_div(mu, var)
                        if use_specific:

                            x_s = x[:, num_gene[self.n_domain]:].float().to(device) #x.shape=torch.Size([22517, 4000])

                            for j in range(self.n_domain):
                                if len(loc[j])>0:
                                    #TODO，loc[j]是什么？,loc[j]应该是索引loc[j]=tensor([  0,   1,   2,   3,...]，loc[j].shape=torch.Size([11258])， z[loc[j]].shape=torch.Size([11258, 16])，z.shape=torch.Size([22517, 16])
                                    # print(f'###vae.py#489rows###################loc[j].shape={loc[j].shape}，z[loc[j]].shape={z[loc[j]].shape}，z.shape={z.shape}，loc[j]={loc[j]}，loc={loc}')
                                    recon_x_s = self.decoder(z[loc[j]], j+1)
                                    recon_loss += lambda_s * loss_func(recon_x_s, x_s[loc[j]][:, 0:num_gene[j]]) * 2000 #计算特异性高表达基因的重建数据和原数据之间的损失函数

                        if drug_response:#TODO drug_response_decoder的bulk模型训练
                            for j in range(self.n_domain):
                                if len(loc[j])>0 and j==0: # 表示取数据集x的隐空间嵌入z（取bulk细胞系的隐空间嵌入z ）
                                    if adata_cm is not None:
                                        # a = idx_origin[y==j] #定位到数据集x（bulk细胞系）的索引，用来拿到其相应的response label
                                        # groundtruth_bulk_label = adata_cm.obs['response'].iloc[a,].values.reshape(-1,1)
                                        groundtruth_bulk_label = response_bulk.reshape(-1,1)
                                        groundtruth_bulk_label = torch.Tensor(groundtruth_bulk_label).to(device)
                                        # print(f'####vae.py#509rows##################groundtruth_bulk_label.shape={groundtruth_bulk_label.shape},groundtruth_bulk_label[:5]={groundtruth_bulk_label[:5]}')
                                        predicted_bulk_label = self.decoder(z[loc[j]], j+2+1) # decoder(,0)表示所有共同数据的共同的解码器；decoder(,1)表示数据集X特异性高可变基因的解码器；decoder(,2)表示数据集Y特异性高可变基因的解码器；decoder(,3)表示所有共同数据里的数据X的药物响应解码器；
                                        # print(f'####vae.py#511rows##################predicted_bulk_label.shape={predicted_bulk_label.shape},predicted_bulk_label[:5]={predicted_bulk_label[:5]}')
                                        drug_response_loss = loss_func(predicted_bulk_label, groundtruth_bulk_label)#计算bulk细胞系药物响应真实标签和预测标签之间的损失函数

                        if len(torch.unique(y))>1 and len(loc[self.ref_id])!=0:

                            mu_dict = {}
                            var_dict = {}

                            for j in range(self.n_domain):
                                if len(loc[j])>0:
                                    mu_dict[j] = mu[loc[j]]
                                    var_dict[j] = var[loc[j]]

                            for j in query_id:
                                if len(loc[j])>0:

                                    if label_weight is None:

                                        ot_loss_tmp, tran_batch[j] = unbalanced_ot(
                                            tran_batch[j],
                                            mu_dict[j],
                                            var_dict[j],
                                            mu_dict[self.ref_id].detach(),
                                            var_dict[self.ref_id].detach(),
                                            reg=reg,
                                            reg_m=reg_m,
                                            idx_q=idx_query[j], # default query_weight is None,这个参数没用到
                                            idx_r=idx_ref, # default ref_weight is None, 这个参数函数也没用到
                                            Couple=Prior_batch, # default Prior is None, @see:881row, 所以Prior_batch也是None，这个参数函数没用到
                                            device=device,
                                        )

                                    else:

                                        ot_loss_tmp, tran_batch[j] = unbalanced_ot(
                                            tran_batch[j],
                                            mu_dict[j],
                                            var_dict[j],
                                            mu_dict[self.ref_id].detach(),
                                            var_dict[self.ref_id].detach(),
                                            reg=reg,
                                            reg_m=reg_m,
                                            idx_q=idx_query[j],
                                            idx_r=idx_ref,
                                            Couple=Prior_batch,
                                            device=device,
                                            query_weight=label_weight[j],
                                            ref_weight=label_weight[self.ref_id],
                                        )

                                    if save_OT:
                                        t0 = np.repeat(idx_query[j].cpu().numpy(), len(idx_ref)).reshape(len(idx_query[j]),len(idx_ref))
                                        t1 = np.tile(idx_ref.cpu().numpy(), (len(idx_query[j]), 1))
                                        tran[j][t0,t1] = tran_batch[j].cpu().numpy()

                                    ot_loss += ot_loss_tmp

                        loss = {'recloss':lambda_recon*recon_loss, 'klloss':lambda_kl*kl_loss, 'otloss':lambda_ot*ot_loss, 'drug_response_loss':lambda_response*drug_response_loss} #

                        optim.zero_grad()
                        sum(loss.values()).backward()#TODO，疑问，可以这样子后向传播吗？
                        optim.step()

                        for k,v in loss.items():
                            epoch_loss[k] += loss[k].item() #TODO 疑问，loss[k].item()是什么, k=recloss,v=2528.068359375,loss.items()=dict_items([('recloss', tensor(...)), ('klloss', tensor(...)), ('otloss', tensor(...))])
                            # print(f'#######vae.py#555rows#############(k,v) in loss.items()====k={k},v={v}')

                        info = ','.join(['{}={:.2f}'.format(k, v) for k,v in loss.items()])
                        tk0.set_postfix_str(info)
                    
                    epoch_loss = {k:v/(i+1) for k, v in epoch_loss.items()}

                    epoch_info = ','.join(['{}={:.2f}'.format(k, v) for k,v in epoch_loss.items()])
                    tq.set_postfix_str(epoch_info) 
                        
                    early_stopping(sum(epoch_loss.values()), self)
                    if early_stopping.early_stop:
                        print('EarlyStopping: run {} epoch'.format(epoch+1))
                        break
                torch.cuda.empty_cache()

    def fit_unshared_decoder( # 使用EarlyStopping早停机制的时候（epoch不固定，由训练过程中损失函数不再下降为止），且不使用共享解码器，且对源域bulk数据进行平衡采样，使用这个函数做训练
            self, 
            # dataloader,
            Ctrainloader, #TODO new feature
            Ptrainloader, #TODO new feature
            tran,
            num_cell,
            num_gene,
            mode='h',
            loss_type='BCE',
            label_weight=None,
            Prior=None,
            save_OT=False,
            use_specific=True,
            lambda_s=0.5,
            lambda_kl=0.5,
            lambda_recon=1.0,
            lambda_ot=1.0,
            lambda_response=1.0, #TODO,药物响应预测模型的权重参数
            reg=0.1,
            reg_m=1.0,
            lr=2e-4,
            max_iteration=30000,
            early_stopping=None,
            device='cuda:0',  
            verbose=False,
            drug_response=True, ###TODO,new feature
            adata_cm=None, ###TODO,new feature
            unshared_decoder=False,
            lambda_cell=1.0, #TODO,单细胞正则化损失函数的权重参数
            cell_regularization=False, ###TODO,new feature
            global_match=False, ###TODO,new feature
            mmd_GAMMA=1000, #Gamma parameter in the kernel of the MMD loss of the transfer learning, default: 1000
            lambda_mmd=1.0, ###TODO,new feature
        ):
        """
        train VAE

        Parameters
        ----------
        dataloader
            An iterable over the given dataset for training.
        tran
            A global OT plan. tran={} if save_OT=False in function.py.
        num_cell
            List of number of cells in different datasets. 
        num_gene
            List of number of genes in different datasets. 
        mode
            Choose from ['h', 'v', 'd']
            If 'h', integrate data with common genes
            If 'v', integrate data profiled from the same cells
            If 'd', inetrgate data without common genes
            Default: 'h'
        loss_type
            type of loss. Choose between ['BCE', 'MSE', 'L1']. Default: 'BCE'
        label_weight
            Prior-guided weighted vectors. Default: None
        Prior
            Prior correspondence matrix.
        save_OT
            If True, output a global OT plan. Default: False
        use_specific
            If True, specific genes in each dataset will be considered. Default: True
        lambda_s
            Balanced parameter for specific genes. Default: 0.5
        lambda_kl: 
            Balanced parameter for KL divergence. Default: 0.5
        lambda_recon:
            Balanced parameter for reconstruction. Default: 1.0
        lambda_ot:
            Balanced parameter for OT. Default: 1.0
        reg:
            Entropy regularization parameter in OT. Default: 0.1
        reg_m:
            Unbalanced OT parameter. Larger values means more balanced OT. Default: 1.0
        lr
            Learning rate. Default: 2e-4
        max_iteration
            Max iterations for training. Training one batch_size samples is one iteration. Default: 60000
        early_stopping
            EarlyStopping class (definite in utils.py) for stoping the training if loss doesn't improve after a given patience. Default: None
        device
            'cuda' or 'cpu' for training. Default: 'cuda'
        verbose
            Verbosity, True or False. Default: False
        drug_response
            if True, use drug_response decoder to predict drug response label. Default: True
        """

        self.to(device)
        # Set distribution loss
        def mmd_loss_func(x,y,GAMMA=mmd_GAMMA):
            result = mmd.mmd_loss(x,y,GAMMA)
            return result
        optim = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=5e-4)
        #n_epoch = int(np.ceil(max_iteration/len(dataloader))) #用于向上取整，即返回不小于输入值的最小整数,len(dataloader)=1
        n_epoch = int(np.ceil(max_iteration/len(Ctrainloader))) #用于向上取整，即返回不小于输入值的最小整数,len(dataloader)=1    
        if loss_type == 'BCE':
            loss_func = nn.BCELoss()
        elif loss_type == 'MSE':
            loss_func = nn.MSELoss()
        elif loss_type == 'L1':
            loss_func = nn.L1Loss()
        
        with tqdm(range(n_epoch), total=n_epoch, desc='Epochs') as tq: # 用于在循环或迭代过程中显示进度条。它可以帮助你直观地了解代码的执行进度。
            for epoch in tq:
                #TODO，dataloader.dataset是什么？，dataloader.dataset.shape=(22518, 4000)
                #print(f'####vae.py#279行####################dataloader.dataset.shape={dataloader.dataset.shape}')
                #tk0 = tqdm(enumerate(dataloader), total=len(dataloader), leave=False, desc='Iterations', disable=(not verbose))
                tk0 = tqdm(enumerate(zip(Ctrainloader,cycle(Ptrainloader))), total=len(Ctrainloader), leave=False, desc='Iterations', disable=(not verbose))
                epoch_loss = defaultdict(float)

                if mode == 'h':

                    for i, data in tk0:#①y表示领域的id，0表示该条数据来自第一个数据集bulk，1表示该条数据来自第二个数据集sc,y=tensor([0, 1, 0, 1...])。   ②idx表示该条数据的索引号idx=tensor([ 9371,  6514,  9710]).
                        # 1. batch里 的bulk数据
                        x_bulk = data[0][0]
                        response_bulk = data[0][1].float()
                        y_bulk = torch.tensor([0] * data[0][0].shape[0]) # bulk数据的domain_id设置为0
                        idx_bulk = data[0][2]
                        # 2. batch里 的sc数据
                        x_sc = data[1][0]
                        response_sc = data[1][1].float()
                        y_sc = torch.tensor([1] * data[1][0].shape[0]) # sc数据的domain_id设置为1
                        idx_sc = data[1][2] + num_cell[0]
                        # 使用 torch.cat() 函数在第一个维度上进行连接
                        x = torch.cat((x_bulk, x_sc), dim=0)
                        y = torch.cat((y_bulk, y_sc), dim=0)
                        idx = torch.cat((idx_bulk, idx_sc), dim=0)
                        response = torch.cat((response_bulk, response_sc), dim=0)

                        idx_origin = idx
                        
                        x_c, y = x.float().to(device), y.long().to(device)
                        idx = idx.to(device)

                        loc_ref = torch.where(y==self.ref_id)[0] #用于根据给定条件选择元素
                        
                        idx_ref = idx[loc_ref] - sum(num_cell[0:self.ref_id])#TODO，这个代码是什么意思？loc_ref是索引,loc_ref=tensor([  0,   1,   2,   3...])
                        # print(f'###vae.py#425rows####################idx_ref.shape={idx_ref.shape}')#idx_ref是找到self.ref_id的索引号，idx_ref=tensor([ 9371,  6514,  9710,...])
                        loc_query = {}
                        idx_query = {}
                        tran_batch = {}
                        Prior_batch = None

                        query_id = list(range(self.n_domain))
                        query_id.remove(self.ref_id)

                        if len(loc_ref) > 0:
                            for j in query_id:

                                loc_query[j] = torch.where(y==j)[0]
                                # print(f'####vae.py#440rows############j={j},loc_query[j]={loc_query[j]}')
                                idx_query[j] = idx[loc_query[j]] - sum(num_cell[0:j])

                                if save_OT:
                                    if len(idx_query[j]) != 0:
                                        if (len(idx_query[j])) == 1:
                                            tran_batch[j] = torch.from_numpy(tran[j]).to(device)[idx_query[j]][idx_ref]
                                        else:
                                            tran_batch[j] = torch.from_numpy(tran[j]).to(device)[idx_query[j]][:,idx_ref]
                                else:
                                    tran_batch[j] = None

                                if Prior is not None:
                                    Prior_batch = Prior[j][idx_query[j]][:,idx_ref].to(device)

                        ot_loss = torch.tensor(0.0).to(device)
                        mmd_loss = torch.tensor(0.0).to(device)
                        cell_regularization_loss = torch.tensor(0.0).to(device)
                        recon_loss = torch.tensor(0.0).to(device)
                        recon_loss_1 = torch.tensor(0.0).to(device)
                        recon_loss_2 = torch.tensor(0.0).to(device)
                        kl_loss = torch.tensor(0.0).to(device)
                        drug_response_loss = torch.tensor(0.0).to(device)

                        loc = loc_query
                        loc[self.ref_id] = loc_ref
                        idx = idx_query
                        idx[self.ref_id] = idx_ref #x_c.shape=torch.Size([256, 2000])
                        #TODO，@see：vae.py, 148rows，使用共享编码器
                        z, mu, var = self.encoder(x_c, 0) #z.shape=torch.Size([256, 16]),mu.shape=torch.Size([256, 16]),var.shape=torch.Size([256, 16])
                        # 计算全局匹配损失mmd
                        if global_match:
                            min_size = min(z[loc[0]].shape[0],z[loc[1]].shape[0])
                            if (z[loc[0]].shape[0]!=z[loc[1]].shape[0]):
                                x_src = z[loc[0]][:min_size,]
                                x_tar = z[loc[1]][:min_size,]
                            # mmd_loss = mmd_loss_func(z[loc[0]], z[loc[1]])
                            mmd_loss = mmd_loss_func(x_src, x_tar)
                        # 计算细胞正则化损失
                        if cell_regularization:# use cell regularization
                            if z[loc[1]].shape[0]<10: # target data（single cell dataset）
                                next
                            else:
                                edgeList = calculateKNNgraphDistanceMatrix(z[loc[1]].cpu().detach().numpy(), distanceType='euclidean', k=10)
                                listResult, size = generateLouvainCluster(edgeList)
                                # sc sim loss
                                loss_s = 0
                                for i in range(size):
                                    #print(i)
                                    s = cosine_similarity(x_c[loc[1]][np.asarray(listResult) == i,:].cpu().detach().numpy())
                                    s = 1-s
                                    loss_s += np.sum(np.triu(s,1))/((s.shape[0]*s.shape[0])*2-s.shape[0])
                                cell_regularization_loss += loss_s
                                
                        if label_weight is None: # default, label_weight is None
                            #TODO, 使用两个解码器，不使用共享解码器
                            recon_x_bulk = self.decoder(z[loc[0]], 4, None)
                            recon_x_sc = self.decoder(z[loc[1]], 5, None)
                            recon_loss_1 = loss_func(recon_x_bulk, x_bulk.float().to(device))
                            recon_loss_2 = loss_func(recon_x_sc, x_sc.float().to(device))
                            recon_loss = recon_loss_1 + recon_loss_2

                        kl_loss = kl_div(mu, var)
                        
                        if drug_response: #TODO drug_response_decoder的bulk模型训练
                            for j in range(self.n_domain):
                                if len(loc[j])>0 and j==0: # 表示取数据集x的隐空间嵌入z（取bulk细胞系的隐空间嵌入z ）
                                    if adata_cm is not None:
                                        # a = idx_origin[y==j] #定位到数据集x（bulk细胞系）的索引，用来拿到其相应的response label
                                        # groundtruth_bulk_label = adata_cm.obs['response'].iloc[a,].values.reshape(-1,1)
                                        groundtruth_bulk_label = response_bulk.reshape(-1,1)
                                        groundtruth_bulk_label = torch.Tensor(groundtruth_bulk_label).to(device)
                                        # print(f'####vae.py#509rows##################groundtruth_bulk_label.shape={groundtruth_bulk_label.shape},groundtruth_bulk_label[:5]={groundtruth_bulk_label[:5]}')
                                        predicted_bulk_label = self.decoder(z[loc[j]], j+2+1) # decoder(,0)表示所有共同数据的共同的解码器；decoder(,1)表示数据集X特异性高可变基因的解码器；decoder(,2)表示数据集Y特异性高可变基因的解码器；decoder(,3)表示所有共同数据里的数据X的药物响应解码器；
                                        # print(f'####vae.py#511rows##################predicted_bulk_label.shape={predicted_bulk_label.shape},predicted_bulk_label[:5]={predicted_bulk_label[:5]}')
                                        drug_response_loss = loss_func(predicted_bulk_label, groundtruth_bulk_label)#计算bulk细胞系药物响应真实标签和预测标签之间的损失函数

                        if len(torch.unique(y))>1 and len(loc[self.ref_id])!=0:

                            mu_dict = {}
                            var_dict = {}

                            for j in range(self.n_domain):
                                if len(loc[j])>0:
                                    mu_dict[j] = mu[loc[j]]
                                    var_dict[j] = var[loc[j]]

                            for j in query_id:
                                if len(loc[j])>0:

                                    if label_weight is None:

                                        ot_loss_tmp, tran_batch[j] = unbalanced_ot(
                                            tran_batch[j],
                                            mu_dict[j],
                                            var_dict[j],
                                            mu_dict[self.ref_id].detach(),
                                            var_dict[self.ref_id].detach(),
                                            reg=reg,
                                            reg_m=reg_m,
                                            idx_q=idx_query[j], # default query_weight is None,这个参数没用到
                                            idx_r=idx_ref, # default ref_weight is None, 这个参数函数也没用到
                                            Couple=Prior_batch, # default Prior is None, @see:881row, 所以Prior_batch也是None，这个参数函数没用到
                                            device=device,
                                        )

                                    else:

                                        ot_loss_tmp, tran_batch[j] = unbalanced_ot(
                                            tran_batch[j],
                                            mu_dict[j],
                                            var_dict[j],
                                            mu_dict[self.ref_id].detach(),
                                            var_dict[self.ref_id].detach(),
                                            reg=reg,
                                            reg_m=reg_m,
                                            idx_q=idx_query[j],
                                            idx_r=idx_ref,
                                            Couple=Prior_batch,
                                            device=device,
                                            query_weight=label_weight[j],
                                            ref_weight=label_weight[self.ref_id],
                                        )

                                    if save_OT:
                                        t0 = np.repeat(idx_query[j].cpu().numpy(), len(idx_ref)).reshape(len(idx_query[j]),len(idx_ref))
                                        t1 = np.tile(idx_ref.cpu().numpy(), (len(idx_query[j]), 1))
                                        tran[j][t0,t1] = tran_batch[j].cpu().numpy()

                                    ot_loss += ot_loss_tmp

                        loss = {'recloss':lambda_recon*recon_loss, 'klloss':lambda_kl*kl_loss, 'otloss':lambda_ot*ot_loss, 'drug_response_loss':lambda_response*drug_response_loss} #
                        if cell_regularization:
                            loss['cell_R_loss'] = lambda_cell*cell_regularization_loss
                        if global_match:
                            loss['mmd_loss'] = lambda_mmd*mmd_loss
                        optim.zero_grad()
                        sum(loss.values()).backward()#TODO，疑问，可以这样子后向传播吗？
                        optim.step()

                        for k,v in loss.items():
                            epoch_loss[k] += loss[k].item() #TODO 疑问，loss[k].item()是什么, k=recloss,v=2528.068359375,loss.items()=dict_items([('recloss', tensor(...)), ('klloss', tensor(...)), ('otloss', tensor(...))])
                            # print(f'#######vae.py#555rows#############(k,v) in loss.items()====k={k},v={v}')

                        info = ','.join(['{}={:.2f}'.format(k, v) for k,v in loss.items()])
                        tk0.set_postfix_str(info)
                    
                    epoch_loss = {k:v/(i+1) for k, v in epoch_loss.items()}

                    epoch_info = ','.join(['{}={:.2f}'.format(k, v) for k,v in epoch_loss.items()])
                    tq.set_postfix_str(epoch_info) 
                        
                    early_stopping(sum(epoch_loss.values()), self)
                    if early_stopping.early_stop:
                        print('EarlyStopping: run {} epoch'.format(epoch+1))
                        break
                torch.cuda.empty_cache()


    def fit_1( # 使用EarlyStopping早停机制的时候（epoch不固定，由训练过程中损失函数不再下降为止），且使用共享解码器，且未对源域bulk数据平衡采样，使用这个fit_1()做训练
            self, 
            dataloader,
            tran,
            num_cell,
            num_gene,
            mode='h',
            loss_type='BCE',
            label_weight=None,
            Prior=None,
            save_OT=False,
            use_specific=True,
            lambda_s=0.5,
            lambda_kl=0.5,
            lambda_recon=1.0,
            lambda_ot=1.0,
            lambda_response=1.0, #TODO,药物响应预测模型的权重参数
            reg=0.1,
            reg_m=1.0,
            lr=2e-4,
            max_iteration=30000,
            early_stopping=None,
            device='cuda:0',  
            verbose=False,
            drug_response=True, ###TODO,new feature
            adata_cm=None ###TODO,new feature
        ):
        """
        train VAE

        Parameters
        ----------
        dataloader
            An iterable over the given dataset for training.
        tran
            A global OT plan. tran={} if save_OT=False in function.py.
        num_cell
            List of number of cells in different datasets. 
        num_gene
            List of number of genes in different datasets. 
        mode
            Choose from ['h', 'v', 'd']
            If 'h', integrate data with common genes
            If 'v', integrate data profiled from the same cells
            If 'd', inetrgate data without common genes
            Default: 'h'
        loss_type
            type of loss. Choose between ['BCE', 'MSE', 'L1']. Default: 'BCE'
        label_weight
            Prior-guided weighted vectors. Default: None
        Prior
            Prior correspondence matrix.
        save_OT
            If True, output a global OT plan. Default: False
        use_specific
            If True, specific genes in each dataset will be considered. Default: True
        lambda_s
            Balanced parameter for specific genes. Default: 0.5
        lambda_kl: 
            Balanced parameter for KL divergence. Default: 0.5
        lambda_recon:
            Balanced parameter for reconstruction. Default: 1.0
        lambda_ot:
            Balanced parameter for OT. Default: 1.0
        reg:
            Entropy regularization parameter in OT. Default: 0.1
        reg_m:
            Unbalanced OT parameter. Larger values means more balanced OT. Default: 1.0
        lr
            Learning rate. Default: 2e-4
        max_iteration
            Max iterations for training. Training one batch_size samples is one iteration. Default: 60000
        early_stopping
            EarlyStopping class (definite in utils.py) for stoping the training if loss doesn't improve after a given patience. Default: None
        device
            'cuda' or 'cpu' for training. Default: 'cuda'
        verbose
            Verbosity, True or False. Default: False
        drug_response
            if True, use drug_response decoder to predict drug response label. Default: True
        """

        self.to(device)

        optim = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=5e-4)

        n_epoch = int(np.ceil(max_iteration/len(dataloader))) #用于向上取整，即返回不小于输入值的最小整数,len(dataloader)=1
        if loss_type == 'BCE':
            loss_func = nn.BCELoss()
        elif loss_type == 'MSE':
            loss_func = nn.MSELoss()
        elif loss_type == 'L1':
            loss_func = nn.L1Loss()
        
        with tqdm(range(n_epoch), total=n_epoch, desc='Epochs') as tq: # 用于在循环或迭代过程中显示进度条。它可以帮助你直观地了解代码的执行进度。
            for epoch in tq:
                #TODO，dataloader.dataset是什么？，dataloader.dataset.shape=(22518, 4000)
                #print(f'####vae.py#279行####################dataloader.dataset.shape={dataloader.dataset.shape}')
                tk0 = tqdm(enumerate(dataloader), total=len(dataloader), leave=False, desc='Iterations', disable=(not verbose))
                epoch_loss = defaultdict(float)

                if mode == 'v':

                    for i, (x, idx) in tk0:
                        x = x.float().to(device)
                        idx = idx.to(device)

                        x_list = []
                        num_sum = []
                        num_sum.append(num_gene[0])
                        x_list.append(x[:, 0:num_sum[0]])

                        for j in range(1, self.n_domain):
                            num_sum.append(num_sum[-1] + num_gene[j])
                            x_list.append(x[:, num_sum[-2]:num_sum[-1]])

                        recon_loss = torch.tensor(0.0).to(device)
                        kl_loss = torch.tensor(0.0).to(device)

                        z, mu, var = self.encoder(x_list[0], 0)
                        kl_loss += kl_div(mu, var) 
                        recon = self.decoder(z, 0)
                        recon_loss = loss_func(recon, x_list[0]) * 2000

                        for j in range(1, self.n_domain):

                            recon = self.decoder(z, j)
                            recon_loss += lambda_s * loss_func(recon, x_list[j]) * 2000   ## TO DO
                    
                        loss = {'recon_loss':lambda_recon*recon_loss, 'kl_loss':lambda_kl*kl_loss} 

                        optim.zero_grad()
                        sum(loss.values()).backward()
                        optim.step()
                        
                        for k,v in loss.items():
                            epoch_loss[k] += loss[k].item()
                            
                        info = ','.join(['{}={:.3f}'.format(k, v) for k,v in loss.items()])
                        tk0.set_postfix_str(info)
                    

                    epoch_loss = {k:v/(i+1) for k, v in epoch_loss.items()}
                    epoch_info = ','.join(['{}={:.3f}'.format(k, v) for k,v in epoch_loss.items()])
                    tq.set_postfix_str(epoch_info) 


                elif mode == 'd':

                    for i, (x,y,idx) in tk0:

                        x, y = x.float().to(device), y.long().to(device)    
                        idx = idx.to(device)

                        if len(torch.unique(y)) < self.n_domain:
                            continue

                        mu_dict = {}
                        var_dict = {}
                               
                        loc_ref = torch.where(y==self.ref_id)[0]
                        idx_ref = idx[loc_ref] - sum(num_cell[0:self.ref_id])

                        loc_query = {}
                        idx_query = {}
                        tran_batch = {}
                        Prior_batch = None

                        query_id = list(range(self.n_domain))
                        query_id.remove(self.ref_id)

                        for j in query_id:

                            loc_query[j] = torch.where(y==j)[0]
                            idx_query[j] = idx[loc_query[j]] - sum(num_cell[0:j])

                            if save_OT:
                                tran_batch[j] = torch.from_numpy(tran[j]).to(device)[idx_query[j]][:,idx_ref]
                            else:
                                tran_batch[j] = None

                            if Prior is not None:
                                Prior_batch = Prior[j][idx_query[j]][:,idx_ref].to(device)

                        recon_loss = torch.tensor(0.0).to(device)
                        kl_loss = torch.tensor(0.0).to(device)
                        ot_loss = torch.tensor(0.0).to(device)

                        loc = loc_query
                        loc[self.ref_id] = loc_ref

                        for j in range(self.n_domain):

                            z_j, mu_j, var_j = self.encoder(x[loc[j]][:, 0:num_gene[j]], j)
                            mu_dict[j] = mu_j
                            var_dict[j] = var_j
                            recon_j = self.decoder(z_j, j)

                            recon_loss += loss_func(recon_j, x[loc[j]][:, 0:num_gene[j]]) * x[loc[j]].size(-1)  ## TO DO
                            kl_loss += kl_div(mu_j, var_j) 

                        for j in query_id:

                            ot_loss_tmp, tran_batch[j] = unbalanced_ot(tran_batch[j], mu_dict[j], var_dict[j], \
                                mu_dict[self.ref_id].detach(), var_dict[self.ref_id].detach(), Couple=Prior_batch, device=device)

                            if save_OT:
                                t0 = np.repeat(idx_query[j].cpu().numpy(), len(idx_ref)).reshape(len(idx_query[j]),len(idx_ref))
                                t1 = np.tile(idx_ref.cpu().numpy(), (len(idx_query[j]), 1))
                                tran[j][t0,t1] = tran_batch[j].cpu().numpy()

                            ot_loss += ot_loss_tmp

                        loss = {'recon_loss':lambda_recon*recon_loss, 'kl_loss':lambda_kl*kl_loss, 'ot_loss':lambda_ot*ot_loss} 

                        optim.zero_grad()
                        sum(loss.values()).backward()
                        optim.step()
          
                        for k,v in loss.items():
                            epoch_loss[k] += loss[k].item()
                            
                        info = ','.join(['{}={:.3f}'.format(k, v) for k,v in loss.items()])
                        tk0.set_postfix_str(info)

                    epoch_loss = {k:v/(i+1) for k, v in epoch_loss.items()}
                    epoch_info = ','.join(['{}={:.3f}'.format(k, v) for k,v in epoch_loss.items()])
                    tq.set_postfix_str(epoch_info) 


                elif mode == 'h':

                    for i, (x, y, idx) in tk0:#①y表示领域的id，0表示该条数据来自第一个数据集，1表示该条数据来自第二个数据集y=tensor([0, 1, 0, 1...])。   ②idx表示该条数据的索引号idx=tensor([ 9371,  6514,  9710]).
                        idx_origin = idx
                        #TODO 对x，y，idx的值有疑问,@see:data_loader.py, 143rows, x.shape=torch.Size([22517, 4000]),y.shape=torch.Size([22517]),idx.shape=torch.Size([22517])
                        # (x, y, idx) in tk0:,  @see:data_loader.py,143rows.
                        #print(f'###vae.py#415rows#################(x.shape, y.shape, idx.shape)===x.shape={x.shape},y.shape={y.shape},idx.shape={idx.shape}')
                        #print(f'###vae.py#416行################self.n_domain==={self.n_domain}') #self.n_domain===2
                        #print(f'###vae.py#427行################num_gene==={num_gene}') #num_gene===[2000, 2000, 2000]
                        x_c, y = x[:, 0:num_gene[self.n_domain]].float().to(device), y.long().to(device)  
                        idx = idx.to(device)
                                                    
                        loc_ref = torch.where(y==self.ref_id)[0] #用于根据给定条件选择元素
                        #TODO num_cell[0:self.ref_id]是什么, num_cell[0:self.ref_id]===[]，num_cell=[11259, 11259]，loc_ref=tensor([    0,     1,     2,  ..., 22510, 22512, 22514]
                        #print(f'###vae.py#423rows############num_cell[0:self.ref_id]==={num_cell[0:self.ref_id]}，num_cell={num_cell}，loc_ref.shape={loc_ref.shape}')
                        idx_ref = idx[loc_ref] - sum(num_cell[0:self.ref_id])#TODO，这个代码是什么意思？loc_ref是索引,loc_ref=tensor([  0,   1,   2,   3...])
                        #print(f'###vae.py#425rows####################idx_ref.shape={idx_ref.shape}')#idx_ref是找到self.ref_id的索引号，idx_ref=tensor([ 9371,  6514,  9710,...])
                        loc_query = {}
                        idx_query = {}
                        tran_batch = {}
                        Prior_batch = None

                        query_id = list(range(self.n_domain))
                        query_id.remove(self.ref_id)

                        if len(loc_ref) > 0:
                            for j in query_id:

                                loc_query[j] = torch.where(y==j)[0]
                                #print(f'####vae.py#440rows############j={j},loc_query[j]={loc_query[j]}')
                                idx_query[j] = idx[loc_query[j]] - sum(num_cell[0:j])

                                if save_OT:
                                    if len(idx_query[j]) != 0:
                                        if (len(idx_query[j])) == 1:
                                            tran_batch[j] = torch.from_numpy(tran[j]).to(device)[idx_query[j]][idx_ref]
                                        else:
                                            tran_batch[j] = torch.from_numpy(tran[j]).to(device)[idx_query[j]][:,idx_ref]
                                else:
                                    tran_batch[j] = None

                                if Prior is not None:
                                    Prior_batch = Prior[j][idx_query[j]][:,idx_ref].to(device)

                        ot_loss = torch.tensor(0.0).to(device)
                        recon_loss = torch.tensor(0.0).to(device)
                        kl_loss = torch.tensor(0.0).to(device)
                        drug_response_loss = torch.tensor(0.0).to(device)

                        loc = loc_query
                        loc[self.ref_id] = loc_ref
                        idx = idx_query
                        idx[self.ref_id] = idx_ref #x_c.shape=torch.Size([256, 2000])
                        #TODO，@see：vae.py, 148rows
                        z, mu, var = self.encoder(x_c, 0) #z.shape=torch.Size([256, 16]),mu.shape=torch.Size([256, 16]),var.shape=torch.Size([256, 16])
                        #TODO,疑问，self.encoder(x_c, 0)==x_c.shape=torch.Size([22517, 2000])，z.shape=torch.Size([22517, 16])，mu.shape=torch.Size([22517, 16])，var.shape=torch.Size([22517, 16])
                        #print(f'###vae.y#474rows##################self.encoder(x_c, 0)==x_c.shape={x_c.shape},z.shape={z.shape},mu.shape={mu.shape},var.shape={var.shape}')
                        recon_x_c = self.decoder(z, 0, y)        
   
                        if label_weight is None: # default, label_weight is None
                            #recon_loss = loss_func(recon_x_c, x_c) * 2000 # default, loss_func = nn.BCELoss()#TODO，这里为什么要乘2000，为什么BCEloss损失为负数
                            #recon_loss = loss_func(recon_x_c, x_c) * num_gene[self.n_domain]
                            recon_loss = loss_func(recon_x_c, x_c) 
                            #recon_loss = abs(loss_func(recon_x_c, x_c)) * 2000 # default, loss_func = nn.BCELoss()#TODO，这里为什么要乘2000，为什么BCEloss损失为负数
                        else:
                            for j, weight in enumerate(label_weight):

                                if len(loc[j])>0:
                                    if weight is None:
                                        recon_loss += 1/self.n_domain * loss_func(recon_x_c[loc[j]], x_c[loc[j]]) * \
                                        2000
                                        # kl_loss += kl_div(mu[loc[j]], var[loc[j]])                               
                                    else:
                                        weight = weight.to(device)
                                        recon_loss += 1/self.n_domain * F.binary_cross_entropy(recon_x_c[loc[j]], x_c[loc[j]], weight=weight[idx[j]]) * \
                                        2000
                                        # kl_loss += kl_div(mu[loc[j]], var[loc[j]], weight[idx[j]])

                        kl_loss = kl_div(mu, var) 
                        if use_specific:

                            x_s = x[:, num_gene[self.n_domain]:].float().to(device) #x.shape=torch.Size([22517, 4000])

                            for j in range(self.n_domain):
                                if len(loc[j])>0:
                                    #TODO，loc[j]是什么？,loc[j]应该是索引loc[j]=tensor([  0,   1,   2,   3,...]，loc[j].shape=torch.Size([11258])， z[loc[j]].shape=torch.Size([11258, 16])，z.shape=torch.Size([22517, 16])
                                    #print(f'###vae.py#489rows###################loc[j].shape={loc[j].shape}，z[loc[j]].shape={z[loc[j]].shape}，z.shape={z.shape}，loc[j]={loc[j]}，loc={loc}')
                                    recon_x_s = self.decoder(z[loc[j]], j+1)
                                    recon_loss += lambda_s * loss_func(recon_x_s, x_s[loc[j]][:, 0:num_gene[j]]) * 2000 #计算特异性高表达基因的重建数据和原数据之间的损失函数

                        if drug_response:#TODO drug_response_decoder的bulk模型训练
                            for j in range(self.n_domain):
                                if len(loc[j])>0 and j==0: # 表示取数据集x的隐空间嵌入z（取bulk细胞系的隐空间嵌入z ）
                                    if adata_cm is not None:
                                        a = idx_origin[y==j] #定位到数据集x（bulk细胞系）的索引，用来拿到其相应的response label
                                        #groundtruth_bulk_label = adata_cm.obs['response'].iloc[a,].values.reshape(-1,1)
                                        groundtruth_bulk_label = adata_cm.obs['response'].iloc[a.tolist(),].values.reshape(-1,1)
                                        groundtruth_bulk_label = torch.Tensor(groundtruth_bulk_label).to(device)
                                        #print(f'####vae.py#509rows##################groundtruth_bulk_label.shape={groundtruth_bulk_label.shape},groundtruth_bulk_label[:5]={groundtruth_bulk_label[:5]}')
                                        predicted_bulk_label = self.decoder(z[loc[j]], j+2+1) # decoder(,0)表示所有共同数据的共同的解码器；decoder(,1)表示数据集X特异性高可变基因的解码器；decoder(,2)表示数据集Y特异性高可变基因的解码器；decoder(,3)表示所有共同数据里的数据X的药物响应解码器；
                                        #print(f'####vae.py#511rows##################predicted_bulk_label.shape={predicted_bulk_label.shape},predicted_bulk_label[:5]={predicted_bulk_label[:5]}')
                                        drug_response_loss = drug_response_loss = loss_func(predicted_bulk_label, groundtruth_bulk_label)#计算bulk细胞系药物响应真实标签和预测标签之间的损失函数
                        
                        if len(torch.unique(y))>1 and len(loc[self.ref_id])!=0:
                            
                            mu_dict = {}
                            var_dict = {}

                            for j in range(self.n_domain):
                                if len(loc[j])>0:
                                    mu_dict[j] = mu[loc[j]]
                                    var_dict[j] = var[loc[j]]

                            for j in query_id:
                                if len(loc[j])>0:

                                    if label_weight is None:

                                        ot_loss_tmp, tran_batch[j] = unbalanced_ot(
                                            tran_batch[j],
                                            mu_dict[j], 
                                            var_dict[j], 
                                            mu_dict[self.ref_id].detach(), 
                                            var_dict[self.ref_id].detach(), 
                                            reg=reg,
                                            reg_m=reg_m,
                                            idx_q=idx_query[j],
                                            idx_r=idx_ref,
                                            Couple=Prior_batch, 
                                            device=device,
                                        )

                                    else:

                                        ot_loss_tmp, tran_batch[j] = unbalanced_ot(
                                            tran_batch[j],
                                            mu_dict[j], 
                                            var_dict[j], 
                                            mu_dict[self.ref_id].detach(), 
                                            var_dict[self.ref_id].detach(), 
                                            reg=reg,
                                            reg_m=reg_m,
                                            idx_q=idx_query[j],
                                            idx_r=idx_ref,
                                            Couple=Prior_batch, 
                                            device=device,
                                            query_weight=label_weight[j],
                                            ref_weight=label_weight[self.ref_id],
                                        )

                                    if save_OT:
                                        t0 = np.repeat(idx_query[j].cpu().numpy(), len(idx_ref)).reshape(len(idx_query[j]),len(idx_ref))
                                        t1 = np.tile(idx_ref.cpu().numpy(), (len(idx_query[j]), 1))
                                        tran[j][t0,t1] = tran_batch[j].cpu().numpy()

                                    ot_loss += ot_loss_tmp
   
                        loss = {'recloss':lambda_recon*recon_loss, 'klloss':lambda_kl*kl_loss, 'otloss':lambda_ot*ot_loss, 'drug_response_loss':lambda_response*drug_response_loss} #
                        
                        optim.zero_grad()
                        sum(loss.values()).backward()#TODO，疑问，可以这样子后向传播吗？
                        optim.step()

                        for k,v in loss.items():
                            epoch_loss[k] += loss[k].item() #TODO 疑问，loss[k].item()是什么, k=recloss,v=2528.068359375,loss.items()=dict_items([('recloss', tensor(...)), ('klloss', tensor(...)), ('otloss', tensor(...))])
                            #print(f'#######vae.py#555rows#############(k,v) in loss.items()====k={k},v={v}')

                        info = ','.join(['{}={:.2f}'.format(k, v) for k,v in loss.items()])
                        tk0.set_postfix_str(info)
                    
                    epoch_loss = {k:v/(i+1) for k, v in epoch_loss.items()}

                    epoch_info = ','.join(['{}={:.2f}'.format(k, v) for k,v in epoch_loss.items()])
                    tq.set_postfix_str(epoch_info) 
                        
                    early_stopping(sum(epoch_loss.values()), self)
                    if early_stopping.early_stop:
                        print('EarlyStopping: run {} epoch'.format(epoch+1))
                        break
                torch.cuda.empty_cache()

    def fit_1_unshared_decoder( # 使用EarlyStopping早停机制的时候（epoch不固定，由训练过程中损失函数不再下降为止），且不使用共享解码器，且未对源域bulk数据平衡采样，使用这个fit_1()做训练
            self, 
            dataloader,
            tran,
            num_cell,
            num_gene,
            mode='h',
            loss_type='BCE',
            label_weight=None,
            Prior=None,
            save_OT=False,
            use_specific=True,
            lambda_s=0.5,
            lambda_kl=0.5,
            lambda_recon=1.0,
            lambda_ot=1.0,
            lambda_response=1.0, #TODO,药物响应预测模型的权重参数
            reg=0.1,
            reg_m=1.0,
            lr=2e-4,
            max_iteration=30000,
            early_stopping=None,
            device='cuda:0',  
            verbose=False,
            drug_response=True, ###TODO,new feature
            adata_cm=None, ###TODO,new feature
            unshared_decoder=False,
            lambda_cell=1.0, #TODO,单细胞正则化损失函数的权重参数
            cell_regularization=False, ###TODO,new feature
            global_match=False, ###TODO,new feature
            mmd_GAMMA=1000, #Gamma parameter in the kernel of the MMD loss of the transfer learning, default: 1000
            lambda_mmd=1.0, ###TODO,new feature
        ):
        """
        train VAE

        Parameters
        ----------
        dataloader
            An iterable over the given dataset for training.
        tran
            A global OT plan. tran={} if save_OT=False in function.py.
        num_cell
            List of number of cells in different datasets. 
        num_gene
            List of number of genes in different datasets. 
        mode
            Choose from ['h', 'v', 'd']
            If 'h', integrate data with common genes
            If 'v', integrate data profiled from the same cells
            If 'd', inetrgate data without common genes
            Default: 'h'
        loss_type
            type of loss. Choose between ['BCE', 'MSE', 'L1']. Default: 'BCE'
        label_weight
            Prior-guided weighted vectors. Default: None
        Prior
            Prior correspondence matrix.
        save_OT
            If True, output a global OT plan. Default: False
        use_specific
            If True, specific genes in each dataset will be considered. Default: True
        lambda_s
            Balanced parameter for specific genes. Default: 0.5
        lambda_kl: 
            Balanced parameter for KL divergence. Default: 0.5
        lambda_recon:
            Balanced parameter for reconstruction. Default: 1.0
        lambda_ot:
            Balanced parameter for OT. Default: 1.0
        reg:
            Entropy regularization parameter in OT. Default: 0.1
        reg_m:
            Unbalanced OT parameter. Larger values means more balanced OT. Default: 1.0
        lr
            Learning rate. Default: 2e-4
        max_iteration
            Max iterations for training. Training one batch_size samples is one iteration. Default: 60000
        early_stopping
            EarlyStopping class (definite in utils.py) for stoping the training if loss doesn't improve after a given patience. Default: None
        device
            'cuda' or 'cpu' for training. Default: 'cuda'
        verbose
            Verbosity, True or False. Default: False
        drug_response
            if True, use drug_response decoder to predict drug response label. Default: True
        """

        self.to(device)
        # Set distribution loss
        def mmd_loss_func(x,y,GAMMA=mmd_GAMMA):
            result = mmd.mmd_loss(x,y,GAMMA)
            return result
        optim = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=5e-4)
        n_epoch = int(np.ceil(max_iteration/len(dataloader))) #用于向上取整，即返回不小于输入值的最小整数,len(dataloader)=1
        if loss_type == 'BCE':
            loss_func = nn.BCELoss()
        elif loss_type == 'MSE':
            loss_func = nn.MSELoss()
        elif loss_type == 'L1':
            loss_func = nn.L1Loss()
        
        with tqdm(range(n_epoch), total=n_epoch, desc='Epochs') as tq: # 用于在循环或迭代过程中显示进度条。它可以帮助你直观地了解代码的执行进度。
            for epoch in tq:
                tk0 = tqdm(enumerate(dataloader), total=len(dataloader), leave=False, desc='Iterations', disable=(not verbose))
                epoch_loss = defaultdict(float)
                if mode == 'h':
                    for i, (x, y, idx) in tk0:#①y表示领域的id，0表示该条数据来自第一个数据集，1表示该条数据来自第二个数据集y=tensor([0, 1, 0, 1...])。   ②idx表示该条数据的索引号idx=tensor([ 9371,  6514,  9710]).
                        idx_origin = idx
                        x_c, y = x[:, 0:num_gene[self.n_domain]].float().to(device), y.long().to(device)  
                        idx = idx.to(device)
                                                    
                        loc_ref = torch.where(y==self.ref_id)[0] #用于根据给定条件选择元素
                        idx_ref = idx[loc_ref] - sum(num_cell[0:self.ref_id])#TODO，这个代码是什么意思？loc_ref是索引,loc_ref=tensor([  0,   1,   2,   3...])
                        #print(f'###vae.py#425rows####################idx_ref.shape={idx_ref.shape}')#idx_ref是找到self.ref_id的索引号，idx_ref=tensor([ 9371,  6514,  9710,...])
                        loc_query = {}
                        idx_query = {}
                        tran_batch = {}
                        Prior_batch = None

                        query_id = list(range(self.n_domain))
                        query_id.remove(self.ref_id)

                        if len(loc_ref) > 0:
                            for j in query_id:

                                loc_query[j] = torch.where(y==j)[0]
                                #print(f'####vae.py#440rows############j={j},loc_query[j]={loc_query[j]}')
                                idx_query[j] = idx[loc_query[j]] - sum(num_cell[0:j])

                                if save_OT:
                                    if len(idx_query[j]) != 0:
                                        if (len(idx_query[j])) == 1:
                                            tran_batch[j] = torch.from_numpy(tran[j]).to(device)[idx_query[j]][idx_ref]
                                        else:
                                            tran_batch[j] = torch.from_numpy(tran[j]).to(device)[idx_query[j]][:,idx_ref]
                                else:
                                    tran_batch[j] = None

                                if Prior is not None:
                                    Prior_batch = Prior[j][idx_query[j]][:,idx_ref].to(device)

                        ot_loss = torch.tensor(0.0).to(device)
                        mmd_loss = torch.tensor(0.0).to(device)
                        cell_regularization_loss = torch.tensor(0.0).to(device)
                        recon_loss = torch.tensor(0.0).to(device)
                        recon_loss_1 = torch.tensor(0.0).to(device)
                        recon_loss_2 = torch.tensor(0.0).to(device)
                        kl_loss = torch.tensor(0.0).to(device)
                        drug_response_loss = torch.tensor(0.0).to(device)

                        loc = loc_query
                        loc[self.ref_id] = loc_ref
                        idx = idx_query
                        idx[self.ref_id] = idx_ref #x_c.shape=torch.Size([256, 2000])
                        z, mu, var = self.encoder(x_c, 0) #z.shape=torch.Size([256, 16]),mu.shape=torch.Size([256, 16]),var.shape=torch.Size([256, 16])
                        # 计算全局匹配损失mmd
                        if global_match:
                            mmd_loss = mmd_loss_func(z[loc[0]], z[loc[1]])
                        # 计算细胞正则化损失
                        if cell_regularization:# use cell regularization
                            if z[loc[1]].shape[0]<10: # target data（single cell dataset）
                                next
                            else:
                                edgeList = calculateKNNgraphDistanceMatrix(z[loc[1]].cpu().detach().numpy(), distanceType='euclidean', k=10)
                                listResult, size = generateLouvainCluster(edgeList)
                                # sc sim loss
                                loss_s = 0
                                for i in range(size):
                                    #print(i)
                                    s = cosine_similarity(x_c[loc[1]][np.asarray(listResult) == i,:].cpu().detach().numpy())
                                    s = 1-s
                                    loss_s += np.sum(np.triu(s,1))/((s.shape[0]*s.shape[0])*2-s.shape[0])
                                cell_regularization_loss += loss_s
                                
                        if label_weight is None: # default, label_weight is None
                            recon_x_bulk = self.decoder(z[loc[0]], 4, None)
                            recon_x_sc = self.decoder(z[loc[1]], 5, None)
                            recon_loss_1 = loss_func(recon_x_bulk, x_c[loc[0]])
                            recon_loss_2 = loss_func(recon_x_sc, x_c[loc[1]])
                            recon_loss = recon_loss_1 + recon_loss_2

                        kl_loss = kl_div(mu, var) 
                        if drug_response: #TODO drug_response_decoder的bulk模型训练
                            for j in range(self.n_domain):
                                if len(loc[j])>0 and j==0: # 表示取数据集x的隐空间嵌入z（取bulk细胞系的隐空间嵌入z ）
                                    if adata_cm is not None:
                                        a = idx_origin[y==j] #定位到数据集x（bulk细胞系）的索引，用来拿到其相应的response label
                                        #groundtruth_bulk_label = adata_cm.obs['response'].iloc[a,].values.reshape(-1,1)
                                        groundtruth_bulk_label = adata_cm.obs['response'].iloc[a.tolist(),].values.reshape(-1,1)
                                        groundtruth_bulk_label = torch.Tensor(groundtruth_bulk_label).to(device)
                                        #print(f'####vae.py#509rows##################groundtruth_bulk_label.shape={groundtruth_bulk_label.shape},groundtruth_bulk_label[:5]={groundtruth_bulk_label[:5]}')
                                        predicted_bulk_label = self.decoder(z[loc[j]], j+2+1) # decoder(,0)表示所有共同数据的共同的解码器；decoder(,1)表示数据集X特异性高可变基因的解码器；decoder(,2)表示数据集Y特异性高可变基因的解码器；decoder(,3)表示所有共同数据里的数据X的药物响应解码器；
                                        #print(f'####vae.py#511rows##################predicted_bulk_label.shape={predicted_bulk_label.shape},predicted_bulk_label[:5]={predicted_bulk_label[:5]}')
                                        drug_response_loss = drug_response_loss = loss_func(predicted_bulk_label, groundtruth_bulk_label)#计算bulk细胞系药物响应真实标签和预测标签之间的损失函数
                        
                        if len(torch.unique(y))>1 and len(loc[self.ref_id])!=0:
                            
                            mu_dict = {}
                            var_dict = {}

                            for j in range(self.n_domain):
                                if len(loc[j])>0:
                                    mu_dict[j] = mu[loc[j]]
                                    var_dict[j] = var[loc[j]]

                            for j in query_id:
                                if len(loc[j])>0:

                                    if label_weight is None:

                                        ot_loss_tmp, tran_batch[j] = unbalanced_ot(
                                            tran_batch[j],
                                            mu_dict[j], 
                                            var_dict[j], 
                                            mu_dict[self.ref_id].detach(), 
                                            var_dict[self.ref_id].detach(), 
                                            reg=reg,
                                            reg_m=reg_m,
                                            idx_q=idx_query[j],
                                            idx_r=idx_ref,
                                            Couple=Prior_batch, 
                                            device=device,
                                        )

                                    else:

                                        ot_loss_tmp, tran_batch[j] = unbalanced_ot(
                                            tran_batch[j],
                                            mu_dict[j], 
                                            var_dict[j], 
                                            mu_dict[self.ref_id].detach(), 
                                            var_dict[self.ref_id].detach(), 
                                            reg=reg,
                                            reg_m=reg_m,
                                            idx_q=idx_query[j],
                                            idx_r=idx_ref,
                                            Couple=Prior_batch, 
                                            device=device,
                                            query_weight=label_weight[j],
                                            ref_weight=label_weight[self.ref_id],
                                        )

                                    if save_OT:
                                        t0 = np.repeat(idx_query[j].cpu().numpy(), len(idx_ref)).reshape(len(idx_query[j]),len(idx_ref))
                                        t1 = np.tile(idx_ref.cpu().numpy(), (len(idx_query[j]), 1))
                                        tran[j][t0,t1] = tran_batch[j].cpu().numpy()

                                    ot_loss += ot_loss_tmp
   
                        loss = {'recloss':lambda_recon*recon_loss, 'klloss':lambda_kl*kl_loss, 'otloss':lambda_ot*ot_loss, 'drug_response_loss':lambda_response*drug_response_loss} #
                        if cell_regularization:
                            loss['cell_R_loss'] = lambda_cell*cell_regularization
                        if global_match:
                            loss['mmd_loss'] = lambda_mmd*mmd_loss
                        optim.zero_grad()
                        sum(loss.values()).backward()#TODO，疑问，可以这样子后向传播吗？
                        optim.step()

                        for k,v in loss.items():
                            epoch_loss[k] += loss[k].item() #TODO 疑问，loss[k].item()是什么, k=recloss,v=2528.068359375,loss.items()=dict_items([('recloss', tensor(...)), ('klloss', tensor(...)), ('otloss', tensor(...))])
                            #print(f'#######vae.py#555rows#############(k,v) in loss.items()====k={k},v={v}')

                        info = ','.join(['{}={:.2f}'.format(k, v) for k,v in loss.items()])
                        tk0.set_postfix_str(info)
                    
                    epoch_loss = {k:v/(i+1) for k, v in epoch_loss.items()}

                    epoch_info = ','.join(['{}={:.2f}'.format(k, v) for k,v in epoch_loss.items()])
                    tq.set_postfix_str(epoch_info) 
                        
                    early_stopping(sum(epoch_loss.values()), self)
                    if early_stopping.early_stop:
                        print('EarlyStopping: run {} epoch'.format(epoch+1))
                        break
                torch.cuda.empty_cache()

    
    
    def fit2( # 不使用早停机制的时候（epoch固定），且对源域bulk数据平衡采样，使用这个fit2()做训练
            self, 
            #dataloader, 
            Ctrainloader, #TODO new feature
            Ptrainloader, #TODO new feature
            tran,
            num_cell,
            num_gene,
            mode='h',
            loss_type='BCE',
            label_weight=None,
            Prior=None,
            save_OT=False,
            use_specific=True,
            lambda_s=0.5,
            lambda_kl=0.5,
            lambda_recon=1.0,
            lambda_ot=1.0,
            lambda_response=1.0, #TODO,药物响应预测模型的权重参数
            lambda_cell=1.0, #TODO,单细胞正则化损失函数的权重参数
            reg=0.1,
            reg_m=1.0,
            lr=2e-4,
            device='cuda:0',  
            verbose=False,
            drug_response=True, ###TODO,new feature
            cell_regularization=False, ###TODO,new feature
            adata_cm=None, ###TODO,new feature
            n_epoch=1000,
            global_match=False, ###TODO,new feature
            mmd_GAMMA=1000, #Gamma parameter in the kernel of the MMD loss of the transfer learning, default: 1000
            lambda_mmd=1.0, ###TODO,new feature
        ):
        """
        train VAE

        Parameters
        ----------
        dataloader
            An iterable over the given dataset for training.
        tran
            A global OT plan. tran={} if save_OT=False in function.py.
        num_cell
            List of number of cells in different datasets. 
        num_gene
            List of number of genes in different datasets. 
        mode
            Choose from ['h', 'v', 'd']
            If 'h', integrate data with common genes
            If 'v', integrate data profiled from the same cells
            If 'd', inetrgate data without common genes
            Default: 'h'
        loss_type
            type of loss. Choose between ['BCE', 'MSE', 'L1']. Default: 'BCE'
        label_weight
            Prior-guided weighted vectors. Default: None
        Prior
            Prior correspondence matrix.
        save_OT
            If True, output a global OT plan. Default: False
        use_specific
            If True, specific genes in each dataset will be considered. Default: True
        lambda_s
            Balanced parameter for specific genes. Default: 0.5
        lambda_kl: 
            Balanced parameter for KL divergence. Default: 0.5
        lambda_recon:
            Balanced parameter for reconstruction. Default: 1.0
        lambda_ot:
            Balanced parameter for OT. Default: 1.0
        reg:
            Entropy regularization parameter in OT. Default: 0.1
        reg_m:
            Unbalanced OT parameter. Larger values means more balanced OT. Default: 1.0
        lr
            Learning rate. Default: 2e-4
        max_iteration
            Max iterations for training. Training one batch_size samples is one iteration. Default: 60000
        early_stopping
            EarlyStopping class (definite in utils.py) for stoping the training if loss doesn't improve after a given patience. Default: None
        device
            'cuda' or 'cpu' for training. Default: 'cuda'
        verbose
            Verbosity, True or False. Default: False
        drug_response
            if True, use drug_response decoder to predict drug response label. Default: True
        """

        self.to(device)
        # Set distribution loss
        def mmd_loss_func(x,y,GAMMA=mmd_GAMMA):
            result = mmd.mmd_loss(x,y,GAMMA)
            return result
        optim = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=5e-4)


        if loss_type == 'BCE':
            loss_func = nn.BCELoss()
        elif loss_type == 'MSE':
            loss_func = nn.MSELoss()
        elif loss_type == 'L1':
            loss_func = nn.L1Loss()
        print(f'####vae.py#839row，n_epoch={n_epoch}')
        with tqdm(range(n_epoch), total=n_epoch, desc='Epochs') as tq: # 用于在循环或迭代过程中显示进度条。它可以帮助你直观地了解代码的执行进度。
            for epoch in tq:
                #TODO，dataloader.dataset是什么？，dataloader.dataset.shape=(22518, 4000)
                #print(f'####vae.py#279行####################dataloader.dataset.shape={dataloader.dataset.shape}')
                #tk0 = tqdm(enumerate(dataloader), total=len(dataloader), leave=False, desc='Iterations', disable=(not verbose))
                tk0 = tqdm(enumerate(zip(Ctrainloader,cycle(Ptrainloader))),total=len(Ctrainloader), leave=False, desc='Iterations', disable=(not verbose))
                epoch_loss = defaultdict(float)

                if mode == 'h':

                    '''for i, (x, y, idx) in tk0:#①y表示领域的id，0表示该条数据来自第一个数据集，1表示该条数据来自第二个数据集y=tensor([0, 1, 0, 1...])。   ②idx表示该条数据的索引号idx=tensor([ 9371,  6514,  9710]).
                        idx_origin = idx
                        #TODO 对x，y，idx的值有疑问,@see:data_loader.py, 143rows, x.shape=torch.Size([22517, 4000]),y.shape=torch.Size([22517]),idx.shape=torch.Size([22517])
                        # (x, y, idx) in tk0:,  @see:data_loader.py,143rows.
                        #print(f'###vae.py#415rows#################(x.shape, y.shape, idx.shape)===x.shape={x.shape},y.shape={y.shape},idx.shape={idx.shape}')
                        #print(f'###vae.py#416行################self.n_domain==={self.n_domain}') #self.n_domain===2
                        #print(f'###vae.py#427行################num_gene==={num_gene}') #num_gene===[2000, 2000, 2000]
                        x_c, y = x[:, 0:num_gene[self.n_domain]].float().to(device), y.long().to(device)  
                        idx = idx.to(device)
                                                    
                        loc_ref = torch.where(y==self.ref_id)[0] #用于根据给定条件选择元素
                        #TODO num_cell[0:self.ref_id]是什么, num_cell[0:self.ref_id]===[]，num_cell=[11259, 11259]，loc_ref=tensor([    0,     1,     2,  ..., 22510, 22512, 22514]
                        #print(f'###vae.py#423rows############num_cell[0:self.ref_id]==={num_cell[0:self.ref_id]}，num_cell={num_cell}，loc_ref.shape={loc_ref.shape}')
                        idx_ref = idx[loc_ref] - sum(num_cell[0:self.ref_id])#TODO，这个代码是什么意思？loc_ref是索引,loc_ref=tensor([  0,   1,   2,   3...])
                        #print(f'###vae.py#425rows####################idx_ref.shape={idx_ref.shape}')#idx_ref是找到self.ref_id的索引号，idx_ref=tensor([ 9371,  6514,  9710,...])
                        loc_query = {}
                        idx_query = {}
                        tran_batch = {}
                        Prior_batch = None

                        query_id = list(range(self.n_domain))
                        query_id.remove(self.ref_id)

                        if len(loc_ref) > 0:
                            for j in query_id:

                                loc_query[j] = torch.where(y==j)[0]
                                #print(f'####vae.py#440rows############j={j},loc_query[j]={loc_query[j]}')
                                idx_query[j] = idx[loc_query[j]] - sum(num_cell[0:j])

                                if save_OT:
                                    if len(idx_query[j]) != 0:
                                        if (len(idx_query[j])) == 1:
                                            tran_batch[j] = torch.from_numpy(tran[j]).to(device)[idx_query[j]][idx_ref]
                                        else:
                                            tran_batch[j] = torch.from_numpy(tran[j]).to(device)[idx_query[j]][:,idx_ref]
                                else:
                                    tran_batch[j] = None

                                if Prior is not None:
                                    Prior_batch = Prior[j][idx_query[j]][:,idx_ref].to(device)

                        ot_loss = torch.tensor(0.0).to(device)
                        recon_loss = torch.tensor(0.0).to(device)
                        kl_loss = torch.tensor(0.0).to(device)
                        drug_response_loss = torch.tensor(0.0).to(device)

                        loc = loc_query
                        loc[self.ref_id] = loc_ref
                        idx = idx_query
                        idx[self.ref_id] = idx_ref #x_c.shape=torch.Size([256, 2000])
                        #TODO，@see：vae.py, 148rows
                        z, mu, var = self.encoder(x_c, 0) #z.shape=torch.Size([256, 16]),mu.shape=torch.Size([256, 16]),var.shape=torch.Size([256, 16])
                        #TODO,疑问，self.encoder(x_c, 0)==x_c.shape=torch.Size([22517, 2000])，z.shape=torch.Size([22517, 16])，mu.shape=torch.Size([22517, 16])，var.shape=torch.Size([22517, 16])
                        #print(f'###vae.y#474rows##################self.encoder(x_c, 0)==x_c.shape={x_c.shape},z.shape={z.shape},mu.shape={mu.shape},var.shape={var.shape}')
                        recon_x_c = self.decoder(z, 0, y)        
   
                        if label_weight is None: # default, label_weight is None
                            #recon_loss = loss_func(recon_x_c, x_c) * 2000 # default, loss_func = nn.BCELoss()#TODO，这里为什么要乘2000，为什么BCEloss损失为负数
                            #recon_loss = loss_func(recon_x_c, x_c) * num_gene[self.n_domain]
                            recon_loss = loss_func(recon_x_c, x_c) 
                            #recon_loss = abs(loss_func(recon_x_c, x_c)) * 2000 # default, loss_func = nn.BCELoss()#TODO，这里为什么要乘2000，为什么BCEloss损失为负数
                        else:
                            for j, weight in enumerate(label_weight):

                                if len(loc[j])>0:
                                    if weight is None:
                                        recon_loss += 1/self.n_domain * loss_func(recon_x_c[loc[j]], x_c[loc[j]]) * \
                                        2000
                                        # kl_loss += kl_div(mu[loc[j]], var[loc[j]])                               
                                    else:
                                        weight = weight.to(device)
                                        recon_loss += 1/self.n_domain * F.binary_cross_entropy(recon_x_c[loc[j]], x_c[loc[j]], weight=weight[idx[j]]) * \
                                        2000
                                        # kl_loss += kl_div(mu[loc[j]], var[loc[j]], weight[idx[j]])

                        kl_loss = kl_div(mu, var) 
                        if use_specific:

                            x_s = x[:, num_gene[self.n_domain]:].float().to(device) #x.shape=torch.Size([22517, 4000])

                            for j in range(self.n_domain):
                                if len(loc[j])>0:
                                    #TODO，loc[j]是什么？,loc[j]应该是索引loc[j]=tensor([  0,   1,   2,   3,...]，loc[j].shape=torch.Size([11258])， z[loc[j]].shape=torch.Size([11258, 16])，z.shape=torch.Size([22517, 16])
                                    #print(f'###vae.py#489rows###################loc[j].shape={loc[j].shape}，z[loc[j]].shape={z[loc[j]].shape}，z.shape={z.shape}，loc[j]={loc[j]}，loc={loc}')
                                    recon_x_s = self.decoder(z[loc[j]], j+1)
                                    recon_loss += lambda_s * loss_func(recon_x_s, x_s[loc[j]][:, 0:num_gene[j]]) * 2000 #计算特异性高表达基因的重建数据和原数据之间的损失函数

                        if drug_response:#TODO drug_response_decoder的bulk模型训练
                            for j in range(self.n_domain):
                                if len(loc[j])>0 and j==0: # 表示取数据集x的隐空间嵌入z（取bulk细胞系的隐空间嵌入z ）
                                    if adata_cm is not None:
                                        a = idx_origin[y==j] #定位到数据集x（bulk细胞系）的索引，用来拿到其相应的response label
                                        groundtruth_bulk_label = adata_cm.obs['response'].iloc[a,].values.reshape(-1,1)
                                        groundtruth_bulk_label = torch.Tensor(groundtruth_bulk_label).to(device)
                                        #print(f'####vae.py#509rows##################groundtruth_bulk_label.shape={groundtruth_bulk_label.shape},groundtruth_bulk_label[:5]={groundtruth_bulk_label[:5]}')
                                        predicted_bulk_label = self.decoder(z[loc[j]], j+2+1) # decoder(,0)表示所有共同数据的共同的解码器；decoder(,1)表示数据集X特异性高可变基因的解码器；decoder(,2)表示数据集Y特异性高可变基因的解码器；decoder(,3)表示所有共同数据里的数据X的药物响应解码器；
                                        #print(f'####vae.py#511rows##################predicted_bulk_label.shape={predicted_bulk_label.shape},predicted_bulk_label[:5]={predicted_bulk_label[:5]}')
                                        drug_response_loss = drug_response_loss = loss_func(predicted_bulk_label, groundtruth_bulk_label)#计算bulk细胞系药物响应真实标签和预测标签之间的损失函数
                        
                        if len(torch.unique(y))>1 and len(loc[self.ref_id])!=0:
                            
                            mu_dict = {}
                            var_dict = {}

                            for j in range(self.n_domain):
                                if len(loc[j])>0:
                                    mu_dict[j] = mu[loc[j]]
                                    var_dict[j] = var[loc[j]]

                            for j in query_id:
                                if len(loc[j])>0:

                                    if label_weight is None:

                                        ot_loss_tmp, tran_batch[j] = unbalanced_ot(
                                            tran_batch[j],
                                            mu_dict[j], 
                                            var_dict[j], 
                                            mu_dict[self.ref_id].detach(), 
                                            var_dict[self.ref_id].detach(), 
                                            reg=reg,
                                            reg_m=reg_m,
                                            idx_q=idx_query[j],
                                            idx_r=idx_ref,
                                            Couple=Prior_batch, 
                                            device=device,
                                        )

                                    else:

                                        ot_loss_tmp, tran_batch[j] = unbalanced_ot(
                                            tran_batch[j],
                                            mu_dict[j], 
                                            var_dict[j], 
                                            mu_dict[self.ref_id].detach(), 
                                            var_dict[self.ref_id].detach(), 
                                            reg=reg,
                                            reg_m=reg_m,
                                            idx_q=idx_query[j],
                                            idx_r=idx_ref,
                                            Couple=Prior_batch, 
                                            device=device,
                                            query_weight=label_weight[j],
                                            ref_weight=label_weight[self.ref_id],
                                        )

                                    if save_OT:
                                        t0 = np.repeat(idx_query[j].cpu().numpy(), len(idx_ref)).reshape(len(idx_query[j]),len(idx_ref))
                                        t1 = np.tile(idx_ref.cpu().numpy(), (len(idx_query[j]), 1))
                                        tran[j][t0,t1] = tran_batch[j].cpu().numpy()

                                    ot_loss += ot_loss_tmp
   
                        loss = {'recloss':lambda_recon*recon_loss, 'klloss':lambda_kl*kl_loss, 'otloss':lambda_ot*ot_loss, 'drug_response_loss':lambda_response*drug_response_loss} #
                        
                        optim.zero_grad()
                        sum(loss.values()).backward()#TODO，疑问，可以这样子后向传播吗？
                        optim.step()

                        for k,v in loss.items():
                            epoch_loss[k] += loss[k].item() #TODO 疑问，loss[k].item()是什么, k=recloss,v=2528.068359375,loss.items()=dict_items([('recloss', tensor(...)), ('klloss', tensor(...)), ('otloss', tensor(...))])
                            #print(f'#######vae.py#555rows#############(k,v) in loss.items()====k={k},v={v}')

                        info = ','.join(['{}={:.2f}'.format(k, v) for k,v in loss.items()])
                        tk0.set_postfix_str(info)
                    '''
                    for i, data in tk0:#①y表示领域的id，0表示该条数据来自第一个数据集bulk，1表示该条数据来自第二个数据集sc,y=tensor([0, 1, 0, 1...])。   ②idx表示该条数据的索引号idx=tensor([ 9371,  6514,  9710]).
                        # print(f'####vae.py#1094row,epoch==={epoch}')
                        # 1. batch里 的bulk数据
                        x_bulk = data[0][0]
                        response_bulk = data[0][1].float()
                        y_bulk = torch.tensor([0] * data[0][0].shape[0]) # bulk数据的domain_id设置为0
                        idx_bulk = data[0][2]
                        # 2. batch里 的sc数据
                        x_sc = data[1][0]
                        response_sc = data[1][1].float()
                        y_sc = torch.tensor([1] * data[1][0].shape[0]) # sc数据的domain_id设置为1
                        idx_sc = data[1][2] + num_cell[0]
                        # 使用 torch.cat() 函数在第一个维度上进行连接
                        x = torch.cat((x_bulk, x_sc), dim=0)
                        y = torch.cat((y_bulk, y_sc), dim=0)
                        idx = torch.cat((idx_bulk, idx_sc), dim=0)
                        response = torch.cat((response_bulk, response_sc), dim=0)

                        idx_origin = idx
                        x_c, y = x.float().to(device), y.long().to(device)
                        # x_c, y = x[:,:].float().to(device), y.long().to(device) #TODO 使用数据集x（bulk细胞系）所有的基因
                        idx = idx.to(device)

                        loc_ref = torch.where(y==self.ref_id)[0] #用于根据给定条件选择元素
                        #TODO num_cell[0:self.ref_id]是什么, num_cell[0:self.ref_id]===[]，num_cell=[11259, 11259]，loc_ref=tensor([    0,     1,     2,  ..., 22510, 22512, 22514]
                        # print(f'###vae.py#423rows############num_cell[0:self.ref_id]==={num_cell[0:self.ref_id]}，num_cell={num_cell}，loc_ref.shape={loc_ref.shape}')
                        idx_ref = idx[loc_ref] - sum(num_cell[0:self.ref_id])#TODO，这个代码是什么意思？loc_ref是索引,loc_ref=tensor([  0,   1,   2,   3...])
                        # print(f'###vae.py#425rows####################idx_ref.shape={idx_ref.shape}')#idx_ref是找到self.ref_id的索引号，idx_ref=tensor([ 9371,  6514,  9710,...])
                        loc_query = {}
                        idx_query = {}
                        tran_batch = {}
                        Prior_batch = None

                        query_id = list(range(self.n_domain))
                        query_id.remove(self.ref_id)

                        if len(loc_ref) > 0:
                            for j in query_id:

                                loc_query[j] = torch.where(y==j)[0]
                                # print(f'####vae.py#440rows############j={j},loc_query[j]={loc_query[j]}')
                                idx_query[j] = idx[loc_query[j]] - sum(num_cell[0:j])

                                if save_OT:
                                    if len(idx_query[j]) != 0:
                                        if (len(idx_query[j])) == 1:
                                            tran_batch[j] = torch.from_numpy(tran[j]).to(device)[idx_query[j]][idx_ref]
                                        else:
                                            tran_batch[j] = torch.from_numpy(tran[j]).to(device)[idx_query[j]][:,idx_ref]
                                else:
                                    tran_batch[j] = None
                                if Prior is not None:
                                    Prior_batch = Prior[j][idx_query[j]][:,idx_ref].to(device)

                        ot_loss = torch.tensor(0.0).to(device)
                        mmd_loss = torch.tensor(0.0).to(device)
                        recon_loss = torch.tensor(0.0).to(device)
                        kl_loss = torch.tensor(0.0).to(device)
                        drug_response_loss = torch.tensor(0.0).to(device)
                        cell_regularization_loss = torch.tensor(0.0).to(device)

                        loc = loc_query
                        loc[self.ref_id] = loc_ref
                        idx = idx_query
                        idx[self.ref_id] = idx_ref #x_c.shape=torch.Size([256, 2000])
                        #TODO，@see：vae.py, 148rows
                        z, mu, var = self.encoder(x_c, 0) #z.shape=torch.Size([256, 16]),mu.shape=torch.Size([256, 16]),var.shape=torch.Size([256, 16])
                        #TODO,疑问，self.encoder(x_c, 0)==x_c.shape=torch.Size([22517, 2000])，z.shape=torch.Size([22517, 16])，mu.shape=torch.Size([22517, 16])，var.shape=torch.Size([22517, 16])
                        # print(f'###vae.y#474rows##################self.encoder(x_c, 0)==x_c.shape={x_c.shape},z.shape={z.shape},mu.shape={mu.shape},var.shape={var.shape}')
                        recon_x_c = self.decoder(z, 0, y)
                        # 计算全局匹配损失mmd
                        if global_match:
                            mmd_loss = mmd_loss_func(z[loc[0]], z[loc[1]])
                        # 计算细胞正则化损失
                        if cell_regularization:# use cell regularization
                            if z[loc[1]].shape[0]<10: # target data（single cell dataset）
                                next
                            else:
                                edgeList = calculateKNNgraphDistanceMatrix(z[loc[1]].cpu().detach().numpy(), distanceType='euclidean', k=10)
                                listResult, size = generateLouvainCluster(edgeList)
                                # sc sim loss
                                loss_s = 0
                                for i in range(size):
                                    #print(i)
                                    s = cosine_similarity(x_c[loc[1]][np.asarray(listResult) == i,:].cpu().detach().numpy())
                                    s = 1-s
                                    loss_s += np.sum(np.triu(s,1))/((s.shape[0]*s.shape[0])*2-s.shape[0])
                                cell_regularization_loss += loss_s

                        if label_weight is None: # default, label_weight is None
                            recon_loss = loss_func(recon_x_c, x_c)
                        else:
                            for j, weight in enumerate(label_weight):

                                if len(loc[j])>0:
                                    if weight is None:
                                        recon_loss += 1/self.n_domain * loss_func(recon_x_c[loc[j]], x_c[loc[j]]) * \
                                                      2000
                                        # kl_loss += kl_div(mu[loc[j]], var[loc[j]])
                                    else:
                                        weight = weight.to(device)
                                        recon_loss += 1/self.n_domain * F.binary_cross_entropy(recon_x_c[loc[j]], x_c[loc[j]], weight=weight[idx[j]]) * \
                                                      2000
                                        # kl_loss += kl_div(mu[loc[j]], var[loc[j]], weight[idx[j]])

                        kl_loss = kl_div(mu, var)
                        if use_specific:

                            x_s = x[:, num_gene[self.n_domain]:].float().to(device) #x.shape=torch.Size([22517, 4000])

                            for j in range(self.n_domain):
                                if len(loc[j])>0:
                                    #TODO，loc[j]是什么？,loc[j]应该是索引loc[j]=tensor([  0,   1,   2,   3,...]，loc[j].shape=torch.Size([11258])， z[loc[j]].shape=torch.Size([11258, 16])，z.shape=torch.Size([22517, 16])
                                    # print(f'###vae.py#489rows###################loc[j].shape={loc[j].shape}，z[loc[j]].shape={z[loc[j]].shape}，z.shape={z.shape}，loc[j]={loc[j]}，loc={loc}')
                                    recon_x_s = self.decoder(z[loc[j]], j+1)
                                    recon_loss += lambda_s * loss_func(recon_x_s, x_s[loc[j]][:, 0:num_gene[j]]) * 2000 #计算特异性高表达基因的重建数据和原数据之间的损失函数

                        if drug_response:#TODO drug_response_decoder的bulk模型训练
                            for j in range(self.n_domain):
                                if len(loc[j])>0 and j==0: # 表示取数据集x的隐空间嵌入z（取bulk细胞系的隐空间嵌入z ）
                                    if adata_cm is not None:
                                        groundtruth_bulk_label = response_bulk.reshape(-1,1)
                                        groundtruth_bulk_label = torch.Tensor(groundtruth_bulk_label).to(device)
                                        # print(f'####vae.py#509rows##################groundtruth_bulk_label.shape={groundtruth_bulk_label.shape},groundtruth_bulk_label[:5]={groundtruth_bulk_label[:5]}')
                                        predicted_bulk_label = self.decoder(z[loc[j]], j+2+1) # decoder(,0)表示所有共同数据的共同的解码器；decoder(,1)表示数据集X特异性高可变基因的解码器；decoder(,2)表示数据集Y特异性高可变基因的解码器；decoder(,3)表示所有共同数据里的数据X的药物响应解码器；
                                        # print(f'####vae.py#511rows##################predicted_bulk_label.shape={predicted_bulk_label.shape},predicted_bulk_label[:5]={predicted_bulk_label[:5]}')
                                        drug_response_loss = loss_func(predicted_bulk_label, groundtruth_bulk_label)#计算bulk细胞系药物响应真实标签和预测标签之间的损失函数

                        if len(torch.unique(y))>1 and len(loc[self.ref_id])!=0:

                            mu_dict = {}
                            var_dict = {}

                            for j in range(self.n_domain):
                                if len(loc[j])>0:
                                    mu_dict[j] = mu[loc[j]]
                                    var_dict[j] = var[loc[j]]

                            for j in query_id:
                                if len(loc[j])>0:

                                    if label_weight is None:

                                        ot_loss_tmp, tran_batch[j] = unbalanced_ot(
                                            tran_batch[j],
                                            mu_dict[j],
                                            var_dict[j],
                                            mu_dict[self.ref_id].detach(),
                                            var_dict[self.ref_id].detach(),
                                            reg=reg,
                                            reg_m=reg_m,
                                            idx_q=idx_query[j],# default query_weight is None,这个参数没用到
                                            idx_r=idx_ref, # default ref_weight is None, 这个参数函数也没用到
                                            Couple=Prior_batch, # default Prior is None, @see:881row, 所以Prior_batch也是None，这个参数函数没用到
                                            device=device,
                                        )

                                    else:

                                        ot_loss_tmp, tran_batch[j] = unbalanced_ot(
                                            tran_batch[j],
                                            mu_dict[j],
                                            var_dict[j],
                                            mu_dict[self.ref_id].detach(),
                                            var_dict[self.ref_id].detach(),
                                            reg=reg,
                                            reg_m=reg_m,
                                            idx_q=idx_query[j],
                                            idx_r=idx_ref,
                                            Couple=Prior_batch,
                                            device=device,
                                            query_weight=label_weight[j],
                                            ref_weight=label_weight[self.ref_id],
                                        )

                                    if save_OT:
                                        t0 = np.repeat(idx_query[j].cpu().numpy(), len(idx_ref)).reshape(len(idx_query[j]),len(idx_ref))
                                        t1 = np.tile(idx_ref.cpu().numpy(), (len(idx_query[j]), 1))
                                        tran[j][t0,t1] = tran_batch[j].cpu().numpy()

                                    ot_loss += ot_loss_tmp

                        if cell_regularization:
                            loss = {'recloss':lambda_recon*recon_loss, 'klloss':lambda_kl*kl_loss, 'otloss':lambda_ot*ot_loss, 'drug_R_loss':lambda_response*drug_response_loss, 'cell_R_loss':lambda_cell*cell_regularization_loss} #
                        else:
                            loss = {'recloss':lambda_recon*recon_loss, 'klloss':lambda_kl*kl_loss, 'otloss':lambda_ot*ot_loss, 'drug_response_loss':lambda_response*drug_response_loss} #

                        if global_match:
                            loss['mmd_loss'] = lambda_mmd*mmd_loss

                        optim.zero_grad()
                        sum(loss.values()).backward()#TODO，疑问，可以这样子后向传播吗？
                        optim.step()

                        for k,v in loss.items():
                            epoch_loss[k] += loss[k].item() #TODO 疑问，loss[k].item()是什么, k=recloss,v=2528.068359375,loss.items()=dict_items([('recloss', tensor(...)), ('klloss', tensor(...)), ('otloss', tensor(...))])
                            # print(f'#######vae.py#555rows#############(k,v) in loss.items()====k={k},v={v}')

                        info = ','.join(['{}={:.2f}'.format(k, v) for k,v in loss.items()])
                        tk0.set_postfix_str(info)
                    
                    epoch_loss = {k:v/(i+1) for k, v in epoch_loss.items()}

                    epoch_info = ','.join(['{}={:.2f}'.format(k, v) for k,v in epoch_loss.items()])
                    tq.set_postfix_str(epoch_info) 
                torch.cuda.empty_cache()           

    def fit2_unshared_decoder( # 不使用早停机制的时候（epoch固定），且对源域bulk数据平衡采样，，且不使用共享解码器
            self, 
            #dataloader, 
            Ctrainloader, #TODO new feature
            Ptrainloader, #TODO new feature
            tran,
            num_cell,
            num_gene,
            mode='h',
            loss_type='BCE',
            label_weight=None,
            Prior=None,
            save_OT=False,
            use_specific=True,
            lambda_s=0.5,
            lambda_kl=0.5,
            lambda_recon=1.0,
            lambda_ot=1.0,
            lambda_response=1.0, #TODO,药物响应预测模型的权重参数
            lambda_cell=1.0, #TODO,单细胞正则化损失函数的权重参数
            reg=0.1,
            reg_m=1.0,
            lr=2e-4,
            device='cuda:0',  
            verbose=False,
            drug_response=True, ###TODO,new feature
            cell_regularization=False, ###TODO,new feature
            adata_cm=None, ###TODO,new feature
            adatas=None, ###TODOq,new feature
            unshared_decoder=False,
            n_epoch=1000,
            global_match=False, ###TODO,new feature
            mmd_GAMMA=1000.0, #Gamma parameter in the kernel of the MMD loss of the transfer learning, default: 1000
            lambda_mmd=1.0, ###TODO,new feature
        ):
        """
        train VAE

        Parameters
        ----------
        dataloader
            An iterable over the given dataset for training.
        tran
            A global OT plan. tran={} if save_OT=False in function.py.
        num_cell
            List of number of cells in different datasets. 
        num_gene
            List of number of genes in different datasets. 
        mode
            Choose from ['h', 'v', 'd']
            If 'h', integrate data with common genes
            If 'v', integrate data profiled from the same cells
            If 'd', inetrgate data without common genes
            Default: 'h'
        loss_type
            type of loss. Choose between ['BCE', 'MSE', 'L1']. Default: 'BCE'
        label_weight
            Prior-guided weighted vectors. Default: None
        Prior
            Prior correspondence matrix.
        save_OT
            If True, output a global OT plan. Default: False
        use_specific
            If True, specific genes in each dataset will be considered. Default: True
        lambda_s
            Balanced parameter for specific genes. Default: 0.5
        lambda_kl: 
            Balanced parameter for KL divergence. Default: 0.5
        lambda_recon:
            Balanced parameter for reconstruction. Default: 1.0
        lambda_ot:
            Balanced parameter for OT. Default: 1.0
        reg:
            Entropy regularization parameter in OT. Default: 0.1
        reg_m:
            Unbalanced OT parameter. Larger values means more balanced OT. Default: 1.0
        lr
            Learning rate. Default: 2e-4
        max_iteration
            Max iterations for training. Training one batch_size samples is one iteration. Default: 60000
        early_stopping
            EarlyStopping class (definite in utils.py) for stoping the training if loss doesn't improve after a given patience. Default: None
        device
            'cuda' or 'cpu' for training. Default: 'cuda'
        verbose
            Verbosity, True or False. Default: False
        drug_response
            if True, use drug_response decoder to predict drug response label. Default: True
        """

        self.to(device)
        def mmd_loss_func(x,y,GAMMA=mmd_GAMMA):
            result = mmd.mmd_loss(x,y,GAMMA)
            return result
        optim = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=5e-4)

        if loss_type == 'BCE':
            loss_func = nn.BCELoss()
        elif loss_type == 'MSE':
            loss_func = nn.MSELoss()
        elif loss_type == 'L1':
            loss_func = nn.L1Loss()
        print(f'####vae.py#839row，n_epoch={n_epoch}')
        with tqdm(range(n_epoch), total=n_epoch, desc='Epochs') as tq: # 用于在循环或迭代过程中显示进度条。它可以帮助你直观地了解代码的执行进度。
            for epoch in tq:
                #TODOq，dataloader.dataset是什么？，dataloader.dataset.shape=(22518, 4000)
                #print(f'####vae.py#279行####################dataloader.dataset.shape={dataloader.dataset.shape}')
                #tk0 = tqdm(enumerate(dataloader), total=len(dataloader), leave=False, desc='Iterations', disable=(not verbose))
                tk0 = tqdm(enumerate(zip(Ctrainloader,cycle(Ptrainloader))),total=len(Ctrainloader), leave=False, desc='Iterations', disable=(not verbose))
                epoch_loss = defaultdict(float)

                if mode == 'h':
                    for i, data in tk0:#①y表示领域的id，0表示该条数据来自第一个数据集bulk，1表示该条数据来自第二个数据集sc,y=tensor([0, 1, 0, 1...])。   ②idx表示该条数据的索引号idx=tensor([ 9371,  6514,  9710]).
                        # print(f'####vae.py#1094row,epoch==={epoch}')
                        # 1. batch里 的bulk数据
                        x_bulk = data[0][0]
                        response_bulk = data[0][1].float()
                        y_bulk = torch.tensor([0] * data[0][0].shape[0]) # bulk数据的domain_id设置为0
                        idx_bulk = data[0][2]
                        # 2. batch里 的sc数据
                        x_sc = data[1][0]
                        response_sc = data[1][1].float()
                        y_sc = torch.tensor([1] * data[1][0].shape[0]) # sc数据的domain_id设置为1
                        idx_sc = data[1][2] + num_cell[0]
                        # 使用 torch.cat() 函数在第一个维度上进行连接
                        x = torch.cat((x_bulk, x_sc), dim=0) #TODO，这里将x_bulk和x_sc数据拼在了一起
                        y = torch.cat((y_bulk, y_sc), dim=0) #TODO，这里将源域的domain_id和目标域的domain_id拼在了一起
                        idx = torch.cat((idx_bulk, idx_sc), dim=0) #TODO，这里将源域的idx和目标域的idx合在了一起
                        response = torch.cat((response_bulk, response_sc), dim=0) #TODO，这里将源域的response_label和目标域的拼在了一起

                        idx_origin = idx
                        x_c, y = x.float().to(device), y.long().to(device)
                        # x_c, y = x[:,:].float().to(device), y.long().to(device) #TODO 使用数据集x（bulk细胞系）所有的基因
                        idx = idx.to(device)

                        loc_ref = torch.where(y==self.ref_id)[0] #用于根据给定条件选择元素
                        #TODOq num_cell[0:self.ref_id]是什么, num_cell[0:self.ref_id]===[]，num_cell=[11259, 11259]，loc_ref=tensor([    0,     1,     2,  ..., 22510, 22512, 22514]
                        # print(f'###vae.py#423rows############num_cell[0:self.ref_id]==={num_cell[0:self.ref_id]}，num_cell={num_cell}，loc_ref.shape={loc_ref.shape}')
                        idx_ref = idx[loc_ref] - sum(num_cell[0:self.ref_id])#TODOq，这个代码是什么意思？loc_ref是索引,loc_ref=tensor([  0,   1,   2,   3...])
                        # print(f'###vae.py#425rows####################idx_ref.shape={idx_ref.shape}')#idx_ref是找到self.ref_id的索引号，idx_ref=tensor([ 9371,  6514,  9710,...])
                        loc_query = {}
                        idx_query = {}
                        tran_batch = {}
                        Prior_batch = None

                        query_id = list(range(self.n_domain))
                        query_id.remove(self.ref_id)

                        if len(loc_ref) > 0:
                            for j in query_id:

                                loc_query[j] = torch.where(y==j)[0]
                                # print(f'####vae.py#440rows############j={j},loc_query[j]={loc_query[j]}')
                                idx_query[j] = idx[loc_query[j]] - sum(num_cell[0:j])

                                if save_OT:
                                    if len(idx_query[j]) != 0:
                                        if (len(idx_query[j])) == 1:
                                            tran_batch[j] = torch.from_numpy(tran[j]).to(device)[idx_query[j]][idx_ref]
                                        else:
                                            tran_batch[j] = torch.from_numpy(tran[j]).to(device)[idx_query[j]][:,idx_ref]
                                else:
                                    tran_batch[j] = None
                                if Prior is not None:
                                    Prior_batch = Prior[j][idx_query[j]][:,idx_ref].to(device)

                        ot_loss = torch.tensor(0.0).to(device)
                        mmd_loss = torch.tensor(0.0).to(device)
                        recon_loss = torch.tensor(0.0).to(device)
                        recon_loss_1 = torch.tensor(0.0).to(device)
                        recon_loss_2 = torch.tensor(0.0).to(device)
                        kl_loss = torch.tensor(0.0).to(device)
                        drug_response_loss = torch.tensor(0.0).to(device)
                        cell_regularization_loss = torch.tensor(0.0).to(device)

                        loc = loc_query
                        loc[self.ref_id] = loc_ref
                        idx = idx_query
                        idx[self.ref_id] = idx_ref #x_c.shape=torch.Size([256, 2000])
                        #TODO，@see：vae.py, 148rows，使用共享编码器
                        z, mu, var = self.encoder(x_c, 0) #z.shape=torch.Size([256, 16]),mu.shape=torch.Size([256, 16]),var.shape=torch.Size([256, 16])
                        # z = self.encoder(x_c, 0)[1] #z.shape=torch.Size([256, 16]),mu.shape=torch.Size([256, 16]),var.shape=torch.Size([256, 16])
                        # _, mu, var = self.encoder(x_c, 0) #z.shape=torch.Size([256, 16]),mu.shape=torch.Size([256, 16]),var.shape=torch.Size([256, 16])
                        #TODO,疑问，self.encoder(x_c, 0)==x_c.shape=torch.Size([22517, 2000])，z.shape=torch.Size([22517, 16])，mu.shape=torch.Size([22517, 16])，var.shape=torch.Size([22517, 16])
                        # print(f'###vae.y#474rows##################self.encoder(x_c, 0)==x_c.shape={x_c.shape},z.shape={z.shape},mu.shape={mu.shape},var.shape={var.shape}')
                        #recon_x_c = self.decoder(z, 0, y)
                        
                        # 计算全局匹配损失mmd
                        if global_match:
                            mmd_loss = mmd_loss_func(z[loc[0]], z[loc[1]])
                        # 计算细胞正则化损失
                        if cell_regularization:# use cell regularization
                            if z[loc[1]].shape[0]<10: # target data（single cell dataset）
                                next
                            else:
                                edgeList = calculateKNNgraphDistanceMatrix(z[loc[1]].cpu().detach().numpy(), distanceType='euclidean', k=10)
                                listResult, size = generateLouvainCluster(edgeList)
                                # sc sim loss
                                loss_s = 0
                                for i in range(size):
                                    #print(i)
                                    s = cosine_similarity(x_c[loc[1]][np.asarray(listResult) == i,:].cpu().detach().numpy())
                                    s = 1-s
                                    loss_s += np.sum(np.triu(s,1))/((s.shape[0]*s.shape[0])*2-s.shape[0])
                                cell_regularization_loss += loss_s

                        if label_weight is None: # default, label_weight is None
                            #TODO, 使用两个解码器，不使用共享解码器
                            recon_x_bulk = self.decoder(z[loc[0]], 4, None)
                            recon_x_sc = self.decoder(z[loc[1]], 5, None)
                            recon_loss_1 = loss_func(recon_x_bulk, x_bulk.float().to(device))
                            recon_loss_2 = loss_func(recon_x_sc, x_sc.float().to(device))
                            recon_loss = recon_loss_1 + recon_loss_2

                        kl_loss = kl_div(mu, var)
                        
                        if drug_response:#TODO drug_response_decoder的bulk模型训练
                            for j in range(self.n_domain):
                                if len(loc[j])>0 and j==0: # 表示取数据集x的隐空间嵌入z（取bulk细胞系的隐空间嵌入z ）
                                    if adata_cm is not None:
                                        groundtruth_bulk_label = response_bulk.reshape(-1,1)
                                        groundtruth_bulk_label = torch.Tensor(groundtruth_bulk_label).to(device)
                                        # print(f'####vae.py#509rows##################groundtruth_bulk_label.shape={groundtruth_bulk_label.shape},groundtruth_bulk_label[:5]={groundtruth_bulk_label[:5]}')
                                        predicted_bulk_label = self.decoder(z[loc[j]], j+2+1) # decoder(,0)表示所有共同数据的共同的解码器；decoder(,1)表示数据集X特异性高可变基因的解码器；decoder(,2)表示数据集Y特异性高可变基因的解码器；decoder(,3)表示所有共同数据里的数据X的药物响应解码器；
                                        # print(f'####vae.py#511rows##################predicted_bulk_label.shape={predicted_bulk_label.shape},predicted_bulk_label[:5]={predicted_bulk_label[:5]}')
                                        drug_response_loss = loss_func(predicted_bulk_label, groundtruth_bulk_label)#计算bulk细胞系药物响应真实标签和预测标签之间的损失函数

                        if len(torch.unique(y))>1 and len(loc[self.ref_id])!=0:

                            mu_dict = {}
                            var_dict = {}

                            for j in range(self.n_domain):
                                if len(loc[j])>0:
                                    mu_dict[j] = mu[loc[j]]
                                    var_dict[j] = var[loc[j]]

                            for j in query_id:
                                if len(loc[j])>0:

                                    if label_weight is None:

                                        ot_loss_tmp, tran_batch[j] = unbalanced_ot(
                                            tran_batch[j],
                                            mu_dict[j],
                                            var_dict[j],
                                            mu_dict[self.ref_id].detach(),
                                            var_dict[self.ref_id].detach(),
                                            reg=reg,
                                            reg_m=reg_m,
                                            idx_q=idx_query[j],# default query_weight is None,这个参数没用到
                                            idx_r=idx_ref, # default ref_weight is None, 这个参数函数也没用到
                                            Couple=Prior_batch, # default Prior is None, @see:881row, 所以Prior_batch也是None，这个参数函数没用到
                                            device=device,
                                        )

                                    else:

                                        ot_loss_tmp, tran_batch[j] = unbalanced_ot(
                                            tran_batch[j],
                                            mu_dict[j],
                                            var_dict[j],
                                            mu_dict[self.ref_id].detach(),
                                            var_dict[self.ref_id].detach(),
                                            reg=reg,
                                            reg_m=reg_m,
                                            idx_q=idx_query[j],
                                            idx_r=idx_ref,
                                            Couple=Prior_batch,
                                            device=device,
                                            query_weight=label_weight[j],
                                            ref_weight=label_weight[self.ref_id],
                                        )

                                    if save_OT:
                                        t0 = np.repeat(idx_query[j].cpu().numpy(), len(idx_ref)).reshape(len(idx_query[j]),len(idx_ref))
                                        t1 = np.tile(idx_ref.cpu().numpy(), (len(idx_query[j]), 1))
                                        tran[j][t0,t1] = tran_batch[j].cpu().numpy()

                                    ot_loss += ot_loss_tmp

                        if cell_regularization:
                            loss = {'recloss':lambda_recon*recon_loss, 'klloss':lambda_kl*kl_loss, 'otloss':lambda_ot*ot_loss, 'drug_R_loss':lambda_response*drug_response_loss, 'cell_R_loss':lambda_cell*cell_regularization_loss} #
                        else:
                            loss = {'recloss':lambda_recon*recon_loss, 'klloss':lambda_kl*kl_loss, 'otloss':lambda_ot*ot_loss, 'drug_response_loss':lambda_response*drug_response_loss} #
                        
                        if global_match:
                            loss['mmd_loss'] = lambda_mmd*mmd_loss
                            
                        optim.zero_grad()
                        sum(loss.values()).backward()#TODO，疑问，可以这样子后向传播吗？
                        optim.step()

                        for k,v in loss.items():
                            epoch_loss[k] += loss[k].item() #TODO 疑问，loss[k].item()是什么, k=recloss,v=2528.068359375,loss.items()=dict_items([('recloss', tensor(...)), ('klloss', tensor(...)), ('otloss', tensor(...))])
                            # print(f'#######vae.py#555rows#############(k,v) in loss.items()====k={k},v={v}')

                        info = ','.join(['{}={:.2f}'.format(k, v) for k,v in loss.items()])
                        tk0.set_postfix_str(info)
                    
                    epoch_loss = {k:v/(i+1) for k, v in epoch_loss.items()}

                    epoch_info = ','.join(['{}={:.2f}'.format(k, v) for k,v in epoch_loss.items()])
                    tq.set_postfix_str(epoch_info) 
                torch.cuda.empty_cache()           

    def fit2_unshared_encoder_decoder( # 不使用早停机制的时候（epoch固定），且对源域bulk数据平衡采样，，且不使用共享解码器
            self, 
            #dataloader, 
            Ctrainloader, #TODO new feature
            Ptrainloader, #TODO new feature
            tran,
            num_cell,
            num_gene,
            mode='h',
            loss_type='BCE',
            label_weight=None,
            Prior=None,
            save_OT=False,
            use_specific=True,
            lambda_s=0.5,
            lambda_kl=0.5,
            lambda_recon=1.0,
            lambda_ot=1.0,
            lambda_response=1.0, #TODO,药物响应预测模型的权重参数
            lambda_cell=1.0, #TODO,单细胞正则化损失函数的权重参数
            reg=0.1,
            reg_m=1.0,
            lr=2e-4,
            device='cuda:0',  
            verbose=False,
            drug_response=True, ###TODO,new feature
            cell_regularization=False, ###TODO,new feature
            adata_cm=None, ###TODO,new feature
            adatas=None, ###TODOq,new feature
            unshared_decoder=False,
            n_epoch=1000
        ):
        """
        train VAE

        Parameters
        ----------
        dataloader
            An iterable over the given dataset for training.
        tran
            A global OT plan. tran={} if save_OT=False in function.py.
        num_cell
            List of number of cells in different datasets. 
        num_gene
            List of number of genes in different datasets. 
        mode
            Choose from ['h', 'v', 'd']
            If 'h', integrate data with common genes
            If 'v', integrate data profiled from the same cells
            If 'd', inetrgate data without common genes
            Default: 'h'
        loss_type
            type of loss. Choose between ['BCE', 'MSE', 'L1']. Default: 'BCE'
        label_weight
            Prior-guided weighted vectors. Default: None
        Prior
            Prior correspondence matrix.
        save_OT
            If True, output a global OT plan. Default: False
        use_specific
            If True, specific genes in each dataset will be considered. Default: True
        lambda_s
            Balanced parameter for specific genes. Default: 0.5
        lambda_kl: 
            Balanced parameter for KL divergence. Default: 0.5
        lambda_recon:
            Balanced parameter for reconstruction. Default: 1.0
        lambda_ot:
            Balanced parameter for OT. Default: 1.0
        reg:
            Entropy regularization parameter in OT. Default: 0.1
        reg_m:
            Unbalanced OT parameter. Larger values means more balanced OT. Default: 1.0
        lr
            Learning rate. Default: 2e-4
        max_iteration
            Max iterations for training. Training one batch_size samples is one iteration. Default: 60000
        early_stopping
            EarlyStopping class (definite in utils.py) for stoping the training if loss doesn't improve after a given patience. Default: None
        device
            'cuda' or 'cpu' for training. Default: 'cuda'
        verbose
            Verbosity, True or False. Default: False
        drug_response
            if True, use drug_response decoder to predict drug response label. Default: True
        """

        self.to(device)

        optim = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=5e-4)


        if loss_type == 'BCE':
            loss_func = nn.BCELoss()
        elif loss_type == 'MSE':
            loss_func = nn.MSELoss()
        elif loss_type == 'L1':
            loss_func = nn.L1Loss()
        print(f'####vae.py#839row，n_epoch={n_epoch}')
        with tqdm(range(n_epoch), total=n_epoch, desc='Epochs') as tq: # 用于在循环或迭代过程中显示进度条。它可以帮助你直观地了解代码的执行进度。
            for epoch in tq:
                tk0 = tqdm(enumerate(zip(Ctrainloader,cycle(Ptrainloader))),total=len(Ctrainloader), leave=False, desc='Iterations', disable=(not verbose))
                epoch_loss = defaultdict(float)
                if mode == 'h':
                    for i, data in tk0:#①y表示领域的id，0表示该条数据来自第一个数据集bulk，1表示该条数据来自第二个数据集sc,y=tensor([0, 1, 0, 1...])。   ②idx表示该条数据的索引号idx=tensor([ 9371,  6514,  9710]).
                        # print(f'####vae.py#1094row,epoch==={epoch}')
                        # 1. batch里 的bulk数据
                        x_bulk = data[0][0]
                        response_bulk = data[0][1].float()
                        y_bulk = torch.tensor([0] * data[0][0].shape[0]) # bulk数据的domain_id设置为0
                        idx_bulk = data[0][2]
                        # 2. batch里 的sc数据
                        x_sc = data[1][0]
                        response_sc = data[1][1].float()
                        y_sc = torch.tensor([1] * data[1][0].shape[0]) # sc数据的domain_id设置为1
                        idx_sc = data[1][2] + num_cell[0]
                        # 使用 torch.cat() 函数在第一个维度上进行连接
                        # x = torch.cat((x_bulk, x_sc), dim=0) #TODO，这里将x_bulk和x_sc数据拼在了一起
                        y = torch.cat((y_bulk, y_sc), dim=0) #TODO，这里将源域的domain_id和目标域的domain_id拼在了一起
                        idx = torch.cat((idx_bulk, idx_sc), dim=0) #TODO，这里将源域的idx和目标域的idx合在了一起
                        response = torch.cat((response_bulk, response_sc), dim=0) #TODO，这里将源域的response_label和目标域的拼在了一起

                        idx_origin = idx
                        # x_c, y = x.float().to(device), y.long().to(device)
                        y = y.long().to(device)
                        # x_c, y = x[:,:].float().to(device), y.long().to(device) #TODO 使用数据集x（bulk细胞系）所有的基因
                        x_bulk = x_bulk.float().to(device) #TODO，把x_bulk放进设备上
                        x_sc = x_sc.float().to(device) #TODO，把x_sc放进设备上
                        idx = idx.to(device)

                        loc_ref = torch.where(y==self.ref_id)[0] #用于根据给定条件选择元素
                        #TODOq num_cell[0:self.ref_id]是什么, num_cell[0:self.ref_id]===[]，num_cell=[11259, 11259]，loc_ref=tensor([    0,     1,     2,  ..., 22510, 22512, 22514]
                        # print(f'###vae.py#423rows############num_cell[0:self.ref_id]==={num_cell[0:self.ref_id]}，num_cell={num_cell}，loc_ref.shape={loc_ref.shape}')
                        idx_ref = idx[loc_ref] - sum(num_cell[0:self.ref_id])#TODOq，这个代码是什么意思？loc_ref是索引,loc_ref=tensor([  0,   1,   2,   3...])
                        # print(f'###vae.py#425rows####################idx_ref.shape={idx_ref.shape}')#idx_ref是找到self.ref_id的索引号，idx_ref=tensor([ 9371,  6514,  9710,...])
                        loc_query = {}
                        idx_query = {}
                        tran_batch = {}
                        Prior_batch = None

                        query_id = list(range(self.n_domain))
                        query_id.remove(self.ref_id)

                        if len(loc_ref) > 0:
                            for j in query_id:

                                loc_query[j] = torch.where(y==j)[0]
                                # print(f'####vae.py#440rows############j={j},loc_query[j]={loc_query[j]}')
                                idx_query[j] = idx[loc_query[j]] - sum(num_cell[0:j])

                                if save_OT:
                                    if len(idx_query[j]) != 0:
                                        if (len(idx_query[j])) == 1:
                                            tran_batch[j] = torch.from_numpy(tran[j]).to(device)[idx_query[j]][idx_ref]
                                        else:
                                            tran_batch[j] = torch.from_numpy(tran[j]).to(device)[idx_query[j]][:,idx_ref]
                                else:
                                    tran_batch[j] = None
                                if Prior is not None:
                                    Prior_batch = Prior[j][idx_query[j]][:,idx_ref].to(device)

                        ot_loss = torch.tensor(0.0).to(device)
                        recon_loss = torch.tensor(0.0).to(device)
                        recon_loss_1 = torch.tensor(0.0).to(device)
                        recon_loss_2 = torch.tensor(0.0).to(device)
                        kl_loss = torch.tensor(0.0).to(device)
                        drug_response_loss = torch.tensor(0.0).to(device)
                        cell_regularization_loss = torch.tensor(0.0).to(device)

                        loc = loc_query
                        loc[self.ref_id] = loc_ref
                        idx = idx_query
                        idx[self.ref_id] = idx_ref #x_c.shape=torch.Size([256, 2000])
                        #TODO，@see：vae.py, 148rows，使用共享编码器
                        # z, mu, var = self.encoder(x_c, 0) #z.shape=torch.Size([256, 16]),mu.shape=torch.Size([256, 16]),var.shape=torch.Size([256, 16])
                        z_bulk, mu_bulk, var_bulk = self.encoder(x_bulk, 3) #z.shape=torch.Size([256, 16]),mu.shape=torch.Size([256, 16]),var.shape=torch.Size([256, 16])
                        z_sc, mu_sc, var_sc = self.encoder(x_sc, 4) #z.shape=torch.Size([256, 16]),mu.shape=torch.Size([256, 16]),var.shape=torch.Size([256, 16])
                        # TODO，把bulk和sc的隐空间数据拼到一起
                        z = torch.cat((z_bulk, z_sc), dim=0)
                        mu = torch.cat((mu_bulk, mu_sc), dim=0)
                        var = torch.cat((var_bulk, var_sc), dim=0)

                        if cell_regularization:# use cell regularization
                            if z_sc.shape[0]<10: # target data（single cell dataset）
                                next
                            else:
                                edgeList = calculateKNNgraphDistanceMatrix(z_sc.cpu().detach().numpy(), distanceType='euclidean', k=10)
                                listResult, size = generateLouvainCluster(edgeList)
                                # sc sim loss
                                loss_s = 0
                                for i in range(size):
                                    #print(i)
                                    s = cosine_similarity(x_sc[np.asarray(listResult) == i,:].cpu().detach().numpy())
                                    s = 1-s
                                    loss_s += np.sum(np.triu(s,1))/((s.shape[0]*s.shape[0])*2-s.shape[0])
                                cell_regularization_loss += loss_s

                        if label_weight is None: # default, label_weight is None
                            #TODO, 使用两个解码器，不使用共享解码器
                            recon_x_bulk = self.decoder(z[loc[0]], 4, None)
                            recon_x_sc = self.decoder(z[loc[1]], 5, None)
                            recon_loss_1 = loss_func(recon_x_bulk, x_bulk.float().to(device))
                            recon_loss_2 = loss_func(recon_x_sc, x_sc.float().to(device))
                            recon_loss = recon_loss_1 + recon_loss_2

                        kl_loss = kl_div(mu, var)
                        
                        if drug_response:#TODO drug_response_decoder的bulk模型训练
                            for j in range(self.n_domain):
                                if len(loc[j])>0 and j==0: # 表示取数据集x的隐空间嵌入z（取bulk细胞系的隐空间嵌入z ）
                                    if adata_cm is not None:
                                        groundtruth_bulk_label = response_bulk.reshape(-1,1)
                                        groundtruth_bulk_label = torch.Tensor(groundtruth_bulk_label).to(device)
                                        # print(f'####vae.py#509rows##################groundtruth_bulk_label.shape={groundtruth_bulk_label.shape},groundtruth_bulk_label[:5]={groundtruth_bulk_label[:5]}')
                                        predicted_bulk_label = self.decoder(z[loc[j]], j+2+1) # decoder(,0)表示所有共同数据的共同的解码器；decoder(,1)表示数据集X特异性高可变基因的解码器；decoder(,2)表示数据集Y特异性高可变基因的解码器；decoder(,3)表示所有共同数据里的数据X的药物响应解码器；
                                        # print(f'####vae.py#511rows##################predicted_bulk_label.shape={predicted_bulk_label.shape},predicted_bulk_label[:5]={predicted_bulk_label[:5]}')
                                        drug_response_loss = loss_func(predicted_bulk_label, groundtruth_bulk_label)#计算bulk细胞系药物响应真实标签和预测标签之间的损失函数

                        if len(torch.unique(y))>1 and len(loc[self.ref_id])!=0:

                            mu_dict = {}
                            var_dict = {}

                            for j in range(self.n_domain):
                                if len(loc[j])>0:
                                    mu_dict[j] = mu[loc[j]]
                                    var_dict[j] = var[loc[j]]

                            for j in query_id:
                                if len(loc[j])>0:

                                    if label_weight is None:

                                        ot_loss_tmp, tran_batch[j] = unbalanced_ot(
                                            tran_batch[j],
                                            mu_dict[j],
                                            var_dict[j],
                                            mu_dict[self.ref_id].detach(),
                                            var_dict[self.ref_id].detach(),
                                            reg=reg,
                                            reg_m=reg_m,
                                            idx_q=idx_query[j],# default query_weight is None,这个参数没用到
                                            idx_r=idx_ref, # default ref_weight is None, 这个参数函数也没用到
                                            Couple=Prior_batch, # default Prior is None, @see:881row, 所以Prior_batch也是None，这个参数函数没用到
                                            device=device,
                                        )

                                    else:

                                        ot_loss_tmp, tran_batch[j] = unbalanced_ot(
                                            tran_batch[j],
                                            mu_dict[j],
                                            var_dict[j],
                                            mu_dict[self.ref_id].detach(),
                                            var_dict[self.ref_id].detach(),
                                            reg=reg,
                                            reg_m=reg_m,
                                            idx_q=idx_query[j],
                                            idx_r=idx_ref,
                                            Couple=Prior_batch,
                                            device=device,
                                            query_weight=label_weight[j],
                                            ref_weight=label_weight[self.ref_id],
                                        )

                                    if save_OT:
                                        t0 = np.repeat(idx_query[j].cpu().numpy(), len(idx_ref)).reshape(len(idx_query[j]),len(idx_ref))
                                        t1 = np.tile(idx_ref.cpu().numpy(), (len(idx_query[j]), 1))
                                        tran[j][t0,t1] = tran_batch[j].cpu().numpy()

                                    ot_loss += ot_loss_tmp

                        if cell_regularization:
                            loss = {'recloss':lambda_recon*recon_loss, 'klloss':lambda_kl*kl_loss, 'otloss':lambda_ot*ot_loss, 'drug_R_loss':lambda_response*drug_response_loss, 'cell_R_loss':lambda_cell*cell_regularization_loss} #
                        else:
                            loss = {'recloss':lambda_recon*recon_loss, 'klloss':lambda_kl*kl_loss, 'otloss':lambda_ot*ot_loss, 'drug_response_loss':lambda_response*drug_response_loss} #

                        optim.zero_grad()
                        sum(loss.values()).backward()#TODO，疑问，可以这样子后向传播吗？
                        optim.step()

                        for k,v in loss.items():
                            epoch_loss[k] += loss[k].item() #TODO 疑问，loss[k].item()是什么, k=recloss,v=2528.068359375,loss.items()=dict_items([('recloss', tensor(...)), ('klloss', tensor(...)), ('otloss', tensor(...))])
                            # print(f'#######vae.py#555rows#############(k,v) in loss.items()====k={k},v={v}')

                        info = ','.join(['{}={:.2f}'.format(k, v) for k,v in loss.items()])
                        tk0.set_postfix_str(info)
                    
                    epoch_loss = {k:v/(i+1) for k, v in epoch_loss.items()}

                    epoch_info = ','.join(['{}={:.2f}'.format(k, v) for k,v in epoch_loss.items()])
                    tq.set_postfix_str(epoch_info) 
                torch.cuda.empty_cache()           



    def fit2_1( # 使用早停机制的时候（epoch固定），且未对源域bulk数据平衡采样，使用这个fit2_1()做训练
            self, 
            dataloader, 
            tran,
            num_cell,
            num_gene,
            mode='h',
            loss_type='BCE',
            label_weight=None,
            Prior=None,
            save_OT=False,
            use_specific=True,
            lambda_s=0.5,
            lambda_kl=0.5,
            lambda_recon=1.0,
            lambda_ot=1.0,
            lambda_response=1.0, #TODO,药物响应预测模型的权重参数
            lambda_cell=1.0, #TODO,单细胞正则化损失函数的权重参数
            reg=0.1,
            reg_m=1.0,
            lr=2e-4,
            device='cuda:0',  
            verbose=False,
            drug_response=True, ###TODO,new feature
            cell_regularization=False, ###TODO,new feature
            adata_cm=None, ###TODO,new feature
            n_epoch=1000,
            global_match=False, ###TODO,new feature
            mmd_GAMMA=1000, #Gamma parameter in the kernel of the MMD loss of the transfer learning, default: 1000
            lambda_mmd=1.0, ###TODO,new feature
        ):
        """
        train VAE

        Parameters
        ----------
        dataloader
            An iterable over the given dataset for training.
        tran
            A global OT plan. tran={} if save_OT=False in function.py.
        num_cell
            List of number of cells in different datasets. 
        num_gene
            List of number of genes in different datasets. 
        mode
            Choose from ['h', 'v', 'd']
            If 'h', integrate data with common genes
            If 'v', integrate data profiled from the same cells
            If 'd', inetrgate data without common genes
            Default: 'h'
        loss_type
            type of loss. Choose between ['BCE', 'MSE', 'L1']. Default: 'BCE'
        label_weight
            Prior-guided weighted vectors. Default: None
        Prior
            Prior correspondence matrix.
        save_OT
            If True, output a global OT plan. Default: False
        use_specific
            If True, specific genes in each dataset will be considered. Default: True
        lambda_s
            Balanced parameter for specific genes. Default: 0.5
        lambda_kl: 
            Balanced parameter for KL divergence. Default: 0.5
        lambda_recon:
            Balanced parameter for reconstruction. Default: 1.0
        lambda_ot:
            Balanced parameter for OT. Default: 1.0
        reg:
            Entropy regularization parameter in OT. Default: 0.1
        reg_m:
            Unbalanced OT parameter. Larger values means more balanced OT. Default: 1.0
        lr
            Learning rate. Default: 2e-4
        max_iteration
            Max iterations for training. Training one batch_size samples is one iteration. Default: 60000
        early_stopping
            EarlyStopping class (definite in utils.py) for stoping the training if loss doesn't improve after a given patience. Default: None
        device
            'cuda' or 'cpu' for training. Default: 'cuda'
        verbose
            Verbosity, True or False. Default: False
        drug_response
            if True, use drug_response decoder to predict drug response label. Default: True
        """

        self.to(device)
        # Set distribution loss
        def mmd_loss_func(x,y,GAMMA=mmd_GAMMA):
            result = mmd.mmd_loss(x,y,GAMMA)
            return result
        optim = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=5e-4)


        if loss_type == 'BCE':
            loss_func = nn.BCELoss()
        elif loss_type == 'MSE':
            loss_func = nn.MSELoss()
        elif loss_type == 'L1':
            loss_func = nn.L1Loss()
        print(f'####vae.py#839row，n_epoch={n_epoch}')
        with tqdm(range(n_epoch), total=n_epoch, desc='Epochs') as tq: # 用于在循环或迭代过程中显示进度条。它可以帮助你直观地了解代码的执行进度。
            for epoch in tq:
                #TODO，dataloader.dataset是什么？，dataloader.dataset.shape=(22518, 4000)
                #print(f'####vae.py#279行####################dataloader.dataset.shape={dataloader.dataset.shape}')
                tk0 = tqdm(enumerate(dataloader), total=len(dataloader), leave=False, desc='Iterations', disable=(not verbose))
                epoch_loss = defaultdict(float)

                if mode == 'h':

                    for i, (x, y, idx) in tk0:#①y表示领域的id，0表示该条数据来自第一个数据集，1表示该条数据来自第二个数据集y=tensor([0, 1, 0, 1...])。   ②idx表示该条数据的索引号idx=tensor([ 9371,  6514,  9710]).
                        idx_origin = idx
                        #TODO 对x，y，idx的值有疑问,@see:data_loader.py, 143rows, x.shape=torch.Size([22517, 4000]),y.shape=torch.Size([22517]),idx.shape=torch.Size([22517])
                        # (x, y, idx) in tk0:,  @see:data_loader.py,143rows.
                        #print(f'###vae.py#415rows#################(x.shape, y.shape, idx.shape)===x.shape={x.shape},y.shape={y.shape},idx.shape={idx.shape}')
                        #print(f'###vae.py#416行################self.n_domain==={self.n_domain}') #self.n_domain===2
                        #print(f'###vae.py#427行################num_gene==={num_gene}') #num_gene===[2000, 2000, 2000]
                        x_c, y = x[:, 0:num_gene[self.n_domain]].float().to(device), y.long().to(device)  
                        idx = idx.to(device)
                                                    
                        loc_ref = torch.where(y==self.ref_id)[0] #用于根据给定条件选择元素
                        #TODO num_cell[0:self.ref_id]是什么, num_cell[0:self.ref_id]===[]，num_cell=[11259, 11259]，loc_ref=tensor([    0,     1,     2,  ..., 22510, 22512, 22514]
                        #print(f'###vae.py#423rows############num_cell[0:self.ref_id]==={num_cell[0:self.ref_id]}，num_cell={num_cell}，loc_ref.shape={loc_ref.shape}')
                        idx_ref = idx[loc_ref] - sum(num_cell[0:self.ref_id])#TODO，这个代码是什么意思？loc_ref是索引,loc_ref=tensor([  0,   1,   2,   3...])
                        #print(f'###vae.py#425rows####################idx_ref.shape={idx_ref.shape}')#idx_ref是找到self.ref_id的索引号，idx_ref=tensor([ 9371,  6514,  9710,...])
                        loc_query = {}
                        idx_query = {}
                        tran_batch = {}
                        Prior_batch = None

                        query_id = list(range(self.n_domain))
                        query_id.remove(self.ref_id)

                        if len(loc_ref) > 0:
                            for j in query_id:

                                loc_query[j] = torch.where(y==j)[0]
                                #print(f'####vae.py#440rows############j={j},loc_query[j]={loc_query[j]}')
                                idx_query[j] = idx[loc_query[j]] - sum(num_cell[0:j])

                                if save_OT:
                                    if len(idx_query[j]) != 0:
                                        if (len(idx_query[j])) == 1:
                                            tran_batch[j] = torch.from_numpy(tran[j]).to(device)[idx_query[j]][idx_ref]
                                        else:
                                            tran_batch[j] = torch.from_numpy(tran[j]).to(device)[idx_query[j]][:,idx_ref]
                                else:
                                    tran_batch[j] = None

                                if Prior is not None:
                                    Prior_batch = Prior[j][idx_query[j]][:,idx_ref].to(device)

                        ot_loss = torch.tensor(0.0).to(device)
                        recon_loss = torch.tensor(0.0).to(device)
                        kl_loss = torch.tensor(0.0).to(device)
                        drug_response_loss = torch.tensor(0.0).to(device)
                        cell_regularization_loss = torch.tensor(0.0).to(device)

                        loc = loc_query
                        loc[self.ref_id] = loc_ref
                        idx = idx_query
                        idx[self.ref_id] = idx_ref #x_c.shape=torch.Size([256, 2000])
                        #TODO，@see：vae.py, 148rows
                        z, mu, var = self.encoder(x_c, 0) #z.shape=torch.Size([256, 16]),mu.shape=torch.Size([256, 16]),var.shape=torch.Size([256, 16])
                        #TODO,疑问，self.encoder(x_c, 0)==x_c.shape=torch.Size([22517, 2000])，z.shape=torch.Size([22517, 16])，mu.shape=torch.Size([22517, 16])，var.shape=torch.Size([22517, 16])
                        #print(f'###vae.y#474rows##################self.encoder(x_c, 0)==x_c.shape={x_c.shape},z.shape={z.shape},mu.shape={mu.shape},var.shape={var.shape}')
                        recon_x_c = self.decoder(z, 0, y)        
                        # 计算全局匹配损失mmd
                        if global_match:
                            mmd_loss = mmd_loss_func(z[loc[0]], z[loc[1]])
                        # 计算细胞正则化损失
                        if cell_regularization:# use cell regularization
                            if z[loc[1]].shape[0]<10: # target data（single cell dataset）
                                next
                            else:
                                edgeList = calculateKNNgraphDistanceMatrix(z[loc[1]].cpu().detach().numpy(), distanceType='euclidean', k=10)
                                listResult, size = generateLouvainCluster(edgeList)
                                # sc sim loss
                                loss_s = 0
                                for i in range(size):
                                    #print(i)
                                    s = cosine_similarity(x_c[loc[1]][np.asarray(listResult) == i,:].cpu().detach().numpy())
                                    s = 1-s
                                    loss_s += np.sum(np.triu(s,1))/((s.shape[0]*s.shape[0])*2-s.shape[0])
                                cell_regularization_loss += loss_s
                        
                        if label_weight is None: # default, label_weight is None
                            #recon_loss = loss_func(recon_x_c, x_c) * 2000 # default, loss_func = nn.BCELoss()#TODO，这里为什么要乘2000，为什么BCEloss损失为负数
                            #recon_loss = loss_func(recon_x_c, x_c) * num_gene[self.n_domain]
                            recon_loss = loss_func(recon_x_c, x_c) 
                            #recon_loss = abs(loss_func(recon_x_c, x_c)) * 2000 # default, loss_func = nn.BCELoss()#TODO，这里为什么要乘2000，为什么BCEloss损失为负数
                        else:
                            for j, weight in enumerate(label_weight):

                                if len(loc[j])>0:
                                    if weight is None:
                                        recon_loss += 1/self.n_domain * loss_func(recon_x_c[loc[j]], x_c[loc[j]]) * \
                                        2000
                                        # kl_loss += kl_div(mu[loc[j]], var[loc[j]])                               
                                    else:
                                        weight = weight.to(device)
                                        recon_loss += 1/self.n_domain * F.binary_cross_entropy(recon_x_c[loc[j]], x_c[loc[j]], weight=weight[idx[j]]) * \
                                        2000
                                        # kl_loss += kl_div(mu[loc[j]], var[loc[j]], weight[idx[j]])
                                
                        kl_loss = kl_div(mu, var) 
                        if use_specific:

                            x_s = x[:, num_gene[self.n_domain]:].float().to(device) #x.shape=torch.Size([22517, 4000])

                            for j in range(self.n_domain):
                                if len(loc[j])>0:
                                    #TODO，loc[j]是什么？,loc[j]应该是索引loc[j]=tensor([  0,   1,   2,   3,...]，loc[j].shape=torch.Size([11258])， z[loc[j]].shape=torch.Size([11258, 16])，z.shape=torch.Size([22517, 16])
                                    #print(f'###vae.py#489rows###################loc[j].shape={loc[j].shape}，z[loc[j]].shape={z[loc[j]].shape}，z.shape={z.shape}，loc[j]={loc[j]}，loc={loc}')
                                    recon_x_s = self.decoder(z[loc[j]], j+1)
                                    recon_loss += lambda_s * loss_func(recon_x_s, x_s[loc[j]][:, 0:num_gene[j]]) * 2000 #计算特异性高表达基因的重建数据和原数据之间的损失函数

                        if drug_response:#TODO drug_response_decoder的bulk模型训练
                            for j in range(self.n_domain):
                                if len(loc[j])>0 and j==0: # 表示取数据集x的隐空间嵌入z（取bulk细胞系的隐空间嵌入z ）
                                    if adata_cm is not None:
                                        a = idx_origin[y==j].tolist() #定位到数据集x（bulk细胞系）的索引，用来拿到其相应的response label
                                        groundtruth_bulk_label = adata_cm.obs['response'].iloc[a,].values.reshape(-1,1)
                                        groundtruth_bulk_label = torch.Tensor(groundtruth_bulk_label).to(device)
                                        #print(f'####vae.py#509rows##################groundtruth_bulk_label.shape={groundtruth_bulk_label.shape},groundtruth_bulk_label[:5]={groundtruth_bulk_label[:5]}')
                                        predicted_bulk_label = self.decoder(z[loc[j]], j+2+1) # decoder(,0)表示所有共同数据的共同的解码器；decoder(,1)表示数据集X特异性高可变基因的解码器；decoder(,2)表示数据集Y特异性高可变基因的解码器；decoder(,3)表示所有共同数据里的数据X的药物响应解码器；
                                        #print(f'####vae.py#511rows##################predicted_bulk_label.shape={predicted_bulk_label.shape},predicted_bulk_label[:5]={predicted_bulk_label[:5]}')
                                        drug_response_loss = drug_response_loss = loss_func(predicted_bulk_label, groundtruth_bulk_label)#计算bulk细胞系药物响应真实标签和预测标签之间的损失函数
                        
                        if len(torch.unique(y))>1 and len(loc[self.ref_id])!=0:
                            
                            mu_dict = {}
                            var_dict = {}

                            for j in range(self.n_domain):
                                if len(loc[j])>0:
                                    mu_dict[j] = mu[loc[j]]
                                    var_dict[j] = var[loc[j]]

                            for j in query_id:
                                if len(loc[j])>0:

                                    if label_weight is None:

                                        ot_loss_tmp, tran_batch[j] = unbalanced_ot(
                                            tran_batch[j],
                                            mu_dict[j], 
                                            var_dict[j], 
                                            mu_dict[self.ref_id].detach(), 
                                            var_dict[self.ref_id].detach(), 
                                            reg=reg,
                                            reg_m=reg_m,
                                            idx_q=idx_query[j],
                                            idx_r=idx_ref,
                                            Couple=Prior_batch, 
                                            device=device,
                                        )

                                    else:

                                        ot_loss_tmp, tran_batch[j] = unbalanced_ot(
                                            tran_batch[j],
                                            mu_dict[j], 
                                            var_dict[j], 
                                            mu_dict[self.ref_id].detach(), 
                                            var_dict[self.ref_id].detach(), 
                                            reg=reg,
                                            reg_m=reg_m,
                                            idx_q=idx_query[j],
                                            idx_r=idx_ref,
                                            Couple=Prior_batch, 
                                            device=device,
                                            query_weight=label_weight[j],
                                            ref_weight=label_weight[self.ref_id],
                                        )

                                    if save_OT:
                                        t0 = np.repeat(idx_query[j].cpu().numpy(), len(idx_ref)).reshape(len(idx_query[j]),len(idx_ref))
                                        t1 = np.tile(idx_ref.cpu().numpy(), (len(idx_query[j]), 1))
                                        tran[j][t0,t1] = tran_batch[j].cpu().numpy()

                                    ot_loss += ot_loss_tmp
   
                        if cell_regularization:
                            loss = {'recloss':lambda_recon*recon_loss, 'klloss':lambda_kl*kl_loss, 'otloss':lambda_ot*ot_loss, 'drug_R_loss':lambda_response*drug_response_loss, 'cell_R_loss':lambda_cell*cell_regularization_loss} #
                        else:
                            loss = {'recloss':lambda_recon*recon_loss, 'klloss':lambda_kl*kl_loss, 'otloss':lambda_ot*ot_loss, 'drug_response_loss':lambda_response*drug_response_loss} #
                        
                        if global_match:
                            loss['mmd_loss'] = lambda_mmd*mmd_loss
                        
                        optim.zero_grad()
                        sum(loss.values()).backward()#TODO，疑问，可以这样子后向传播吗？
                        optim.step()

                        for k,v in loss.items():
                            epoch_loss[k] += loss[k].item() #TODO 疑问，loss[k].item()是什么, k=recloss,v=2528.068359375,loss.items()=dict_items([('recloss', tensor(...)), ('klloss', tensor(...)), ('otloss', tensor(...))])
                            #print(f'#######vae.py#555rows#############(k,v) in loss.items()====k={k},v={v}')

                        info = ','.join(['{}={:.2f}'.format(k, v) for k,v in loss.items()])
                        tk0.set_postfix_str(info)
                                        
                    epoch_loss = {k:v/(i+1) for k, v in epoch_loss.items()}

                    epoch_info = ','.join(['{}={:.2f}'.format(k, v) for k,v in epoch_loss.items()])
                    tq.set_postfix_str(epoch_info) 
                torch.cuda.empty_cache()           

    def fit2_1_unshared_decoder( # 使用EarlyStopping早停机制的时候（epoch不固定，由训练过程中损失函数不再下降为止），且未对源域bulk数据平衡采样，且不使用共享解码器
            self, 
            dataloader, 
            tran,
            num_cell,
            num_gene,
            mode='h',
            loss_type='BCE',
            label_weight=None,
            Prior=None,
            save_OT=False,
            use_specific=True,
            lambda_s=0.5,
            lambda_kl=0.5,
            lambda_recon=1.0,
            lambda_ot=1.0,
            lambda_response=1.0, #TODO,药物响应预测模型的权重参数
            lambda_cell=1.0, #TODO,单细胞正则化损失函数的权重参数
            reg=0.1,
            reg_m=1.0,
            lr=2e-4,
            device='cuda:0',  
            verbose=False,
            drug_response=True, ###TODO,new feature
            cell_regularization=False, ###TODO,new feature
            adata_cm=None, ###TODO,new feature
            n_epoch=1000,
            global_match=False, ###TODO,new feature
            mmd_GAMMA=1000.0, #Gamma parameter in the kernel of the MMD loss of the transfer learning, default: 1000
            lambda_mmd=1.0, ###TODO,new feature
            
        ):
        """
        train VAE

        Parameters
        ----------
        dataloader
            An iterable over the given dataset for training.
        tran
            A global OT plan. tran={} if save_OT=False in function.py.
        num_cell
            List of number of cells in different datasets. 
        num_gene
            List of number of genes in different datasets. 
        mode
            Choose from ['h', 'v', 'd']
            If 'h', integrate data with common genes
            If 'v', integrate data profiled from the same cells
            If 'd', inetrgate data without common genes
            Default: 'h'
        loss_type
            type of loss. Choose between ['BCE', 'MSE', 'L1']. Default: 'BCE'
        label_weight
            Prior-guided weighted vectors. Default: None
        Prior
            Prior correspondence matrix.
        save_OT
            If True, output a global OT plan. Default: False
        use_specific
            If True, specific genes in each dataset will be considered. Default: True
        lambda_s
            Balanced parameter for specific genes. Default: 0.5
        lambda_kl: 
            Balanced parameter for KL divergence. Default: 0.5
        lambda_recon:
            Balanced parameter for reconstruction. Default: 1.0
        lambda_ot:
            Balanced parameter for OT. Default: 1.0
        reg:
            Entropy regularization parameter in OT. Default: 0.1
        reg_m:
            Unbalanced OT parameter. Larger values means more balanced OT. Default: 1.0
        lr
            Learning rate. Default: 2e-4
        max_iteration
            Max iterations for training. Training one batch_size samples is one iteration. Default: 60000
        early_stopping
            EarlyStopping class (definite in utils.py) for stoping the training if loss doesn't improve after a given patience. Default: None
        device
            'cuda' or 'cpu' for training. Default: 'cuda'
        verbose
            Verbosity, True or False. Default: False
        drug_response
            if True, use drug_response decoder to predict drug response label. Default: True
        """

        self.to(device)
        # Set distribution loss
        def mmd_loss_func(x,y,GAMMA=mmd_GAMMA):
            result = mmd.mmd_loss(x,y,GAMMA)
            return result
        optim = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=5e-4)

        if loss_type == 'BCE':
            loss_func = nn.BCELoss()
        elif loss_type == 'MSE':
            loss_func = nn.MSELoss()
        elif loss_type == 'L1':
            loss_func = nn.L1Loss()
        print(f'####vae.py#839row，n_epoch={n_epoch}')
        with tqdm(range(n_epoch), total=n_epoch, desc='Epochs') as tq: # 用于在循环或迭代过程中显示进度条。它可以帮助你直观地了解代码的执行进度。
            for epoch in tq:
                #TODOq，dataloader.dataset是什么？，dataloader.dataset.shape=(22518, 4000)
                #print(f'####vae.py#279行####################dataloader.dataset.shape={dataloader.dataset.shape}')
                tk0 = tqdm(enumerate(dataloader), total=len(dataloader), leave=False, desc='Iterations', disable=(not verbose))
                epoch_loss = defaultdict(float)

                if mode == 'h':

                    for i, (x, y, idx) in tk0:#①y表示领域的id，0表示该条数据来自第一个数据集，1表示该条数据来自第二个数据集y=tensor([0, 1, 0, 1...])。   ②idx表示该条数据的索引号idx=tensor([ 9371,  6514,  9710]).
                        idx_origin = idx
                        #TODOq 对x，y，idx的值有疑问,@see:data_loader.py, 143rows, x.shape=torch.Size([22517, 4000]),y.shape=torch.Size([22517]),idx.shape=torch.Size([22517])
                        # (x, y, idx) in tk0:,  @see:data_loader.py,143rows.
                        #print(f'###vae.py#415rows#################(x.shape, y.shape, idx.shape)===x.shape={x.shape},y.shape={y.shape},idx.shape={idx.shape}')
                        #print(f'###vae.py#416行################self.n_domain==={self.n_domain}') #self.n_domain===2
                        #print(f'###vae.py#427行################num_gene==={num_gene}') #num_gene===[2000, 2000, 2000]
                        x_c, y = x[:, 0:num_gene[self.n_domain]].float().to(device), y.long().to(device)  
                        idx = idx.to(device)
                                                    
                        loc_ref = torch.where(y==self.ref_id)[0] #用于根据给定条件选择元素
                        #TODOq num_cell[0:self.ref_id]是什么, num_cell[0:self.ref_id]===[]，num_cell=[11259, 11259]，loc_ref=tensor([    0,     1,     2,  ..., 22510, 22512, 22514]
                        #print(f'###vae.py#423rows############num_cell[0:self.ref_id]==={num_cell[0:self.ref_id]}，num_cell={num_cell}，loc_ref.shape={loc_ref.shape}')
                        idx_ref = idx[loc_ref] - sum(num_cell[0:self.ref_id])#TODOq，这个代码是什么意思？loc_ref是索引,loc_ref=tensor([  0,   1,   2,   3...])
                        #print(f'###vae.py#425rows####################idx_ref.shape={idx_ref.shape}')#idx_ref是找到self.ref_id的索引号，idx_ref=tensor([ 9371,  6514,  9710,...])
                        loc_query = {}
                        idx_query = {}
                        tran_batch = {}
                        Prior_batch = None

                        query_id = list(range(self.n_domain))
                        query_id.remove(self.ref_id)

                        if len(loc_ref) > 0:
                            for j in query_id:

                                loc_query[j] = torch.where(y==j)[0]
                                #print(f'####vae.py#440rows############j={j},loc_query[j]={loc_query[j]}')
                                idx_query[j] = idx[loc_query[j]] - sum(num_cell[0:j])

                                if save_OT:
                                    if len(idx_query[j]) != 0:
                                        if (len(idx_query[j])) == 1:
                                            tran_batch[j] = torch.from_numpy(tran[j]).to(device)[idx_query[j]][idx_ref]
                                        else:
                                            tran_batch[j] = torch.from_numpy(tran[j]).to(device)[idx_query[j]][:,idx_ref]
                                else:
                                    tran_batch[j] = None

                                if Prior is not None:
                                    Prior_batch = Prior[j][idx_query[j]][:,idx_ref].to(device)

                        ot_loss = torch.tensor(0.0).to(device)
                        mmd_loss = torch.tensor(0.0).to(device)
                        recon_loss = torch.tensor(0.0).to(device)
                        recon_loss_1 = torch.tensor(0.0).to(device)
                        recon_loss_2 = torch.tensor(0.0).to(device)
                        kl_loss = torch.tensor(0.0).to(device)
                        drug_response_loss = torch.tensor(0.0).to(device)
                        cell_regularization_loss = torch.tensor(0.0).to(device)

                        loc = loc_query
                        loc[self.ref_id] = loc_ref
                        idx = idx_query
                        idx[self.ref_id] = idx_ref #x_c.shape=torch.Size([256, 2000])
                        #TODOq，@see：vae.py, 148rows
                        z, mu, var = self.encoder(x_c, 0) #z.shape=torch.Size([256, 16]),mu.shape=torch.Size([256, 16]),var.shape=torch.Size([256, 16])
                        #TODOq,疑问，self.encoder(x_c, 0)==x_c.shape=torch.Size([22517, 2000])，z.shape=torch.Size([22517, 16])，mu.shape=torch.Size([22517, 16])，var.shape=torch.Size([22517, 16])
                        #print(f'###vae.y#474rows##################self.encoder(x_c, 0)==x_c.shape={x_c.shape},z.shape={z.shape},mu.shape={mu.shape},var.shape={var.shape}')
                        # 计算全局匹配损失mmd
                        if global_match:
                            mmd_loss = mmd_loss_func(z[loc[0]], z[loc[1]])
                        # 计算细胞正则化损失
                        if cell_regularization:# use cell regularization
                            if z[loc[1]].shape[0]<10: # target data（single cell dataset）
                                next
                            else:
                                edgeList = calculateKNNgraphDistanceMatrix(z[loc[1]].cpu().detach().numpy(), distanceType='euclidean', k=10)
                                listResult, size = generateLouvainCluster(edgeList)
                                # sc sim loss
                                loss_s = 0
                                for i in range(size):
                                    #print(i)
                                    s = cosine_similarity(x_c[loc[1]][np.asarray(listResult) == i,:].cpu().detach().numpy())
                                    s = 1-s
                                    loss_s += np.sum(np.triu(s,1))/((s.shape[0]*s.shape[0])*2-s.shape[0])
                                cell_regularization_loss += loss_s
   
                        if label_weight is None: # default, label_weight is None
                            recon_x_bulk = self.decoder(z[loc[0]], 4, None)
                            recon_x_sc = self.decoder(z[loc[1]], 5, None)
                            recon_loss_1 = loss_func(recon_x_bulk, x_c[loc[0]])
                            recon_loss_2 = loss_func(recon_x_sc, x_c[loc[1]])
                            recon_loss = recon_loss_1 + recon_loss_2

                        kl_loss = kl_div(mu, var) 
                        if use_specific:

                            x_s = x[:, num_gene[self.n_domain]:].float().to(device) #x.shape=torch.Size([22517, 4000])

                            for j in range(self.n_domain):
                                if len(loc[j])>0:
                                    #TODO，loc[j]是什么？,loc[j]应该是索引loc[j]=tensor([  0,   1,   2,   3,...]，loc[j].shape=torch.Size([11258])， z[loc[j]].shape=torch.Size([11258, 16])，z.shape=torch.Size([22517, 16])
                                    #print(f'###vae.py#489rows###################loc[j].shape={loc[j].shape}，z[loc[j]].shape={z[loc[j]].shape}，z.shape={z.shape}，loc[j]={loc[j]}，loc={loc}')
                                    recon_x_s = self.decoder(z[loc[j]], j+1)
                                    recon_loss += lambda_s * loss_func(recon_x_s, x_s[loc[j]][:, 0:num_gene[j]]) * 2000 #计算特异性高表达基因的重建数据和原数据之间的损失函数

                        if drug_response:#TODO drug_response_decoder的bulk模型训练
                            for j in range(self.n_domain):
                                if len(loc[j])>0 and j==0: # 表示取数据集x的隐空间嵌入z（取bulk细胞系的隐空间嵌入z ）
                                    if adata_cm is not None:
                                        a = idx_origin[y==j].tolist() #定位到数据集x（bulk细胞系）的索引，用来拿到其相应的response label
                                        groundtruth_bulk_label = adata_cm.obs['response'].iloc[a,].values.reshape(-1,1)
                                        groundtruth_bulk_label = torch.Tensor(groundtruth_bulk_label).to(device)
                                        #print(f'####vae.py#509rows##################groundtruth_bulk_label.shape={groundtruth_bulk_label.shape},groundtruth_bulk_label[:5]={groundtruth_bulk_label[:5]}')
                                        predicted_bulk_label = self.decoder(z[loc[j]], j+2+1) # decoder(,0)表示所有共同数据的共同的解码器；decoder(,1)表示数据集X特异性高可变基因的解码器；decoder(,2)表示数据集Y特异性高可变基因的解码器；decoder(,3)表示所有共同数据里的数据X的药物响应解码器；
                                        #print(f'####vae.py#511rows##################predicted_bulk_label.shape={predicted_bulk_label.shape},predicted_bulk_label[:5]={predicted_bulk_label[:5]}')
                                        drug_response_loss = drug_response_loss = loss_func(predicted_bulk_label, groundtruth_bulk_label)#计算bulk细胞系药物响应真实标签和预测标签之间的损失函数
                        
                        if len(torch.unique(y))>1 and len(loc[self.ref_id])!=0:
                            
                            mu_dict = {}
                            var_dict = {}

                            for j in range(self.n_domain):
                                if len(loc[j])>0:
                                    mu_dict[j] = mu[loc[j]]
                                    var_dict[j] = var[loc[j]]

                            for j in query_id:
                                if len(loc[j])>0:

                                    if label_weight is None:

                                        ot_loss_tmp, tran_batch[j] = unbalanced_ot(
                                            tran_batch[j],
                                            mu_dict[j], 
                                            var_dict[j], 
                                            mu_dict[self.ref_id].detach(), 
                                            var_dict[self.ref_id].detach(), 
                                            reg=reg,
                                            reg_m=reg_m,
                                            idx_q=idx_query[j],
                                            idx_r=idx_ref,
                                            Couple=Prior_batch, 
                                            device=device,
                                        )

                                    else:

                                        ot_loss_tmp, tran_batch[j] = unbalanced_ot(
                                            tran_batch[j],
                                            mu_dict[j], 
                                            var_dict[j], 
                                            mu_dict[self.ref_id].detach(), 
                                            var_dict[self.ref_id].detach(), 
                                            reg=reg,
                                            reg_m=reg_m,
                                            idx_q=idx_query[j],
                                            idx_r=idx_ref,
                                            Couple=Prior_batch, 
                                            device=device,
                                            query_weight=label_weight[j],
                                            ref_weight=label_weight[self.ref_id],
                                        )

                                    if save_OT:
                                        t0 = np.repeat(idx_query[j].cpu().numpy(), len(idx_ref)).reshape(len(idx_query[j]),len(idx_ref))
                                        t1 = np.tile(idx_ref.cpu().numpy(), (len(idx_query[j]), 1))
                                        tran[j][t0,t1] = tran_batch[j].cpu().numpy()

                                    ot_loss += ot_loss_tmp
   
                        if cell_regularization:
                            loss = {'recloss':lambda_recon*recon_loss, 'klloss':lambda_kl*kl_loss, 'otloss':lambda_ot*ot_loss, 'drug_R_loss':lambda_response*drug_response_loss, 'cell_R_loss':lambda_cell*cell_regularization_loss} #
                        else:
                            loss = {'recloss':lambda_recon*recon_loss, 'klloss':lambda_kl*kl_loss, 'otloss':lambda_ot*ot_loss, 'drug_response_loss':lambda_response*drug_response_loss} #
                        
                        if global_match:
                            loss['mmd_loss'] = lambda_mmd*mmd_loss
                            
                        optim.zero_grad()
                        sum(loss.values()).backward()#TODO，疑问，可以这样子后向传播吗？
                        optim.step()

                        for k,v in loss.items():
                            epoch_loss[k] += loss[k].item() #TODO 疑问，loss[k].item()是什么, k=recloss,v=2528.068359375,loss.items()=dict_items([('recloss', tensor(...)), ('klloss', tensor(...)), ('otloss', tensor(...))])
                            #print(f'#######vae.py#555rows#############(k,v) in loss.items()====k={k},v={v}')

                        info = ','.join(['{}={:.2f}'.format(k, v) for k,v in loss.items()])
                        tk0.set_postfix_str(info)
                                        
                    epoch_loss = {k:v/(i+1) for k, v in epoch_loss.items()}

                    epoch_info = ','.join(['{}={:.2f}'.format(k, v) for k,v in epoch_loss.items()])
                    tq.set_postfix_str(epoch_info) 
                torch.cuda.empty_cache()           

    def fit2_1_unshared_encoder_decoder( # 使用EarlyStopping早停机制的时候（epoch不固定，由训练过程中损失函数不再下降为止），且未对源域bulk数据平衡采样，且不使用共享解码器
            self, 
            dataloader, 
            tran,
            num_cell,
            num_gene,
            mode='h',
            loss_type='BCE',
            label_weight=None,
            Prior=None,
            save_OT=False,
            use_specific=True,
            lambda_s=0.5,
            lambda_kl=0.5,
            lambda_recon=1.0,
            lambda_ot=1.0,
            lambda_response=1.0, #TODO,药物响应预测模型的权重参数
            lambda_cell=1.0, #TODO,单细胞正则化损失函数的权重参数
            reg=0.1,
            reg_m=1.0,
            lr=2e-4,
            device='cuda:0',  
            verbose=False,
            drug_response=True, ###TODO,new feature
            cell_regularization=False, ###TODO,new feature
            adata_cm=None, ###TODO,new feature
            adatas=None, ###TODOq,new feature
            n_epoch=1000
        ):
        """
        train VAE

        Parameters
        ----------
        dataloader
            An iterable over the given dataset for training.
        tran
            A global OT plan. tran={} if save_OT=False in function.py.
        num_cell
            List of number of cells in different datasets. 
        num_gene
            List of number of genes in different datasets. 
        mode
            Choose from ['h', 'v', 'd']
            If 'h', integrate data with common genes
            If 'v', integrate data profiled from the same cells
            If 'd', inetrgate data without common genes
            Default: 'h'
        loss_type
            type of loss. Choose between ['BCE', 'MSE', 'L1']. Default: 'BCE'
        label_weight
            Prior-guided weighted vectors. Default: None
        Prior
            Prior correspondence matrix.
        save_OT
            If True, output a global OT plan. Default: False
        use_specific
            If True, specific genes in each dataset will be considered. Default: True
        lambda_s
            Balanced parameter for specific genes. Default: 0.5
        lambda_kl: 
            Balanced parameter for KL divergence. Default: 0.5
        lambda_recon:
            Balanced parameter for reconstruction. Default: 1.0
        lambda_ot:
            Balanced parameter for OT. Default: 1.0
        reg:
            Entropy regularization parameter in OT. Default: 0.1
        reg_m:
            Unbalanced OT parameter. Larger values means more balanced OT. Default: 1.0
        lr
            Learning rate. Default: 2e-4
        max_iteration
            Max iterations for training. Training one batch_size samples is one iteration. Default: 60000
        early_stopping
            EarlyStopping class (definite in utils.py) for stoping the training if loss doesn't improve after a given patience. Default: None
        device
            'cuda' or 'cpu' for training. Default: 'cuda'
        verbose
            Verbosity, True or False. Default: False
        drug_response
            if True, use drug_response decoder to predict drug response label. Default: True
        """

        self.to(device)
        optim = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=5e-4)
        if loss_type == 'BCE':
            loss_func = nn.BCELoss()
        elif loss_type == 'MSE':
            loss_func = nn.MSELoss()
        elif loss_type == 'L1':
            loss_func = nn.L1Loss()
        # print(f'####vae.py#839row，n_epoch={n_epoch}')
        with tqdm(range(n_epoch), total=n_epoch, desc='Epochs') as tq: # 用于在循环或迭代过程中显示进度条。它可以帮助你直观地了解代码的执行进度。
            for epoch in tq:
                tk0 = tqdm(enumerate(dataloader), total=len(dataloader), leave=False, desc='Iterations', disable=(not verbose))
                epoch_loss = defaultdict(float)
                if mode == 'h':
                    for i, (x, y, idx) in tk0:#①y表示领域的id，0表示该条数据来自第一个数据集，1表示该条数据来自第二个数据集y=tensor([0, 1, 0, 1...])。   ②idx表示该条数据的索引号idx=tensor([ 9371,  6514,  9710]).
                        idx_origin = idx
                        # x_c, y = x[:, 0:num_gene[self.n_domain]].float().to(device), y.long().to(device)  
                        x_c = x.float().to(device)
                        y =  y.long().to(device)
                        idx = idx.to(device)
                        #print(f'####vae.py,#3977rows##x_c.shape={x_c.shape}')
                        # new code
                        loc_bulk = torch.where(y==0)[0]
                        loc_sc = torch.where(y==1)[0]
                        #print(f'####vae.py## 3981rows, num_gene={num_gene}')
                        # 把x拆包分成x_bulk和x_sc数据
                        x_bulk = x_c[loc_bulk, 0:num_gene[0]]
                        #print(f'####vae.py## 3984rows, x_bulk.shape={x_bulk.shape}')
                        x_sc = x_c[loc_sc, 0:num_gene[1]]
                        #print(f'####vae.py## 3986rows, x_sc.shape={x_sc.shape}')
                        # 分别使用两个编码器
                        z_bulk, mu_bulk, var_bulk = self.encoder(x_bulk, 3)
                        z_sc, mu_sc, var_sc = self.encoder(x_sc, 4)
                        # 把分别得到的两种隐空间数据部分，拼到一起
                        z = torch.zeros(size=(z_bulk.shape[0]+z_sc.shape[0], z_bulk.shape[1]), device=torch.device(device))
                        mu = torch.zeros(size=(mu_bulk.shape[0]+mu_sc.shape[0], mu_bulk.shape[1]), device=torch.device(device))
                        var = torch.zeros(size=(var_bulk.shape[0]+var_sc.shape[0], var_bulk.shape[1]), device=torch.device(device))
                        # 使用 loc_bulk 索引将 z_bulk 填充到 z
                        z[loc_bulk] = z_bulk
                        mu[loc_bulk] = mu_bulk
                        var[loc_bulk] = var_bulk
                        # 使用 loc_sc 索引将 z_sc 填充到 z
                        z[loc_sc] = z_sc
                        mu[loc_sc] = mu_sc
                        var[loc_sc] = var_sc
                        
                        loc_ref = torch.where(y==self.ref_id)[0] #用于根据给定条件选择元素
                        idx_ref = idx[loc_ref] - sum(num_cell[0:self.ref_id])#TODOq，这个代码是什么意思？loc_ref是索引,loc_ref=tensor([  0,   1,   2,   3...])
                        loc_query = {}
                        idx_query = {}
                        tran_batch = {}
                        Prior_batch = None

                        query_id = list(range(self.n_domain))
                        query_id.remove(self.ref_id)

                        if len(loc_ref) > 0:
                            for j in query_id:

                                loc_query[j] = torch.where(y==j)[0]
                                #print(f'####vae.py#440rows############j={j},loc_query[j]={loc_query[j]}')
                                idx_query[j] = idx[loc_query[j]] - sum(num_cell[0:j])

                                if save_OT:
                                    if len(idx_query[j]) != 0:
                                        if (len(idx_query[j])) == 1:
                                            tran_batch[j] = torch.from_numpy(tran[j]).to(device)[idx_query[j]][idx_ref]
                                        else:
                                            tran_batch[j] = torch.from_numpy(tran[j]).to(device)[idx_query[j]][:,idx_ref]
                                else:
                                    tran_batch[j] = None

                                if Prior is not None:
                                    Prior_batch = Prior[j][idx_query[j]][:,idx_ref].to(device)

                        ot_loss = torch.tensor(0.0).to(device)
                        recon_loss = torch.tensor(0.0).to(device)
                        recon_loss_1 = torch.tensor(0.0).to(device)
                        recon_loss_2 = torch.tensor(0.0).to(device)
                        kl_loss = torch.tensor(0.0).to(device)
                        drug_response_loss = torch.tensor(0.0).to(device)
                        cell_regularization_loss = torch.tensor(0.0).to(device)

                        loc = loc_query
                        loc[self.ref_id] = loc_ref
                        idx = idx_query
                        idx[self.ref_id] = idx_ref #x_c.shape=torch.Size([256, 2000])
                        #TODOq，@see：vae.py, 148rows
                        # z, mu, var = self.encoder(x_c, 0) #z.shape=torch.Size([256, 16]),mu.shape=torch.Size([256, 16]),var.shape=torch.Size([256, 16])
                        #TODOq,疑问，self.encoder(x_c, 0)==x_c.shape=torch.Size([22517, 2000])，z.shape=torch.Size([22517, 16])，mu.shape=torch.Size([22517, 16])，var.shape=torch.Size([22517, 16])
                        
                        if cell_regularization:# use cell regularization
                            if z_sc.shape[0]<10: # target data（single cell dataset）
                                next
                            else:
                                edgeList = calculateKNNgraphDistanceMatrix(z_sc.cpu().detach().numpy(), distanceType='euclidean', k=10)
                                listResult, size = generateLouvainCluster(edgeList)
                                # sc sim loss
                                loss_s = 0
                                for i in range(size):
                                    #print(i)
                                    s = cosine_similarity(x_sc[np.asarray(listResult) == i,:].cpu().detach().numpy())
                                    s = 1-s
                                    loss_s += np.sum(np.triu(s,1))/((s.shape[0]*s.shape[0])*2-s.shape[0])
                                cell_regularization_loss += loss_s
                        
                        if label_weight is None: # default, label_weight is None
                            recon_x_bulk = self.decoder(z_bulk, 4, None)
                            recon_x_sc = self.decoder(z_sc, 5, None)
                            recon_loss_1 = loss_func(recon_x_bulk, x_bulk)
                            recon_loss_2 = loss_func(recon_x_sc, x_sc)
                            recon_loss = recon_loss_1 + recon_loss_2

                        kl_loss_1 = kl_div(mu_bulk, var_bulk)
                        kl_loss_2 = kl_div(mu_sc, var_sc)
                        kl_loss = kl_loss_1 + kl_loss_2
                        if drug_response:#TODO drug_response_decoder的bulk模型训练
                            for j in range(self.n_domain):
                                if len(loc[j])>0 and j==0: # 表示取数据集x的隐空间嵌入z（取bulk细胞系的隐空间嵌入z ）
                                    if adata_cm is not None:
                                        a = idx_origin[y==0].tolist() #定位到数据集x（bulk细胞系）的索引，用来拿到其相应的response label
                                        groundtruth_bulk_label = adata_cm.obs['response'].iloc[a,].values.reshape(-1,1)
                                        groundtruth_bulk_label = torch.Tensor(groundtruth_bulk_label).to(device)
                                        predicted_bulk_label = self.decoder(z_bulk, 0+2+1) # decoder(,0)表示所有共同数据的共同的解码器；decoder(,1)表示数据集X特异性高可变基因的解码器；decoder(,2)表示数据集Y特异性高可变基因的解码器；decoder(,3)表示所有共同数据里的数据X的药物响应解码器；
                                        drug_response_loss = drug_response_loss = loss_func(predicted_bulk_label, groundtruth_bulk_label)#计算bulk细胞系药物响应真实标签和预测标签之间的损失函数
                        
                        if len(torch.unique(y))>1 and len(loc[self.ref_id])!=0:
                            
                            mu_dict = {}
                            var_dict = {}

                            for j in range(self.n_domain):
                                if len(loc[j])>0:
                                    mu_dict[j] = mu[loc[j]]
                                    var_dict[j] = var[loc[j]]

                            for j in query_id:
                                if len(loc[j])>0:

                                    if label_weight is None:

                                        ot_loss_tmp, tran_batch[j] = unbalanced_ot(
                                            tran_batch[j],
                                            mu_dict[j], 
                                            var_dict[j], 
                                            mu_dict[self.ref_id].detach(), 
                                            var_dict[self.ref_id].detach(), 
                                            reg=reg,
                                            reg_m=reg_m,
                                            idx_q=idx_query[j],
                                            idx_r=idx_ref,
                                            Couple=Prior_batch, 
                                            device=device,
                                        )

                                    else:

                                        ot_loss_tmp, tran_batch[j] = unbalanced_ot(
                                            tran_batch[j],
                                            mu_dict[j], 
                                            var_dict[j], 
                                            mu_dict[self.ref_id].detach(), 
                                            var_dict[self.ref_id].detach(), 
                                            reg=reg,
                                            reg_m=reg_m,
                                            idx_q=idx_query[j],
                                            idx_r=idx_ref,
                                            Couple=Prior_batch, 
                                            device=device,
                                            query_weight=label_weight[j],
                                            ref_weight=label_weight[self.ref_id],
                                        )

                                    if save_OT:
                                        t0 = np.repeat(idx_query[j].cpu().numpy(), len(idx_ref)).reshape(len(idx_query[j]),len(idx_ref))
                                        t1 = np.tile(idx_ref.cpu().numpy(), (len(idx_query[j]), 1))
                                        tran[j][t0,t1] = tran_batch[j].cpu().numpy()

                                    ot_loss += ot_loss_tmp
   
                        if cell_regularization:
                            loss = {'recloss':lambda_recon*recon_loss, 'klloss':lambda_kl*kl_loss, 'otloss':lambda_ot*ot_loss, 'drug_R_loss':lambda_response*drug_response_loss, 'cell_R_loss':lambda_cell*cell_regularization_loss} #
                        else:
                            loss = {'recloss':lambda_recon*recon_loss, 'klloss':lambda_kl*kl_loss, 'otloss':lambda_ot*ot_loss, 'drug_response_loss':lambda_response*drug_response_loss} #
                        
                        optim.zero_grad()
                        sum(loss.values()).backward()#TODO，疑问，可以这样子后向传播吗？
                        optim.step()

                        for k,v in loss.items():
                            epoch_loss[k] += loss[k].item() #TODO 疑问，loss[k].item()是什么, k=recloss,v=2528.068359375,loss.items()=dict_items([('recloss', tensor(...)), ('klloss', tensor(...)), ('otloss', tensor(...))])
                            #print(f'#######vae.py#555rows#############(k,v) in loss.items()====k={k},v={v}')

                        info = ','.join(['{}={:.2f}'.format(k, v) for k,v in loss.items()])
                        tk0.set_postfix_str(info)
                                        
                    epoch_loss = {k:v/(i+1) for k, v in epoch_loss.items()}

                    epoch_info = ','.join(['{}={:.2f}'.format(k, v) for k,v in epoch_loss.items()])
                    tq.set_postfix_str(epoch_info) 
                torch.cuda.empty_cache()           

    
    
    def encodeBatch2( # 使用五折交叉验证
            self,
            dataloader,
            num_gene,
            mode='h',
            out='latent',
            device='cuda',
            eval=False,
            adata_cm=None, ###TODO,new feature
            DRUG='Gefitinib', ###TODO,new feature
            split='split1',
            train='valid'
    ):
        """
        Inference

        Parameters
        ----------
        dataloader
            An iterable over the given dataset for inference.
        num_gene
            List of number of genes in different datasets.
        mode
            Choose from ['h', 'v', 'd']
            If 'h', integrate data with common genes
            If 'v', integrate data profiled from the same cells
            If 'd', inetrgate data without common genes
            Default: 'h'.
        out
            Output of uniPort. Choose from ['latent', 'project', 'predict'].
            If out='latent', train the network and output cell embeddings.
            If out='project', project data into the latent space and output cell embeddings.
            If out='predict', project data into the latent space and output cell embeddings through a specified decoder.
            Default: 'latent'.
        batch_id
            Choose which encoder to project data when mode='d'. Default: 0.
        pred_id
            Choose which decoder to reconstruct data when out='predict'.
        device
            'cuda' or 'cpu' for . Default: 'cuda'.
        eval
            If True, set the model to evaluation mode. If False, set the model to train mode. Default: False.

        Returns
        -------
        output
            Cell embeddings (if out='latent' or 'project') or Predicted data (if out='predict').
        """

        self.to(device)
        if eval:
            self.eval()
        else:
            self.train()
        AUC_bulk_test=0.0
        APR_bulk_test=0.0
        AUC_sc_test=0.0
        APR_sc_test=0.0

        AUC_bulk_test_avg = 0.0
        APR_bulk_test_avg = 0.0
        AUC_sc_test_avg = 0.0
        APR_sc_test_avg = 0.0
        output = []
        if out == 'latent' or out == 'project':
            if mode == 'h':
                count_a=0
                count_b=0
                output = np.zeros((dataloader.dataset.shape[0], self.z_dim)) #output.shape=(22518, 16)
                for x,y,idx in dataloader: #=x.shape=torch.Size([11259, 4000])
                    print(f'####vae.py#838rows########count_a={count_a},count_b={count_b}')
                    x_c = x[:, 0:num_gene[self.n_domain]].float().to(device)
                    z = self.encoder(x_c, 0)[1] #？？？这里竟然取得是mu，不重采样吗，
                    loc = {}
                    loc[0] = torch.where(y==0)[0]
                    loc[1] = torch.where(y==1)[0]
                    if adata_cm is not None:
                        if len(loc[0])>0: # 表示取数据集x的隐空间嵌入z（取bulk细胞系的隐空间嵌入z ）
                            count_a+=1
                            a = idx[y==0].tolist() #定位到数据集x（bulk细胞系）的索引，用来拿到其相应的response label
                            groundtruth_bulk_label = adata_cm.obs['response'].iloc[a,].values.reshape(-1,1)
                            groundtruth_bulk_label = torch.Tensor(groundtruth_bulk_label).to(device)
                            groundtruth_bulk_label = groundtruth_bulk_label.detach().cpu().numpy()
                            predicted_bulk_label = self.decoder(z[loc[0]], 0+2+1) # decoder(,0)表示所有共同数据的共同的解码器；decoder(,1)表示数据集X特异性高可变基因的解码器；decoder(,2)表示数据集Y特异性高可变基因的解码器；decoder(,3)表示所有共同数据里的数据X的药物响应解码器；
                            predicted_bulk_label = predicted_bulk_label.detach().cpu().numpy()
                            tmp_1 = roc_auc_score(groundtruth_bulk_label, predicted_bulk_label)
                            AUC_bulk_test += tmp_1 # AUC
                            #
                            tmp_2 = average_precision_score(groundtruth_bulk_label, predicted_bulk_label)
                            APR_bulk_test += tmp_2
                        if len(loc[1])>0: # 表示取数据集x的隐空间嵌入z（取bulk细胞系的隐空间嵌入z ）
                            count_b+=1
                            b = idx[y==1].tolist() #定位到数据集x（bulk细胞系）的索引，用来拿到其相应的response label
                            groundtruth_sc_label = adata_cm.obs['response'].iloc[b,].values.reshape(-1,1)
                            groundtruth_sc_label = torch.Tensor(groundtruth_sc_label).to(device)
                            groundtruth_sc_label = groundtruth_sc_label.detach().cpu().numpy()
                            predicted_sc_label = self.decoder(z[loc[1]], 0+2+1) # decoder(,0)表示所有共同数据的共同的解码器；decoder(,1)表示数据集X特异性高可变基因的解码器；decoder(,2)表示数据集Y特异性高可变基因的解码器；decoder(,3)表示所有共同数据里的数据X的药物响应解码器；
                            predicted_sc_label = predicted_sc_label.detach().cpu().numpy()
                            tmp_3 = roc_auc_score(groundtruth_sc_label, predicted_sc_label)
                            AUC_sc_test += tmp_3 # AUC
                            #
                            tmp_4 = average_precision_score(groundtruth_sc_label, predicted_sc_label)
                            APR_sc_test += tmp_4
                    output[idx] = z.detach().cpu().numpy()

                AUC_bulk_test_avg = AUC_bulk_test/count_a
                APR_bulk_test_avg = APR_bulk_test/count_a

                AUC_sc_test_avg = AUC_sc_test/count_b
                APR_sc_test_avg = APR_sc_test/count_b
                #bulk预测
                print(f'{split}====AUC_bulk_'+train+f'==={AUC_bulk_test_avg},APR_bulk_'+train+f'==={APR_bulk_test_avg}')
                #sc预测
                print(f'{split}====AUC_sc_'+train+f'==={AUC_sc_test_avg},APR_sc_'+train+f'==={APR_sc_test_avg}')

                # 保存性能指标
                now=time.strftime("%Y-%m-%d-%H-%M-%S")
                file = './drug/'+str(DRUG)+'/'+str(DRUG)+'_auc_apr.txt'
                with open(file, 'a+') as f:
                    f.write(split+'====AUC_bulk_'+train+'==='+str(AUC_bulk_test_avg)+'\t'+
                            'APR_bulk_'+train+'==='+str(APR_bulk_test_avg)+'\t'+
                            'AUC_sc_'+train+'==='+str(AUC_sc_test_avg)+'\t'+
                            'APR_sc_'+train+'==='+str(APR_sc_test_avg)+'\t'+str(now)+'\t'+'\n')
        return AUC_bulk_test_avg,APR_bulk_test_avg,AUC_sc_test_avg,APR_sc_test_avg,output     

class TargetModel(nn.Module):
    def __init__(self, decoder,encoder):
        super(TargetModel, self).__init__()
        self.source_predcitor = decoder
        self.target_encoder = encoder

    def forward(self, X_target):
        z_tar = self.target_encoder(X_target,0)[1]
        y_src = self.source_predcitor(z_tar,3)
        return y_src

    
    
def calculateKNNgraphDistanceMatrix(featureMatrix, distanceType='euclidean', k=10):
    distMat = distance.cdist(featureMatrix,featureMatrix, distanceType)
        #print(distMat)
    edgeList=[]
    
    for i in np.arange(distMat.shape[0]):
        res = distMat[:,i].argsort()[:k]
        for j in np.arange(k):
            edgeList.append((i,res[j],distMat[i,j]))
    
    return edgeList

def generateLouvainCluster(edgeList):
   
    """
    Louvain Clustering using igraph
    """
    Gtmp = nx.Graph()
    Gtmp.add_weighted_edges_from(edgeList)
    W = nx.adjacency_matrix(Gtmp)
    W = W.todense()
    graph = Graph.Weighted_Adjacency(
        W.tolist(), mode=ADJ_UNDIRECTED, attr="weight", loops=False)
    louvain_partition = graph.community_multilevel(
        weights=graph.es['weight'], return_levels=False)
    size = len(louvain_partition)
    hdict = {}
    count = 0
    for i in range(size):
        tlist = louvain_partition[i]
        for j in range(len(tlist)):
            hdict[tlist[j]] = i
            count += 1

    listResult = []
    for i in range(count):
        listResult.append(hdict[i])

    return listResult, size
