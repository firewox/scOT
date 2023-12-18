import uniport as up
import scanpy as sc
import argparse
import time
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.preprocessing import MinMaxScaler

import numpy as np


parser = argparse.ArgumentParser(description='feature settings')
parser.add_argument('--drug_name', type=str, default='Gefitinib', help='drug name')
parser.add_argument('--batch_size', type=int, default=256, help='batch size')
parser.add_argument('--lambda_recon', type=float, default=1000, help='lambda recon')
parser.add_argument('--lambda_response', type=float, default=5, help='lambda response')
parser.add_argument('--lambda_ot', type=float, default=3, help='lambda ot')
parser.add_argument('--lambda_kl', type=float, default=0.5, help='lambda kl')
parser.add_argument('--encoder_h_dims', type=str, default="256,256", help='encoder_hidden_dims') #VAE编码器和解码器的网络结构 TODO，new feature
parser.add_argument('--ref_id', type=int, default=0, help='targetID of optimal transmission')
parser.add_argument('--patience', type=int, default=50, help='early stop patient')
parser.add_argument('--seed', type=int, default=124, help='random seed')
parser.add_argument('--geneset', type=str, default='', help='geneset name:all、_tp4k、_ppi')
parser.add_argument('--sampler', type=str, default='none', help='sampling method: smote、weight、none; none represent do not sample; default:smote')
parser.add_argument('--source_batch', type=int, default=0, help='batch size of Source domain data')
parser.add_argument('--target_batch', type=int, default=0, help='batch size of Target domain data')
parser.add_argument('--over_sampling_strategy', type=float, default=0.5, help='')
parser.add_argument('--under_sampling_strategy', type=float, default=0.5, help='')
parser.add_argument('--unshared_decoder', type=int, default=0, help='if unshared_decoder=1,then use two decoder; else use shared decoder')
parser.add_argument('--encoder_h_dims_source', type=str, default="1024,512,256", help='encoder_hidden_dims of bulk data')
parser.add_argument('--encoder_h_dims_target', type=str, default="256,256,256", help='encoder_hidden_dims of sc data')
args = parser.parse_args()
DRUG = args.drug_name
# 共享编码器 shared encoder
encoder_h_dims = args.encoder_h_dims.split(",")
encoder_h_dims = list(map(int, encoder_h_dims))
# 源域数据的编码器、解码器 encoder and decoder of source domain data
encoder_h_dims_source = args.encoder_h_dims_source.split(",")
encoder_h_dims_source = list(map(int, encoder_h_dims_source))
# 目标域数据的编码器、解码器 encoder and decoder of target domain data
encoder_h_dims_target = args.encoder_h_dims_target.split(",")
encoder_h_dims_target = list(map(int, encoder_h_dims_target))

t0 = time.time()
###########################################################1 读取bulk_data，读取scRNA数据，START
# linux版本
data_r=pd.read_csv("/mnt/usb/code/lyutian/git_repositories/SCAD/data/split_norm/"+str(DRUG)+"/Source_expr_resp_z."+str(DRUG)+str(args.geneset)+".tsv", sep='\t', index_col=0, decimal='.') # source_data_path = args.bulk_data
# windows版本
# data_r=pd.read_csv("F:\\git_repositories\\SCAD\\data\\split_norm\\"+str(DRUG)+"\\Source_expr_resp_z."+str(DRUG)+".tsv", sep='\t', index_col=0, decimal='.') # source_data_path = args.bulk_data
data_bulk = data_r.iloc[:,2:]
data_bulk_label = data_r.iloc[:,:1]
data_bulk_logIC50 = data_r.iloc[:,1:2]
data_bulk_adata = sc.AnnData(data_bulk)
data_bulk_adata.obs['response']=data_bulk_label.values
data_bulk_adata.obs['logIC50']=data_bulk_logIC50.values
#归一化到0-1之间，方便使用BCEloss损失函数
scaler = MinMaxScaler()
data_bulk_adata.X = scaler.fit_transform(data_bulk_adata.X)
print(f'data_bulk_adata==={data_bulk_adata}') #n_obs × n_vars = 829 × 10610

# linux版本
data_t=pd.read_csv("/mnt/usb/code/lyutian/git_repositories/SCAD/data/split_norm/"+str(DRUG)+"/Target_expr_resp_z."+str(DRUG)+str(args.geneset)+".tsv", sep='\t', index_col=0, decimal='.')
# windows版本
# data_t=pd.read_csv("F:\\git_repositories\\SCAD\\data\\split_norm\\"+str(DRUG)+"\\Target_expr_resp_z."+str(DRUG)+".tsv", sep='\t', index_col=0, decimal='.')
data_sc = data_t.iloc[:,1:]
data_sc_label = data_t.iloc[:,:1]
data_sc_adata = sc.AnnData(data_sc)
data_sc_adata.obs['response']=data_sc_label.values
#归一化到0-1之间，方便使用BCEloss损失函数
scaler = MinMaxScaler()
data_sc_adata.X = scaler.fit_transform(data_sc_adata.X)
print(f'data_sc_adata==={data_sc_adata}') #n_obs × n_vars = 66 × 10610
###########################################################1 读取bulk_data，读取scRNA数据，END



# 1.1 Add ‘domain_id’ and ‘source’ to the AnnDataobjects.
data_bulk_adata.obs['domain_id'] = 0
data_bulk_adata.obs['domain_id'] = data_bulk_adata.obs['domain_id'].astype('category')
data_bulk_adata.obs['source'] = 'bulk'

data_sc_adata.obs['domain_id'] = 1
data_sc_adata.obs['domain_id'] = data_sc_adata.obs['domain_id'].astype('category')
data_sc_adata.obs['source'] = 'scRNA'

# 1.2 Concatenate bulkRNA-seq and scRNA-seq with common genes using AnnData.concatenate
adata_cm = data_bulk_adata.concatenate(data_sc_adata, join='inner', batch_key='domain_id')
print(f'adata_cm==={adata_cm}') #n_obs × n_vars = 895 × 10610

# 1.3 将矩阵稀疏化
data_bulk_adata.X = csr_matrix(data_bulk_adata.X)
data_sc_adata.X = csr_matrix(data_sc_adata.X)
adata_cm.X = csr_matrix(adata_cm.X)


###########################################################2.模型训练，START


# adata = up.Run(adatas=[data_bulk_adata,data_sc_adata], adata_cm=adata_cm, ref_id=0, num_workers=0, batch_size=22518-1, model_info=True)
adata = up.Run(adatas=[data_bulk_adata, data_sc_adata], adata_cm=adata_cm, num_workers=4, 
                batch_size=args.batch_size,
                ref_id=args.ref_id,
                model_info=False,
                drug_response=True,
                use_specific=False,
                DRUG=args.drug_name,
                lambda_kl=args.lambda_kl,
                lambda_recon=args.lambda_recon,
                lambda_response=args.lambda_response,
                lambda_ot=args.lambda_ot,
                encoder_h_dims=encoder_h_dims,
                patience=args.patience,
                seed=args.seed,
                sampler=args.sampler,
                source_batch=args.source_batch,
                target_batch=args.target_batch,
                over_sampling_strategy=args.over_sampling_strategy,
                under_sampling_strategy=args.under_sampling_strategy,
                unshared_decoder=bool(args.unshared_decoder),
                encoder_h_dims_source=encoder_h_dims_source,
                encoder_h_dims_target=encoder_h_dims_target,) #TODO,new feature

'''adata2 = up.Run2(LOAD_DATA_FROM='/mnt/usb/code/lyutian/git_repositories/SCAD/data/split_norm/',
               SOURCE_DIR='source_5_folds',
               TARGET_DIR='target_5_folds',
                ref_id=args.ref_id, num_workers=4, 
                batch_size=args.batch_size,
                model_info=False,
                drug_response=True,
                use_specific=False,
                DRUG=args.drug_name,
                lambda_recon=args.lambda_recon,
                lambda_response=args.lambda_response,
                lambda_ot=args.lambda_ot,
                encoder_h_dims=encoder_h_dims,
                patience=args.patience,
                seed=args.seed) #TODO,new feature''' #TODO,new feature
###########################################################2.模型训练，END

t1 = time.time()
print(f'####################################运行时间==={(t1-t0):0.2f}s==={((t1-t0)/60):0.1f}min')
print("all finished......")