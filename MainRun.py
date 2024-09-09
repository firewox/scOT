import scot as so
import scanpy as sc
import argparse
import time
import pandas as pd

import numpy as np

t0 = time.time()
parser = argparse.ArgumentParser(description='feature settings')
parser.add_argument('--drug_name', type=str, default='Gefitinib_scDEAL', help='drug name')
parser.add_argument('--batch_size', type=int, default=256, help='batch size')
parser.add_argument('--lambda_recon', type=float, default=10, help='lambda recon')
parser.add_argument('--lambda_response', type=float, default=1, help='lambda response')
parser.add_argument('--lambda_ot', type=float, default=1, help='lambda ot')
parser.add_argument('--lambda_kl', type=float, default=0.5, help='lambda kl')
parser.add_argument('--encoder_h_dims', type=str, default="256,256", help='encoder_hidden_dims') #VAE编码器和解码器的网络结构 TODO，new feature
parser.add_argument('--ref_id', type=int, default=0, help='targetID of optimal transmission')
parser.add_argument('--seed', type=int, default=124, help='random seed')
parser.add_argument('--geneset', type=str, default='', help='geneset name:all、_tp4k、_ppi')
parser.add_argument('--n_epoch', type=int, default=1000, help='epoch of model training')
parser.add_argument('--n_replicates', type=int, default=1, help='replicates of model training')
parser.add_argument('--seed_flag', type=int, default=1, help='1 represent fix seed; 0 represent do not fix seed')
parser.add_argument('--sampler', type=str, default='none', help='sampling method: smote、weight、none; none represent do not sample; default:smote')
parser.add_argument('--source_batch', type=int, default=190, help='batch size of Source domain data')
parser.add_argument('--target_batch', type=int, default=128, help='batch size of Target domain data')
parser.add_argument('--over_sampling_strategy', type=float, default=0.8, help='')
parser.add_argument('--under_sampling_strategy', type=float, default=0.8, help='')
parser.add_argument('--unshared_decoder', type=int, default=0, help='if unshared_decoder=1,then use two decoder; else use shared decoder')
parser.add_argument('--encoder_h_dims_source', type=str, default="1024,512,256", help='encoder_hidden_dims of bulk data')
parser.add_argument('--encoder_h_dims_target', type=str, default="256,256,256", help='encoder_hidden_dims of sc data')
parser.add_argument('--unshared_encoder', type=int, default=0, help='if unshared_encoder=1,then use two encoder; else use shared encoder')
parser.add_argument('--lambda_cell', type=float, default=1.0, help='weight of cell_regularization_loss')
parser.add_argument('--cell_regularization', type=int, default=0, help='if cell_regularization=1,then use cell regularization; else do not use')
parser.add_argument('--printgene', type=int, default=0, help='if printgene=1,then use IntegratedGradients; else do not use')
parser.add_argument('--mmd_match', type=int, default=0, help='if mmd_match=1,then use mmd matching; else do not use')
parser.add_argument('--mmd_GAMMA', type=float, default=1000.0, help='Gamma parameter in the kernel of the MMD loss of the transfer learning, default: 1000')
parser.add_argument('--lambda_mmd', type=float, default=1.0, help='weight of mmd_loss')
parser.add_argument('--drop', type=float, default=0.5, help='drop of drug response predictor model')
parser.add_argument('--out', type=str, default='latent', help='option: (latent, predict). latent represents training, and predict represents using checkpoint to predict')
parser.add_argument('--save_OT', type=int, default=0, help='option: (0, 1). 0 means not saving OT plan, and 1 means saving OT plan')
parser.add_argument('--optimal_transmission', type=int, default=1, help='option: (0, 1). 0 means not using optimal transmission, and 1 means using optimal transmission')
parser.add_argument('--random_sample', type=int, default=0, help='option: (0, 1). 0 means not randomly stratified sampling, and 1 means randomly stratified sampling')
parser.add_argument('--data_path', type=str, default="/mnt/usb/code/lyutian/git_repositories/scOT/data/", help='data path')
parser.add_argument('--verbose', type=int, default=0, help='0: display tqdm process; 1: do not display tqdm process')
args = parser.parse_args()
seed = args.seed
n_epoch = args.n_epoch
DRUG = args.drug_name
# shared encoder nodes
encoder_h_dims = args.encoder_h_dims.split(",")
encoder_h_dims = list(map(int, encoder_h_dims))
# encoder and decoder nodes of source domain data
encoder_h_dims_source = args.encoder_h_dims_source.split(",")
encoder_h_dims_source = list(map(int, encoder_h_dims_source))
# encoder and decoder nodes of target domain data
encoder_h_dims_target = args.encoder_h_dims_target.split(",")
encoder_h_dims_target = list(map(int, encoder_h_dims_target))
###########################################################1 READ Data，START
print(str(DRUG))
# linux
# data_r=pd.read_csv(args.data_path + str(DRUG)+"/Source_expr_resp_z."+str(DRUG)+str(args.geneset)+".tsv", sep='\t', index_col=0, decimal='.') # source_data_path = args.bulk_data
# windows
# data_r=pd.read_csv("F:\\git_repositories\\SCAD\\data\\split_norm\\"+str(DRUG)+"\\Source_expr_resp_z."+str(DRUG)+".tsv", sep='\t', index_col=0, decimal='.') # source_data_path = args.bulk_data
data_bulk = data_r.iloc[:,2:]
data_bulk_label = data_r.iloc[:,:1]
data_bulk_logIC50 = data_r.iloc[:,1:2]

# linux
data_t=pd.read_csv(args.data_path +str(DRUG)+"/Target_expr_resp_z."+str(DRUG)+str(args.geneset)+".tsv", sep='\t', index_col=0, decimal='.')
# windows
# data_t=pd.read_csv("F:\\git_repositories\\SCAD\\data\\split_norm\\"+str(DRUG)+"\\Target_expr_resp_z."+str(DRUG)+".tsv", sep='\t', index_col=0, decimal='.')
if args.random_sample==1:
    print(f'randomly stratified sampling')
    data_t = data_t.sample(frac=0.8)
data_sc = data_t.iloc[:,1:]
data_sc_label = data_t.iloc[:,:1]

# the lengths of data_bulk_adata and data_sc_adata are filled in by adding 0 to the column
if bool(args.unshared_encoder) and bool(args.unshared_decoder):
    data_bulk_1 = data_bulk
    data_sc_1 = data_sc
    data_bulk_1.columns = [f'a_{i}' for i in range(data_bulk.shape[1])]
    data_sc_1.columns = [f'a_{i}' for i in range(data_sc.shape[1])]
    if data_bulk.shape[1] > data_sc.shape[1]:
        l = data_bulk.shape[1] - data_sc.shape[1]
        for i in range(l):
            data_sc_1[f'a_{i+data_sc.shape[1]}'] = 0
    elif data_bulk.shape[1] < data_sc.shape[1]:
        l = data_sc.shape[1] - data_bulk.shape[1]
        for i in range(l):
            data_bulk_1[f'a_{i+data_bulk.shape[1]}'] = 0
    # transfer bulk to adata
    data_bulk_adata_1 = sc.AnnData(data_bulk_1)
    data_bulk_adata_1.obs['response']=data_bulk_label.values.reshape(-1,)
    data_bulk_adata_1.obs['status']=data_bulk_logIC50.values.reshape(-1,)
    # transfer sc to adata
    data_sc_adata_1 = sc.AnnData(data_sc_1)
    data_sc_adata_1.obs['response']=data_sc_label.values
    # 1.1 Add ‘domain_id’ and ‘source’ to the AnnDataobjects.
    data_bulk_adata_1.obs['domain_id'] = 0
    data_bulk_adata_1.obs['domain_id'] = data_bulk_adata_1.obs['domain_id'].astype('category')
    data_bulk_adata_1.obs['source'] = 'bulk'
    data_sc_adata_1.obs['domain_id'] = 1
    data_sc_adata_1.obs['domain_id'] = data_sc_adata_1.obs['domain_id'].astype('category')
    data_sc_adata_1.obs['source'] = 'scRNA'
    # 1.2 Concatenate bulkRNA-seq and scRNA-seq with common genes using AnnData.concatenate
    adata_cm = data_bulk_adata_1.concatenate(data_sc_adata_1, join='inner', batch_key='domain_id')


# transfer bulk to adata
data_bulk_adata = sc.AnnData(data_bulk)
data_bulk_adata.obs['response']=data_bulk_label.values.reshape(-1,)
data_bulk_adata.obs['status']=data_bulk_logIC50.values.reshape(-1,)

# transfer sc to adata
data_sc_adata = sc.AnnData(data_sc)
data_sc_adata.obs['response']=data_sc_label.values
###########################################################1 READ Data，END



# 1.1 Add ‘domain_id’ and ‘source’ to the AnnDataobjects.
data_bulk_adata.obs['domain_id'] = 0
data_bulk_adata.obs['domain_id'] = data_bulk_adata.obs['domain_id'].astype('category')
data_bulk_adata.obs['source'] = 'bulk'

data_sc_adata.obs['domain_id'] = 1
data_sc_adata.obs['domain_id'] = data_sc_adata.obs['domain_id'].astype('category')
data_sc_adata.obs['source'] = 'scRNA'


# 1.2 Concatenate bulkRNA-seq and scRNA-seq with common genes using AnnData.concatenate
# adata_cm = data_bulk_adata.concatenate(data_sc_adata, join='inner', batch_key='domain_id')
if not (bool(args.unshared_encoder) and bool(args.unshared_decoder)):
    adata_cm = data_bulk_adata.concatenate(data_sc_adata, join='inner', batch_key='domain_id')

###########################################################2.Training Model，START

tmp = so.Run(adatas=[data_bulk_adata, data_sc_adata],
                adata_cm=adata_cm,
                num_workers=4,
                batch_size=args.batch_size,
                ref_id=args.ref_id,
                model_info=False,
                drug_response=True,
                cell_regularization=bool(args.cell_regularization),
                DRUG=DRUG,
                lambda_kl=args.lambda_kl,
                lambda_recon=args.lambda_recon,
                lambda_response=args.lambda_response,
                lambda_ot=args.lambda_ot,
                lambda_cell=args.lambda_cell,
                encoder_h_dims=encoder_h_dims,
                n_epoch=n_epoch,
                seed=seed,
                seed_flag=bool(args.seed_flag),
                sampler=args.sampler,
                source_batch=args.source_batch,
                target_batch=args.target_batch,
                over_sampling_strategy=args.over_sampling_strategy,
                under_sampling_strategy=args.under_sampling_strategy,
                unshared_decoder=bool(args.unshared_decoder),
                encoder_h_dims_source=encoder_h_dims_source,
                encoder_h_dims_target=encoder_h_dims_target,
                unshared_encoder=bool(args.unshared_encoder),
                printgene=bool(args.printgene),
                mmd_match=bool(args.mmd_match),
                mmd_GAMMA=args.mmd_GAMMA,
                lambda_mmd=args.lambda_mmd,
                drop=args.drop,
                out=args.out,
                save_OT=bool(args.save_OT),
                optimal_transmission=bool(args.optimal_transmission),
                verbose=bool(args.verbose),)

if bool(args.save_OT):
    train = tmp[1][1]
    print(f'####MainRun.py##,tmp[1][1].shape={tmp[1][1].shape}')
    print(f'####MainRun.py##,tmp[0]={tmp[0]}')
    train = pd.DataFrame(train)
    train.to_csv('./drug/'+DRUG+'/'+str(DRUG)+'_OT_plan.csv', sep=',', index=False)
###########################################################2.Training Model，END

t1 = time.time()
print(f'####################################run total time==={(t1-t0):0.2f}s==={((t1-t0)/60):0.1f}min')
print("all finished......")