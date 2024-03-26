
import torch
import numpy as np

import os
import scanpy as sc
from anndata import AnnData
import scipy
import sklearn
import pandas as pd
from scipy.sparse import issparse

from .model.vae import VAE
from .model.vae import TargetModel
from .model.utils import EarlyStopping
from .logger import create_logger
from .data_loader import load_data

from anndata import AnnData
from sklearn.preprocessing import MaxAbsScaler
from .data_loader import load_data,load_data_smote,load_data_weight, load_data_smote_unshared_encoder, load_data_weight_unshared_encoder, load_data_unshared_encoder
from glob import glob
from captum.attr import IntegratedGradients


def Run_1(
        adatas=None,     
        adata_cm=None,   
        mode='h',
        lambda_s=0.5,
        lambda_recon=1.0,
        lambda_kl=0.5,
        lambda_ot=1.0,
        lambda_response=1.0,
        iteration=30000,
        ref_id=None,    
        save_OT=False,
        use_rep=['X', 'X'],
        out='latent',
        label_weight=None,
        reg=0.1,
        reg_m=1.0,
        batch_size=256, 
        lr=2e-4, 
        enc=None,
        gpu=0, 
        prior=None,
        loss_type='BCE',
        outdir='output/', 
        input_id=0,
        pred_id=1,
        seed=124, 
        num_workers=4,
        patience=30,
        batch_key='domain_id',
        source_name='source',
        model_info=False,
        verbose=False,
        drug_response=True,
        DRUG='Gefitinib',
        encoder_h_dims=[1024,16],
        drop=0,
        sampler='smote',
        source_batch=190,
        target_batch=128,
        over_sampling_strategy=0.5,
        under_sampling_strategy=0.5,
        unshared_decoder=False,
        encoder_h_dims_source=[1024,512,256],
        encoder_h_dims_target=[256,256,256],
        seed_flag=True,
        unshared_encoder=False,
        printgene=False,
        lambda_cell=1.0,
        cell_regularization=False,
        global_match=False,
        mmd_GAMMA=1000.0,
        lambda_mmd=1.0,
    ):

    """
    Run data integration
    
    Parameters
    ----------
    adatas
        List of AnnData matrices, e.g. [adata1, adata2].
    adata_cm
        AnnData matrices containing common genes.
    mode
        Choose from ['h', 'v', 'd']
        If 'h', integrate data with common genes (Horizontal integration)
        If 'v', integrate data profiled from the same cells (Vertical integration)
        If 'd', inetrgate data without common genes (Diagonal integration)
        Default: 'h'.
    lambda_s
        Balanced parameter for common and specific genes. Default: 0.5
    lambda_recon: 
        Balanced parameter for reconstruct term. Default: 1.0
    lambda_kl: 
        Balanced parameter for KL divergence. Default: 0.5
    lambda_ot:
        Balanced parameter for OT. Default: 1.0
    iteration
        Max iterations for training. Training one batch_size samples is one iteration. Default: 30000
    ref_id
        Id of reference dataset. Default: None
    save_OT
        If True, output a global OT plan. Need more memory. Default: False
    use_rep
        Use '.X' or '.obsm'. For mode='d' only.
        If use_rep=['X','X'], use 'adatas[0].X' and 'adatas[1].X' for integration.
        If use_rep=['X','X_lsi'],  use 'adatas[0].X' and 'adatas[1].obsm['X_lsi']' for integration.
        If use_rep=['X_pca', 'X_lsi'], use 'adatas[0].obsm['X_pca']' and 'adatas[1].obsm['X_lsi']' for integration.
        Default: ['X','X']
    out
        If out=='latent', train model.
        If out=='predict', predict model.
        Default: 'latent'. 
    label_weight
        Prior-guided weighted vectors. Default: None
    reg:
        Entropy regularization parameter in OT. Default: 0.1
    reg_m:
        Unbalanced OT parameter. Larger values means more balanced OT. Default: 1.0
    batch_size
        Number of samples per batch to load. Default: 256
    lr
        Learning rate. Default: 2e-4
    enc
        Structure of encoder
    gpu
        Index of GPU to use if GPU is available. Default: 0
    prior
        Prior correspondence matrix. Default: None
    loss_type
        type of loss. 'BCE', 'MSE' or 'L1'. Default: 'BCE'
    outdir
        Output directory. Default: 'output/'
    input_id
        Only used when mode=='d' and out=='predict' to choose a encoder to project data. Default: 0
    pred_id
        Only used when out=='predict' to choose a decoder to predict data. Default: 1
    seed
        Random seed for torch and numpy. Default: 124
    patience
        early stopping patience. Default: 10
    batch_key
        Name of batch in AnnData. Default: domain_id
    source_name
        Name of source in AnnData. Default: source
    rep_celltype
        Names of cell-type annotation in AnnData. Default: 'cell_type'   
    umap
        If True, perform UMAP for visualization. Default: False
    model_info
        If True, show structures of encoder and decoders.
    verbose
        Verbosity, True or False. Default: False
    assess
        If True, calculate the entropy_batch_mixing score and silhouette score to evaluate integration results. Default: False
    show
        If True, show the UMAP visualization of latent space. Default: False
    drug_response
        if True, use drug_response decoder to predict drug response label. Default: True

    Returns
    -------
    adata.h5ad
        The AnnData matrice after integration. The representation of the data is stored at adata.obsm['latent'], adata.obsm['project'] or adata.obsm['predict'].
    checkpoint
        model.pt contains the variables of the model and config.pt contains the parameters of the model.
    log.txt
        Records model parameters.
    umap.pdf 
        UMAP plot for visualization if umap=True.
    """
    if seed_flag:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        print(f'####function.py#416rows,fix seed={seed}')
    else:
        print(f'####function.py#418rows,do not fix seed')
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark=False
    
    if mode == 'h' and adata_cm is None:
        raise AssertionError('adata_cm is needed when mode == "h"!')

    if mode not in ['h', 'd', 'v']:
        raise AssertionError('mode must be "h", "v" or "d" ')

    if adatas is None and adata_cm is None:
        raise AssertionError('at least one of adatas and adata_cm should be given!')
    if seed_flag:
        np.random.seed(seed) # seed
        torch.manual_seed(seed)

    if torch.cuda.is_available(): # cuda device
        device='cuda'
        torch.cuda.set_device(gpu)
    else:
        device='cpu'
    
    print('Device:', device)

    outdir = outdir+'/'
    os.makedirs(outdir+'/checkpoint', exist_ok=True)
    log = create_logger('', fh=outdir+'log.txt')

    # use_specific=True #TODO new feature

    # split adata_cm to adatas
    if adatas is None:  
        use_specific = False
        _, idx = np.unique(adata_cm.obs[source_name], return_index=True)
        batches = adata_cm.obs[source_name][np.sort(idx)] #bulk，scRNA
        flagged = []
        for batch in batches:
            flagged.append(adata_cm[adata_cm.obs[source_name]==batch].copy())
        adatas = flagged

    n_domain = len(adatas)

    # give reference datasets
    if ref_id is None:  
        ref_id = n_domain-1

    tran = {}
    num_cell = []
    num_gene = []

    for i, adata in enumerate(adatas):
        if use_rep[i]=='X':
            num_cell.append(adata.X.shape[0])
            num_gene.append(adata.X.shape[1])
        else:
            num_cell.append(adata.obsm[use_rep[i]].shape[0])
            num_gene.append(adata.obsm[use_rep[i]].shape[1])

    num_cell_copy = [i for i in num_cell]
    # training
    if out == 'latent':

        for i, adata in enumerate(adatas):
            # print('Dataset {}:'.format(i), adata.obs[source_name][0])
            print(adata)

        print('Reference dataset is dataset {}'.format(ref_id))
        print('\n')

        if adata_cm is not None:
            print('Data with common HVG')
            print(adata_cm)
            print('\n')

        if save_OT:
            for i in range(n_domain):
                if i != ref_id:
                    ns = num_cell[i]
                    nt = num_cell[ref_id]
                    tran_tmp = np.ones((ns, nt)) / (ns * nt)
                    tran[i] = tran_tmp.astype(np.float32)

                    print('Size of transport plan between datasets {} and {}:'.format(i, ref_id), np.shape(tran[i]))

        #trainloader, testloader = load_data(
        #    adatas=adatas, 
        #    mode=mode,
        #    use_rep=use_rep,
        #    num_cell=num_cell,
        #    max_gene=max(num_gene), 
        #    adata_cm=adata_cm,
        #    use_specific=use_specific, 
        #    domain_name=batch_key, #batch_key，Name of batch in AnnData. Default: domain_id
        #    batch_size=batch_size, #batch_size，default 256
        #    num_workers=num_workers #number parallel load processes according to cpu cores.
        #)
        if global_match:
            print(f'####function.py##515rows, 全局匹配MMD+局部匹配OT')
        else:
            print(f'####function.py##517rows, 局部匹配OT')
        if sampler=='smote':
            print(f'####function.py#497row,取样方式=sampler={sampler}')
            Ctrainloader, Ptrainloader, testloader = load_data_smote(
                num_cell_copy,
                adatas=adatas,
                source_batch=source_batch,
                target_batch=target_batch,
                drop_last=False,
                shuffle=True,
                num_workers=num_workers,
                over_sampling_strategy=over_sampling_strategy,
                under_sampling_strategy=under_sampling_strategy)
        elif sampler=='weight':
            print(f'####function.py#506row,取样方式=sampler={sampler}')
            Ctrainloader, Ptrainloader, testloader = load_data_weight(
                adatas=adatas,
                source_batch=source_batch,
                target_batch=target_batch,
                drop_last=False,
                shuffle=True,
                num_workers=num_workers)
        else:
            print(f'####function.py#515row, 不取样=sampler={sampler}')
            trainloader, testloader = load_data(
                adatas=adatas,
                mode=mode,
                use_rep=use_rep,
                num_cell=num_cell,
                max_gene=max(num_gene),
                adata_cm=adata_cm,
                use_specific=use_specific,
                domain_name=batch_key, #batch_key，Name of batch in AnnData. Default: domain_id
                batch_size=batch_size, #batch_size，default 256
                num_workers=num_workers #number parallel load processes according to cpu cores.
            )        
        #TODO,疑问，trainloader里的数据和testloader里的数据有什么区别？
        # print(f'###function.py#1##########trainloader={trainloader},testloader={testloader}')
        early_stopping = EarlyStopping(patience=patience, checkpoint_file=outdir+'/checkpoint/'+DRUG+'_model.pt', verbose=False)
        
        # encoder structure
        if enc is None:
            # enc = [] #TODO,new feature
            enc = {} #TODO,new feature, enc = {0:[...], 1:[...], 2:[...], 4:[...], 5:[...]}
            enc[0]=[]
            enc[1]=[]
            enc[2]=[]
            for index,i in enumerate(encoder_h_dims): #TODO 共享编码器0
                if index == 0:
                    enc[0].append(['fc', i, 1, 'relu', drop])
                elif index == (len(encoder_h_dims)-1):
                    enc[0].append(['fc', i, '', '', 0])
                else:
                    enc[0].append(['fc', i, 1, 'relu', drop])
            # enc = [['fc', 1024, 1, 'relu'], ['fc', 16, '', '']]
            enc[1].append(['fc', 2000, 1, 'relu', drop])
            enc[1].append(['fc', 16, 1, '', 0])
            enc[2].append(['fc', 2000, 1, 'relu', drop])
            enc[2].append(['fc', 16, 1, '', 0])    
        # decoder structure
        dec = {} 
        if mode == 'h':      
            num_gene.append(adata_cm.X.shape[1]) 
            #TODOq, new feature
            if encoder_h_dims == [1024, 16]: #TODO 共享解码器1
                dec[0] = [['fc', num_gene[n_domain], n_domain, 'sigmoid']]  # common decoder
            else: #TODO 共享解码器2
                encoder_h_dims.pop(-1)
                encoder_h_dims.reverse() #TODOq, new feature
                encoder_h_dims.append(num_gene[n_domain]) #TODOq, new feature
                dec[0]=[] #TODOq, new feature
                for index,i in enumerate(encoder_h_dims): #TODOq, new feature
                    if index == (len(encoder_h_dims)-1):
                        dec[0].append(['fc', i, n_domain, 'sigmoid', 0]) #TODO, new feature [model_name='drug_response', out_dim=1, norm=1, activation='sigmoid', droupt=0]
                    else:
                        dec[0].append(['fc', i, 1, 'relu', drop])
                #TO DO, new feature
                # dec[0] = [['fc', num_gene[n_domain], n_domain, 'sigmoid']]  # common decoder
            if use_specific: #TODO 特异性基因解码器1
                for i in range(1, n_domain+1):
                    dec[i] = [['fc', num_gene[i-1], 1, 'sigmoid']]   # dataset-specific decoder
            else: #TODO 特异性基因解码器2
                for i in range(1, n_domain+1):
                    dec[i] = [['fc', 2000, 1, 'sigmoid']]   # 不使用特异性基因的话，为了填补decoder的结构，设置个默认的，不参与模型训练
            if drug_response: #TODO 添加药物响应预测模型结构到框架中
                for i in range(n_domain+1,n_domain+2):
                    dec[i] = [['drug_response', 128, 1, 'relu', 0.5],
                              ['drug_response', 128, 1, 'relu', 0.5],
                              ['drug_response', 128, 1, 'relu', 0.5],
                              ['drug_response', 128, 1, 'relu', 0.5],
                              ['drug_response', 64, 1, 'relu', 0.5],
                              ['drug_response', 1, 3, 'sigmoid', 0]] # 使用bulk细胞系数据训练药物响应预测模型, [model_name='drug_response', out_dim=1, norm=1, activation='sigmoid', droupt=0]
                    #print(f'####药物响应预测={i},dec[i]={dec[i]}')
            if unshared_decoder: #TODO,设置两个解码器，不使用共享解码器,两个解码器的网络结构和相对应的编码器的结构刚好相反
                for i in range(n_domain+2, n_domain+4):
                    if i==(n_domain+2): #如果是源域bulk数据，先建立源域bulk的解码器
                        encoder_h_dims_source.pop(-1)
                        encoder_h_dims_source.reverse() #TODO1, new feature
                        encoder_h_dims_source.append(num_gene[0]) #TODO1, 解码器最后一层为源域数据的基因长度
                        dec[n_domain+2]=[] #TODO1, 初始为空
                        for index,i in enumerate(encoder_h_dims_source): #TODO1, 开始建立解码器结构
                            if index == (len(encoder_h_dims_source)-1): #如果是最后一层
                                dec[n_domain+2].append(['fc', i, 1, 'sigmoid', 0]) #TODO, new feature [model_name='drug_response', out_dim=1, norm=1, activation='sigmoid', droupt=0]
                            else:
                                dec[n_domain+2].append(['fc', i, 1, 'relu', drop])
                    else: #如果是目标域sc数据，建立目标域sc的解码器
                        encoder_h_dims_target.pop(-1)
                        encoder_h_dims_target.reverse() #TODO1, new feature
                        encoder_h_dims_target.append(num_gene[1]) #TODO1, 解码器最后一层为目标域sc数据的基因长度
                        dec[n_domain+3]=[] #TODO1, 初始为空
                        for index,i in enumerate(encoder_h_dims_target): #TODO1, 开始建立解码器结构
                            if index == (len(encoder_h_dims_target)-1): #如果是最后一层
                                dec[n_domain+3].append(['fc', i, 1, 'sigmoid', 0]) #TODO, new feature [model_name='drug_response', out_dim=1, norm=1, activation='sigmoid', droupt=0]
                            else:
                                dec[n_domain+3].append(['fc', i, 1, 'relu', drop])
            
        else:
            for i in range(n_domain):
                dec[i] = [['fc', num_gene[i], 1, 'sigmoid']]    # dataset-specific decoder

        # init model
        model = VAE(enc, dec, ref_id=ref_id, n_domain=n_domain, mode=mode,
                    batch_size=batch_size,
                    lambda_recon=lambda_recon,
                    lambda_kl=lambda_kl,
                    lambda_ot=lambda_ot,
                    lambda_response=lambda_response)
        # print(f'####function.py#504rows###############model={model}')
        if model_info:
            log.info('model\n'+model.__repr__())
        if cell_regularization:
            print(f'####function.py##648rows, 使用细胞正则化')
        if unshared_decoder: #TODO 不使用共享解码器，使用两个不同的解码器, unshared_decoder=True
            if sampler=='smote' or sampler=='weight':
                #print(f'####function.py#598row, 测试0')
                model.fit_unshared_decoder(
                    # trainloader,
                    Ctrainloader, #TODO new feature
                    Ptrainloader, #TODO new feature
                    tran,
                    num_cell=num_cell_copy,
                    num_gene=num_gene,
                    mode=mode,
                    label_weight=label_weight,
                    Prior=prior,
                    save_OT=save_OT,
                    use_specific=use_specific,
                    lambda_s=lambda_s,
                    lambda_recon=lambda_recon,
                    lambda_kl=lambda_kl,
                    lambda_ot=lambda_ot,
                    lambda_response=lambda_response,#TODO new feature
                    reg=reg,
                    reg_m=reg_m,
                    lr=lr,
                    max_iteration=iteration,
                    device=device,
                    early_stopping=early_stopping,
                    verbose=verbose,
                    loss_type=loss_type,
                    drug_response=drug_response,
                    adata_cm=adata_cm,
                    unshared_decoder=unshared_decoder,
                    lambda_cell=lambda_cell,#TODO new feature
                    cell_regularization=cell_regularization, #TODO new feature
                    global_match=global_match, #TODO new feature
                    mmd_GAMMA=mmd_GAMMA, #TODO new feature
                    lambda_mmd=lambda_mmd, #TODO new feature
                )
            else:
                model.fit_1_unshared_decoder(
                    trainloader,
                    tran,
                    num_cell,
                    num_gene,
                    mode=mode,
                    label_weight=label_weight,
                    Prior=prior,
                    save_OT=save_OT,
                    use_specific=use_specific,
                    lambda_s=lambda_s,
                    lambda_recon=lambda_recon,
                    lambda_kl=lambda_kl,
                    lambda_ot=lambda_ot,
                    lambda_response=lambda_response,#TODO new feature
                    reg=reg,
                    reg_m=reg_m,
                    lr=lr,
                    max_iteration=iteration,
                    device=device,
                    early_stopping=early_stopping,
                    verbose=verbose,
                    loss_type=loss_type,
                    drug_response=drug_response,
                    adata_cm=adata_cm,
                    unshared_decoder=unshared_decoder,
                    lambda_cell=lambda_cell,#TODO new feature
                    cell_regularization=cell_regularization, #TODO new feature
                    global_match=global_match, #TODO new feature
                    mmd_GAMMA=mmd_GAMMA, #TODO new feature
                    lambda_mmd=lambda_mmd, #TODO new feature
                )
        else: #TODO 使用共享解码器, unshared_decoder=False
            if sampler=='smote' or sampler=='weight':
                #print(f'####function.py#598row, 测试0')
                model.fit(
                    # trainloader,
                    Ctrainloader, #TODO new feature
                    Ptrainloader, #TODO new feature
                    tran,
                    num_cell=num_cell_copy,
                    num_gene=num_gene,
                    mode=mode,
                    label_weight=label_weight,
                    Prior=prior,
                    save_OT=save_OT,
                    use_specific=use_specific,
                    lambda_s=lambda_s,
                    lambda_recon=lambda_recon,
                    lambda_kl=lambda_kl,
                    lambda_ot=lambda_ot,
                    lambda_response=lambda_response,#TODO new feature
                    reg=reg,
                    reg_m=reg_m,
                    lr=lr,
                    max_iteration=iteration,
                    device=device,
                    early_stopping=early_stopping,
                    verbose=verbose,
                    loss_type=loss_type,
                    drug_response=drug_response,
                    adata_cm=adata_cm
                )
            else:
                model.fit_1(
                    trainloader,
                    tran,
                    num_cell,
                    num_gene,
                    mode=mode,
                    label_weight=label_weight,
                    Prior=prior,
                    save_OT=save_OT,
                    use_specific=use_specific,
                    lambda_s=lambda_s,
                    lambda_recon=lambda_recon,
                    lambda_kl=lambda_kl,
                    lambda_ot=lambda_ot,
                    lambda_response=lambda_response,#TODO new feature
                    reg=reg,
                    reg_m=reg_m,
                    lr=lr,
                    max_iteration=iteration,
                    device=device,
                    early_stopping=early_stopping,
                    verbose=verbose,
                    loss_type=loss_type,
                    drug_response=drug_response,
                    adata_cm=adata_cm
                )
        torch.save({'enc':enc, 'dec':dec, 'n_domain':n_domain, 'ref_id':ref_id, 'num_gene':num_gene}, outdir+'/checkpoint/'+DRUG+'_config.pt')     


    # project or predict
    else:
        state = torch.load(outdir+'/checkpoint/'+DRUG+'_config.pt')
        enc, dec, n_domain, ref_id, num_gene = state['enc'], state['dec'], state['n_domain'], state['ref_id'], state['num_gene']
        model = VAE(enc, dec, ref_id=ref_id, n_domain=n_domain, mode=mode)
        model.load_model(outdir+'/checkpoint/model.pt')
        model.to(device)
        
        _, testloader = load_data(
            adatas=adatas, 
            max_gene=max(num_gene), 
            num_cell=num_cell,
            adata_cm=adata_cm, 
            domain_name=batch_key,
            batch_size=batch_size, 
            mode=mode
        )

    if mode == 'v':
        adatas[0].obsm[out] = model.encodeBatch(testloader, num_gene, pred_id=pred_id, device=device, mode=mode, out=out,source_batch=source_batch,target_batch=target_batch,sampler=sampler)
        return adatas[0]

    elif mode == 'd':
        if out == 'latent' or out == 'project':
            for i in range(n_domain):
                adatas[i].obsm[out] = model.encodeBatch(testloader, num_gene, batch_id=i, device=device, mode=mode, out=out,source_batch=source_batch,target_batch=target_batch,sampler=sampler)
            for i in range(n_domain-1):
                adata_concat = adatas[i].concatenate(adatas[i+1])
        elif out == 'predict':
            adatas[0].obsm[out] = model.encodeBatch(testloader, num_gene, batch_id=input_id, pred_id=pred_id, device=device, mode=mode, out=out,source_batch=source_batch,target_batch=target_batch,sampler=sampler)

    elif mode == 'h':
        if out == 'latent' or out == 'project':
            #adata_cm.obsm[out] = model.encodeBatch(testloader, num_gene, device=device, mode=mode,out=out, eval=True, DRUG=DRUG, adata_cm=adata_cm,source_batch=source_batch,target_batch=target_batch,sampler=sampler) # save latent rep
            #TODO new feature
            if sampler=='smote' or sampler=='weight':
                model.encodeBatch(adata_cm, adatas, dataloader=testloader, num_cell=num_cell,num_gene=num_gene, device=device, mode=mode, out=out, eval=True,DRUG=DRUG, source_batch=source_batch,target_batch=target_batch,sampler=sampler) # save latent rep
            else:
                model.encodeBatch_1(adata_cm, adatas, dataloader=testloader,num_gene=num_gene, num_cell=num_cell, device=device, mode=mode, out=out, eval=True,DRUG=DRUG, source_batch=source_batch,target_batch=target_batch,sampler=sampler) # save latent rep
            #adata_cm.obsm[out] = model.encodeBatch(testloader, num_gene, pred_id=pred_id, device=device, mode=mode,out=out, eval=True, DRUG=DRUG, adata_cm=adata_cm,source_batch=source_batch,target_batch=target_batch,sampler=sampler)
    torch.cuda.empty_cache()
    if mode == 'h':
        if save_OT:
            return adata_cm, tran
        return adata_cm

    else:
        if save_OT:
            return tran


def Run_2(
        LOAD_DATA_FROM='F:/git_repositories/SCAD'+'/data/split_norm/',
        SOURCE_DIR='source_5_folds',
        TARGET_DIR='target_5_folds',  
        mode='h',
        lambda_s=0.5,
        lambda_recon=1.0,
        lambda_kl=0.5,
        lambda_ot=1.0,
        lambda_response=1.0,
        iteration=30000,
        ref_id=None,    
        save_OT=False,
        use_rep=['X', 'X'],
        out='latent',
        label_weight=None,
        reg=0.1,
        reg_m=1.0,
        batch_size=256, 
        lr=2e-4, 
        enc=None,
        gpu=0, 
        prior=None,
        loss_type='BCE',
        outdir='output/', 
        input_id=0,
        pred_id=1,
        seed=124, 
        num_workers=4,
        patience=30,
        batch_key='domain_id',
        source_name='source',
        model_info=False,
        verbose=False,
        drug_response=True,
        use_specific=False,
        DRUG='Gefitinib',
        encoder_h_dims=[1024,16],
        drop=0
    ):

    """
    Run data integration
    
    Parameters
    ----------
    lambda_s
        Balanced parameter for common and specific genes. Default: 0.5
    lambda_recon: 
        Balanced parameter for reconstruct term. Default: 1.0
    lambda_kl: 
        Balanced parameter for KL divergence. Default: 0.5
    lambda_ot:
        Balanced parameter for OT. Default: 1.0
    iteration
        Max iterations for training. Training one batch_size samples is one iteration. Default: 30000
    ref_id
        Id of reference dataset. Default: None
    save_OT
        If True, output a global OT plan. Need more memory. Default: False
    use_rep
        Use '.X' or '.obsm'. For mode='d' only.
        If use_rep=['X','X'], use 'adatas[0].X' and 'adatas[1].X' for integration.
        If use_rep=['X','X_lsi'],  use 'adatas[0].X' and 'adatas[1].obsm['X_lsi']' for integration.
        If use_rep=['X_pca', 'X_lsi'], use 'adatas[0].obsm['X_pca']' and 'adatas[1].obsm['X_lsi']' for integration.
        Default: ['X','X']
    out
        If out=='latent', train model.
        If out=='predict', predict.
        Default: 'latent'. 
    label_weight
        Prior-guided weighted vectors. Default: None
    reg:
        Entropy regularization parameter in OT. Default: 0.1
    reg_m:
        Unbalanced OT parameter. Larger values means more balanced OT. Default: 1.0
    batch_size
        Number of samples per batch to load. Default: 256
    lr
        Learning rate. Default: 2e-4
    enc
        Structure of encoder
    gpu
        Index of GPU to use if GPU is available. Default: 0
    prior
        Prior correspondence matrix. Default: None
    loss_type
        type of loss. 'BCE', 'MSE' or 'L1'. Default: 'BCE'
    outdir
        Output directory. Default: 'output/'
    input_id
        Only used when mode=='d' and out=='predict' to choose a encoder to project data. Default: 0
    pred_id
        Only used when out=='predict' to choose a decoder to predict data. Default: 1
    seed
        Random seed for torch and numpy. Default: 124
    patience
        early stopping patience. Default: 10
    batch_key
        Name of batch in AnnData. Default: domain_id
    source_name
        Name of source in AnnData. Default: source
    rep_celltype
        Names of cell-type annotation in AnnData. Default: 'cell_type'   
    umap
        If True, perform UMAP for visualization. Default: False
    model_info
        If True, show structures of encoder and decoders.
    verbose
        Verbosity, True or False. Default: False
    assess
        If True, calculate the entropy_batch_mixing score and silhouette score to evaluate integration results. Default: False
    show
        If True, show the UMAP visualization of latent space. Default: False
    drug_response
        if True, use drug_response decoder to predict drug response label. Default: True

    Returns
    -------
    adata.h5ad
        The AnnData matrice after integration. The representation of the data is stored at adata.obsm['latent'], adata.obsm['project'] or adata.obsm['predict'].
    checkpoint
        model.pt contains the variables of the model and config.pt contains the parameters of the model.
    log.txt
        Records model parameters.
    umap.pdf 
        UMAP plot for visualization if umap=True.
    """


    np.random.seed(seed) # seed
    torch.manual_seed(seed)

    if torch.cuda.is_available(): # cuda device
        device='cuda'
        torch.cuda.set_device(gpu)
    else:
        device='cpu'
    
    print('Device:', device)

    outdir = outdir+'/'
    os.makedirs(outdir+'/checkpoint', exist_ok=True)
    log = create_logger('', fh=outdir+'log.txt')

    ##############START# 加载0.8bulk数据和0.8sc数据，这个数据用作训练模型。
    ls_splits = ['split1', 'split2', 'split3','split4','split5']
    LOAD_DATA_FROM = LOAD_DATA_FROM + DRUG + '/stratified/'
    AUCtest_splits_total_ValTarget = []
    APRtest_splits_total_ValTarget = []
    AUCtest_splits_total_TrainTarget = []
    APRtest_splits_total_TrainTarget = []
    AUCtest_splits_total_ValSource = [] #TODO，添加，new feature
    APRtest_splits_total_ValSource = [] #TODO，添加，new feature
    AUCtest_splits_total_TrainSource = [] #TODO，添加，new feature
    APRtest_splits_total_TrainSource = [] #TODO，添加，new feature
    for split in ls_splits:		# for each split
        print("\n\nReading data for {} ...\n".format(split))
        XTrainGDSC = pd.read_csv(LOAD_DATA_FROM + SOURCE_DIR + '/' + split + '/X_train_source.tsv',
                                 sep='\t', index_col=0, decimal='.')
        YTrainGDSC = pd.read_csv(LOAD_DATA_FROM + SOURCE_DIR + '/' + split + '/Y_train_source.tsv',
                                 sep='\t', index_col=0, decimal='.')
        XValGDSC = pd.read_csv(LOAD_DATA_FROM + SOURCE_DIR + '/' + split + '/X_val_source.tsv',
                               sep='\t', index_col=0, decimal='.')
        YValGDSC = pd.read_csv(LOAD_DATA_FROM + SOURCE_DIR + '/' + split + '/Y_val_source.tsv',
                               sep='\t', index_col=0, decimal='.')
        # Loading Target (Single cell) Data, YTrainCells are not used during training#
        XTrainCells = pd.read_csv(LOAD_DATA_FROM + TARGET_DIR + '/' + split + '/X_train_target.tsv',
                                  sep='\t', index_col=0, decimal='.')
        YTrainCells = pd.read_csv(LOAD_DATA_FROM + TARGET_DIR + '/' + split + '/Y_train_target.tsv',
                                  sep='\t', index_col=0, decimal='.')
        XTestCells = pd.read_csv(LOAD_DATA_FROM + TARGET_DIR + '/' + split + '/X_test_target.tsv',
                                 sep='\t', index_col=0, decimal='.')
        YTestCells = pd.read_csv(LOAD_DATA_FROM + TARGET_DIR + '/' + split + '/Y_test_target.tsv',
                                 sep='\t', index_col=0, decimal='.')
        #读取0.8bulk数据
        from sklearn.preprocessing import MinMaxScaler
        from scipy.sparse import csr_matrix
        data_bulk_adata = sc.AnnData(XTrainGDSC)
        data_bulk_adata.obs['response']=YTrainGDSC.values
        scaler = MinMaxScaler()
        data_bulk_adata.X = scaler.fit_transform(data_bulk_adata.X)
        data_bulk_adata.obs['domain_id'] = 0
        data_bulk_adata.obs['domain_id'] = data_bulk_adata.obs['domain_id'].astype('category')
        data_bulk_adata.obs['source'] = 'bulk'
        #读取0.8sc数据
        data_sc_adata = sc.AnnData(XTrainCells)
        data_sc_adata.obs['response']=YTrainCells.values
        #TODO，归一化到0-1之间，方便使用BCEloss损失函数
        scaler = MinMaxScaler()
        data_sc_adata.X = scaler.fit_transform(data_sc_adata.X)
        data_sc_adata.obs['domain_id'] = 1
        data_sc_adata.obs['domain_id'] = data_sc_adata.obs['domain_id'].astype('category')
        data_sc_adata.obs['source'] = 'scRNA'
        # Concatenate bulkRNA-seq and scRNA-seq with common genes using AnnData.concatenate
        adata_cm = data_bulk_adata.concatenate(data_sc_adata, join='inner', batch_key='domain_id')
        # 将矩阵稀疏化
        data_bulk_adata.X = csr_matrix(data_bulk_adata.X)
        data_sc_adata.X = csr_matrix(data_sc_adata.X)
        adata_cm.X = csr_matrix(adata_cm.X)

        # 读取0.2bulk数据
        data_bulk_adata_valid = sc.AnnData(XValGDSC)
        data_bulk_adata_valid.obs['response']=YValGDSC.values
        scaler = MinMaxScaler()
        data_bulk_adata_valid.X = scaler.fit_transform(data_bulk_adata_valid.X)
        data_bulk_adata_valid.obs['domain_id'] = 0
        data_bulk_adata_valid.obs['domain_id'] = data_bulk_adata_valid.obs['domain_id'].astype('category')
        data_bulk_adata_valid.obs['source'] = 'bulk'
        #读取0.2sc数据
        data_sc_adata_valid = sc.AnnData(XTestCells)
        data_sc_adata_valid.obs['response']=YTestCells.values
        scaler = MinMaxScaler()
        data_sc_adata_valid.X = scaler.fit_transform(data_sc_adata_valid.X)
        data_sc_adata_valid.obs['domain_id'] = 1
        data_sc_adata_valid.obs['domain_id'] = data_sc_adata_valid.obs['domain_id'].astype('category')
        data_sc_adata_valid.obs['source'] = 'scRNA'
        # Concatenate bulkRNA-seq and scRNA-seq with common genes using AnnData.concatenate
        adata_cm_valid = data_bulk_adata_valid.concatenate(data_sc_adata_valid, join='inner', batch_key='domain_id')
        # 将矩阵稀疏化
        data_bulk_adata_valid.X = csr_matrix(data_bulk_adata_valid.X)
        data_sc_adata_valid.X = csr_matrix(data_sc_adata_valid.X)
        adata_cm_valid.X = csr_matrix(adata_cm_valid.X)

        adatas = [data_bulk_adata,data_sc_adata]
        adatas_valid = [data_bulk_adata_valid,data_sc_adata_valid]

        # split adata_cm to adatas
        if adatas is None:
            use_specific = False
            _, idx = np.unique(adata_cm.obs[source_name], return_index=True)
            batches = adata_cm.obs[source_name][np.sort(idx)] #bulk，scRNA
            flagged = []
            for batch in batches:
                flagged.append(adata_cm[adata_cm.obs[source_name]==batch].copy())
            adatas = flagged

        n_domain = len(adatas)

        # give reference datasets
        if ref_id is None:
            ref_id = n_domain-1

        tran = {}
        num_cell = []
        num_gene = []

        for i, adata in enumerate(adatas):
            if use_rep[i]=='X':
                num_cell.append(adata.X.shape[0])
                num_gene.append(adata.X.shape[1])
            else:
                num_cell.append(adata.obsm[use_rep[i]].shape[0])
                num_gene.append(adata.obsm[use_rep[i]].shape[1])


        # training
        if out == 'latent':

            for i, adata in enumerate(adatas):
                # print('Dataset {}:'.format(i), adata.obs[source_name][0])
                print(adata)

            print('Reference dataset is dataset {}'.format(ref_id))
            print('\n')

            if adata_cm is not None:
                print('Data with common HVG')
                print(adata_cm)
                print('\n')

            if save_OT:
                for i in range(n_domain):
                    if i != ref_id:
                        ns = num_cell[i]
                        nt = num_cell[ref_id]
                        tran_tmp = np.ones((ns, nt)) / (ns * nt)
                        tran[i] = tran_tmp.astype(np.float32)

                        print('Size of transport plan between datasets {} and {}:'.format(i, ref_id), np.shape(tran[i]))

            trainloader, testloader = load_data(
                adatas=adatas,
                mode=mode,
                use_rep=use_rep,
                num_cell=num_cell,
                max_gene=max(num_gene),
                adata_cm=adata_cm,
                use_specific=use_specific,
                domain_name=batch_key, #batch_key，Name of batch in AnnData. Default: domain_id
                batch_size=batch_size, #batch_size，default 256
                num_workers=num_workers #number parallel load processes according to cpu cores.
            )
            _,testloader_valid = load_data(
                adatas=adatas_valid,
                mode=mode,
                use_rep=use_rep,
                num_cell=num_cell,
                max_gene=max(num_gene),
                adata_cm=adata_cm_valid,
                use_specific=use_specific,
                domain_name=batch_key, #batch_key，Name of batch in AnnData. Default: domain_id
                batch_size=batch_size, #batch_size，default 256
                num_workers=num_workers #number parallel load processes according to cpu cores.
            )
            #TODO,疑问，trainloader里的数据和testloader里的数据有什么区别？
            # print(f'###function.py#1##########trainloader={trainloader},testloader={testloader}')
            early_stopping = EarlyStopping(patience=patience, checkpoint_file=outdir+'/checkpoint/'+DRUG+'_model.pt', verbose=False)

            # encoder structure
            if enc is None:
                enc = [] #TODO,new feature
                for index,i in enumerate(encoder_h_dims):
                    if index == 0:
                        enc.append(['fc', i, 1, 'relu', drop])
                    elif index == (len(encoder_h_dims)-1):
                        enc.append(['fc', i, '', '', 0])
                    else:
                        enc.append(['fc', i, 1, 'relu', drop])
                # enc = [['fc', 1024, 1, 'relu'], ['fc', 16, '', '']]

            # decoder structure
            dec = {}

            if mode == 'h':
                num_gene.append(adata_cm.X.shape[1])
                #TODO, new feature
                if encoder_h_dims == [1024, 16]:
                    dec[0] = [['fc', num_gene[n_domain], n_domain, 'sigmoid']]  # common decoder
                else:
                    encoder_h_dims.pop(-1)
                    encoder_h_dims.reverse() #TODO, new feature
                    encoder_h_dims.append(num_gene[n_domain]) #TODO, new feature
                    dec[0]=[] #TODO, new feature
                    for index,i in enumerate(encoder_h_dims): #TODO, new feature
                        if index == (len(encoder_h_dims)-1):
                            dec[0].append(['fc', i, n_domain, 'sigmoid', 0]) #TODO, new feature
                        else:
                            dec[0].append(['fc', i, 1, 'relu', drop])
                    #TO DO, new feature
                    # dec[0] = [['fc', num_gene[n_domain], n_domain, 'sigmoid']]  # common decoder
                if use_specific:
                    for i in range(1, n_domain+1):
                        dec[i] = [['fc', num_gene[i-1], 1, 'sigmoid']]   # dataset-specific decoder
                else:
                    for i in range(1, n_domain+1):
                        dec[i] = [['fc', 2000, 1, 'sigmoid']]   # 不使用特异性基因的话，为了填补decoder的结构，设置个默认的，不参与模型训练
                if drug_response: #TODO 添加药物响应预测模型结构到框架中
                    for i in range(n_domain+1,n_domain+2):
                        dec[i] = [['drug_response', 128, 1, 'relu', 0.5],
                                  ['drug_response', 128, 1, 'relu', 0.5],
                                  ['drug_response', 128, 1, 'relu', 0.5],
                                  ['drug_response', 128, 1, 'relu', 0.5],
                                  ['drug_response', 64, 1, 'relu', 0.5],
                                  ['drug_response', 1, 3, 'sigmoid', 0]] # 使用bulk细胞系数据训练药物响应预测模型, [model_name='drug_response', out_dim=1, norm=1, activation='sigmoid', droupt=0]
                        #print(f'####药物响应预测={i},dec[i]={dec[i]}')

            else:
                for i in range(n_domain):
                    dec[i] = [['fc', num_gene[i], 1, 'sigmoid']]    # dataset-specific decoder

            # init model
            model = VAE(enc, dec, ref_id=ref_id, n_domain=n_domain, mode=mode,
                        batch_size=batch_size,
                        lambda_recon=lambda_recon,
                        lambda_kl=lambda_kl,
                        lambda_ot=lambda_ot,
                        lambda_response=lambda_response)
            # print(f'####function.py#504rows###############model={model}')
            if model_info:
                log.info('model\n'+model.__repr__())

            model.fit(
                trainloader,
                tran,
                num_cell,
                num_gene,
                mode=mode,
                label_weight=label_weight,
                Prior=prior,
                save_OT=save_OT,
                use_specific=use_specific,
                lambda_s=lambda_s,
                lambda_recon=lambda_recon,
                lambda_kl=lambda_kl,
                lambda_ot=lambda_ot,
                lambda_response=lambda_response,#TODO new feature
                reg=reg,
                reg_m=reg_m,
                lr=lr,
                max_iteration=iteration,
                device=device,
                early_stopping=early_stopping,
                verbose=verbose,
                loss_type=loss_type,
                drug_response=drug_response,
                adata_cm=adata_cm
            )
            torch.save({'enc':enc, 'dec':dec, 'n_domain':n_domain, 'ref_id':ref_id, 'num_gene':num_gene}, outdir+'/checkpoint/'+DRUG+'_config.pt')

        if mode == 'h':
            if out == 'latent' or out == 'project':
                #0.8的bulk和0.8的sc
                AUC_TrainSource,APR_TrainSource,AUC_TrainTarget,APR_TrainTarget,adata_cm.obsm[out] = model.encodeBatch2(testloader, num_gene, device=device, mode=mode, out=out, eval=True,DRUG=DRUG,adata_cm=adata_cm,train='train',split=split) # save latent rep
                #0.8bulk数据测试的结果
                AUCtest_splits_total_TrainSource.append(AUC_TrainSource)
                APRtest_splits_total_TrainSource.append(APR_TrainSource)
                #0.8sc数据测试的结果
                AUCtest_splits_total_TrainTarget.append(AUC_TrainTarget)
                APRtest_splits_total_TrainTarget.append(APR_TrainTarget)
                #0.2的sc和0.2的sc
                AUC_ValidSource,APR_ValidSource,AUC_ValidTarget,APR_ValidTarget,adata_cm_valid.obsm[out] = model.encodeBatch2(testloader_valid, num_gene, device=device, mode=mode, out=out, eval=True,DRUG=DRUG,adata_cm=adata_cm_valid,train='valid',split=split) # save latent rep
                #0.2bulk数据测试的结果
                AUCtest_splits_total_ValSource.append(AUC_ValidSource)
                APRtest_splits_total_ValSource.append(APR_ValidSource)
                #0.2sc数据测试的结果
                AUCtest_splits_total_ValTarget.append(AUC_ValidTarget)
                APRtest_splits_total_ValTarget.append(APR_ValidTarget)

    #0.8bulk
    AUCtest_splits_total_TrainSource = np.array(AUCtest_splits_total_TrainSource)
    APRtest_splits_total_TrainSource = np.array(APRtest_splits_total_TrainSource)
    avgAUC_bulk_train = np.mean(AUCtest_splits_total_TrainSource)
    avgAPR_bulk_train = np.mean(APRtest_splits_total_TrainSource)
    #0.8sc
    AUCtest_splits_total_TrainTarget = np.array(AUCtest_splits_total_TrainTarget)
    APRtest_splits_total_TrainTarget = np.array(APRtest_splits_total_TrainTarget)
    avgAUC_sc_train = np.mean(AUCtest_splits_total_TrainTarget)
    avgAPR_sc_train = np.mean(APRtest_splits_total_TrainTarget)

    #0.2bulk
    AUCtest_splits_total_ValSource = np.array(AUCtest_splits_total_ValSource)
    APRtest_splits_total_ValSource = np.array(APRtest_splits_total_ValSource)
    avgAUC_bulk_valid = np.mean(AUCtest_splits_total_ValSource)
    avgAPR_bulk_valid = np.mean(APRtest_splits_total_ValSource)
    #0.2sc
    AUCtest_splits_total_ValTarget = np.array(AUCtest_splits_total_ValTarget)
    APRtest_splits_total_ValTarget = np.array(APRtest_splits_total_ValTarget)
    avgAUC_sc_valid = np.mean(AUCtest_splits_total_ValTarget)
    avgAPR_sc_valid = np.mean(APRtest_splits_total_ValTarget)
    #0.8bulk预测
    print(f'avg====AUC_bulk_train==={avgAUC_bulk_train},APR_bulk_train==={avgAPR_bulk_train}')
    #0.8sc预测
    print(f'avg====AUC_sc_train==={avgAUC_sc_train},APR_sc_train==={avgAPR_sc_train}')
    
    #0.2bulk预测
    print(f'avg====AUC_bulk_valid==={avgAUC_bulk_valid},APR_bulk_valid==={avgAPR_bulk_valid}')
    #0.2sc预测
    print(f'avg====AUC_sc_valid==={avgAUC_sc_valid},APR_sc_valid==={avgAPR_sc_valid}')
    import time
    now=time.strftime("%Y-%m-%d-%H-%M-%S")
    file = './drug/'+str(DRUG)+'/'+str(DRUG)+'_auc_apr.txt'
    with open(file, 'a+') as f:
        f.write('====AUC_bulk_train_avg==='+str(avgAUC_bulk_train)+'\t'+
                'APR_bulk_train_avg==='+str(avgAPR_bulk_train)+'\t'+
                'AUC_sc_train_avg==='+str(avgAUC_sc_train)+'\t'+
                'APR_sc_train_avg==='+str(avgAPR_sc_train)+'\t'+str(now)+'\t'+'\n')
        f.write('====AUC_bulk_valid_avg==='+str(avgAUC_bulk_valid)+'\t'+
                'APR_bulk_valid_avg==='+str(avgAPR_bulk_valid)+'\t'+
                'AUC_sc_valid_avg==='+str(avgAUC_sc_valid)+'\t'+
                'APR_sc_valid_avg==='+str(avgAPR_sc_valid)+'\t'+str(now)+'\t'+'\n\n')
                
                
def Run(
        adatas=None,     
        adata_cm=None,   
        lambda_recon=1.0,#作用在重建损失上，包括共同高表达基因、特异性高表达基因的损失
        lambda_kl=0.5,
        lambda_ot=1.0,
        lambda_response=1.0,
        lambda_cell=1.0,
        ref_id=0,
        save_OT=False,
        out='latent',
        reg=0.1,
        reg_m=1.0,
        batch_size=256, 
        lr=2e-4, 
        enc=None,
        gpu=0, 
        prior=None,
        loss_type='BCE',
        outdir='output/', 
        num_workers=4,
        batch_key='domain_id',
        source_name='source',
        model_info=False,
        verbose=False,
        drug_response=True,
        cell_regularization=False,
        DRUG='Gefitinib',
        encoder_h_dims=[1024,16],
        drop=0.5,
        n_epoch=1000,
        seed=124,
        seed_flag=True,#TODO new feature
        sampler='none',
        source_batch=190,
        target_batch=128,
        over_sampling_strategy=0.5,#smote sampling strategy
        under_sampling_strategy=0.5,#smote sampling strategy
        unshared_decoder=True,
        encoder_h_dims_source=[1024,512,256],
        encoder_h_dims_target=[256,256,256],
        unshared_encoder=False,
        printgene=False,
        mmd_match=False,
        mmd_GAMMA=1000.0,
        lambda_mmd=1.0,
        optimal_transmission=True,
    ):

    """
    Run data integration
    
    Parameters
    ----------
    adatas
        List of AnnData matrices, e.g. [adata1, adata2].
    adata_cm
        AnnData matrices containing common genes.
    lambda_recon: 
        Balanced parameter for reconstruct term. Default: 1.0
    lambda_kl: 
        Balanced parameter for KL divergence. Default: 0.5
    lambda_ot:
        Balanced parameter for OT. Default: 1.0
    ref_id
        Id of reference dataset. Default: 0
    save_OT
        If True, output a global OT plan. Need more memory. Default: False
    out
        Output of scot. Choose from ['latent', 'predict'].
        If out=='latent', train the network and output cell embeddings.
        If out=='predict', project single cell data into the latent space and output cell drug response predictions through a specified decoder.
        Default: 'latent'. 
    reg:
        Entropy regularization parameter in OT. Default: 0.1
    reg_m:
        Unbalanced OT parameter. Larger values means more balanced OT. Default: 1.0
    batch_size
        Number of samples per batch to load. Default: 256
    lr
        Learning rate. Default: 2e-4
    enc
        Structure of encoder
    gpu
        Index of GPU to use if GPU is available. Default: 0
    loss_type
        type of loss. 'BCE', 'MSE' or 'L1'. Default: 'BCE'
    outdir
        Output directory. Default: 'output/'
    seed
        Random seed for torch and numpy. Default: 124
    batch_key
        Name of batch in AnnData. Default: domain_id
    source_name
        Name of source in AnnData. Default: source
    model_info
        If True, show structures of encoder and decoders.
    verbose
        Verbosity, True or False. Default: False
    drug_response
        if True, use drug_response decoder to predict drug response label. Default: True

    Returns
    -------
    adata.h5ad
        The AnnData matrice after integration. The representation of the data is stored at adata.obsm['latent'] or adata.obsm['predict'].
    checkpoint
        model.pt contains the variables of the model and config.pt contains the parameters of the model.
    log.txt
        Records model parameters.
    """
    
    if seed_flag:
        torch.manual_seed(seed)
        print(f'####function.py#1252row,fix seed={seed}')
    else:
        print(f'####function.py#1254row,do not fix seed')
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    np.random.seed(42)
    os.environ['PYTHONHASHSEED'] = str(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark=False
    
    if adata_cm is None:
        raise AssertionError('adata_cm is None')

    if adatas is None:
        raise AssertionError('adatas is None')

    if torch.cuda.is_available(): # cuda device
        device='cuda'
        torch.cuda.set_device(gpu)
    else:
        device='cpu'
    
    print('Device:', device)

    outdir = outdir+'/'
    os.makedirs(outdir+'/checkpoint', exist_ok=True)
    log = create_logger('', fh=outdir+'log.txt')

    n_domain = len(adatas)

    tran = {}
    num_cell = []
    num_gene = []

    for i, adata in enumerate(adatas):
        num_cell.append(adata.X.shape[0])
        num_gene.append(adata.X.shape[1])

    num_cell_copy = [i for i in num_cell]
    # training
    if out == 'latent':
        print(f'####function.py#1522rows, out={out},START Training......')
        for i, adata in enumerate(adatas):
            print(adata)
        print('\n')

        if mmd_match and optimal_transmission:
            print(f'####function.py##, Both MMD and OT')
        elif not mmd_match and optimal_transmission:
            print(f'####function.py##, Only OT')
        elif mmd_match and not optimal_transmission:
            print(f'####function.py##, Only MMD')
        
        if sampler=='smote': 
            print(f'####function.py##, sampler={sampler}')
            if unshared_encoder: #TODOq 分开使用两个编码器， unshared_encoder=True
                Ctrainloader, Ptrainloader, testloader = load_data_smote_unshared_encoder(
                    num_cell_copy,
                    adatas=adatas,
                    source_batch=source_batch,
                    target_batch=target_batch,
                    drop_last=False,
                    shuffle=True,
                    num_workers=num_workers,
                    over_sampling_strategy=over_sampling_strategy,
                    under_sampling_strategy=under_sampling_strategy)
            else:            
                Ctrainloader, Ptrainloader, testloader = load_data_smote(
                    num_cell_copy,
                    adatas=adatas,
                    source_batch=source_batch,
                    target_batch=target_batch,
                    drop_last=False,
                    shuffle=True,
                    num_workers=num_workers,
                    over_sampling_strategy=over_sampling_strategy,
                    under_sampling_strategy=under_sampling_strategy)
        elif sampler=='weight':
            print(f'####function.py##, sampler={sampler}')
            if unshared_encoder: #TODO1 分开使用两个编码器， unshared_encoder=True
                Ctrainloader, Ptrainloader, testloader = load_data_weight_unshared_encoder(
                    adatas=adatas,
                    source_batch=source_batch,
                    target_batch=target_batch,
                    drop_last=False,
                    shuffle=True,
                    num_workers=num_workers)
            else:            
                Ctrainloader, Ptrainloader, testloader = load_data_weight(
                    adatas=adatas,
                    source_batch=source_batch,
                    target_batch=target_batch,
                    drop_last=False,
                    shuffle=True,
                    num_workers=num_workers)
        else:
            print(f'####function.py##,sampler={sampler}')
            if unshared_encoder: #TODO1 分开使用两个编码器， unshared_encoder=True
                trainloader, testloader = load_data_unshared_encoder(
                    num_cell=num_cell,
                    adata_cm=adata_cm,
                    domain_name=batch_key,
                    batch_size=batch_size, 
                    num_workers=num_workers,
                    )
            else:
                trainloader, testloader = load_data(
                    num_cell=num_cell,
                    adata_cm=adata_cm,
                    domain_name=batch_key,
                    batch_size=batch_size, 
                    num_workers=num_workers, 
                    )

        if save_OT:
            if sampler == 'smote':
                num_cell_tmp = num_cell_copy
            else:
                num_cell_tmp = num_cell
            for i in range(n_domain):
                if i != ref_id:
                    ns = num_cell_tmp[i]
                    nt = num_cell_tmp[ref_id]
                    tran_tmp = np.ones((ns, nt)) / (ns * nt)
                    tran[i] = tran_tmp.astype(np.float32)
                    print('Size of transport plan between datasets {} and {}:'.format(i, ref_id), np.shape(tran[i]))
        
        # encoder structure
        if enc is None:
            enc = {} #TODO1,new feature, enc = {0:[...], 1:[...], 2:[...], 4:[...], 5:[...]}
            enc[0]=[]
            enc[1]=[]
            enc[2]=[]
            for index,i in enumerate(encoder_h_dims): #TODO1 shared encoder 0
                if index == 0:
                    enc[0].append(['fc', i, 1, 'relu', drop])
                elif index == (len(encoder_h_dims)-1):
                    enc[0].append(['fc', i, '', '', 0])
                else:
                    enc[0].append(['fc', i, 1, 'relu', drop])
            enc[1].append(['fc', 2000, 1, 'relu', drop])
            enc[1].append(['fc', 16, 1, '', 0])
            enc[2].append(['fc', 2000, 1, 'relu', drop])
            enc[2].append(['fc', 16, 1, '', 0])
            if unshared_encoder: #TODO1 set two data-specific encoders
                enc[4]=[]
                enc[5]=[]
                for index,i in enumerate(encoder_h_dims_source): #TODO1 bulk data-specific encoder
                    if index == 0:
                        enc[4].append(['fc', i, 1, 'relu', drop])
                    elif index == (len(encoder_h_dims_source)-1):
                        enc[4].append(['fc', i, '', '', 0])
                    else:
                        enc[4].append(['fc', i, 1, 'relu', drop])
                for index,i in enumerate(encoder_h_dims_target): #TODO1 sc data-specific encoder
                    if index == 0:
                        enc[5].append(['fc', i, 1, 'relu', drop])
                    elif index == (len(encoder_h_dims_target)-1):
                        enc[5].append(['fc', i, '', '', 0])
                    else:
                        enc[5].append(['fc', i, 1, 'relu', drop])
        
        # decoder structure
        dec = {} 
        num_gene.append(adata_cm.X.shape[1])
        # 1.shared decoder
        encoder_h_dims.pop(-1)
        encoder_h_dims.reverse()
        encoder_h_dims.append(num_gene[n_domain])
        dec[0]=[]
        for index,i in enumerate(encoder_h_dims):
            if index == (len(encoder_h_dims)-1):
                dec[0].append(['fc', i, n_domain, 'sigmoid', 0])
            else:
                dec[0].append(['fc', i, 1, 'relu', drop])
        for i in range(1, n_domain+1):
            dec[i] = [['fc', 2000, 1, 'sigmoid']]   # 不使用特异性基因的话，为了填补decoder的结构，设置个默认的，不参与模型训练
        if drug_response:
            if DRUG!='Simulated_Drug':
                for i in range(n_domain+1,n_domain+2):
                    dec[i] = [['drug_response', 128, 1, 'relu', drop],
                             ['drug_response', 128, 1, 'relu', drop],
                             ['drug_response', 128, 1, 'relu', drop],
                             ['drug_response', 128, 1, 'relu', drop],
                             ['drug_response', 64, 1, 'relu', drop],
                             ['drug_response', 1, 3, 'sigmoid', 0]] # 使用bulk细胞系数据训练药物响应预测模型, [model_name='drug_response', out_dim=1, norm=1, activation='sigmoid', droupt=0]
            else:
                for i in range(n_domain+1,n_domain+2):
                    dec[i] = [['drug_response', 256, 1, 'relu', drop],
                              ['drug_response', 256, 1, 'relu', drop],
                              ['drug_response', 1, 3, 'sigmoid', 0]] # 使用bulk细胞系数据训练药物响应预测模型, [model_name='drug_response', out_dim=1, norm=1, activation='sigmoid', droupt=0]

        if unshared_decoder: #set two data-specfic decoders，structure of the decoder is the opposite of the corresponding encoder
            for i in range(n_domain+2, n_domain+4):
                if i==(n_domain+2): #bulk data，set bulk data decoder
                    encoder_h_dims_source.pop(-1)
                    encoder_h_dims_source.reverse()
                    encoder_h_dims_source.append(num_gene[0])
                    dec[n_domain+2]=[]
                    for index,i in enumerate(encoder_h_dims_source):
                        if index == (len(encoder_h_dims_source)-1):
                            dec[n_domain+2].append(['fc', i, 1, 'sigmoid', 0]) #TODO1, [model_name='drug_response', out_dim=1, norm=1, activation='sigmoid', droupt=0]
                        else:
                            dec[n_domain+2].append(['fc', i, 1, 'relu', drop])
                else: #sc data，set sc data decoder
                    encoder_h_dims_target.pop(-1)
                    encoder_h_dims_target.reverse()
                    encoder_h_dims_target.append(num_gene[1])
                    dec[n_domain+3]=[]
                    for index,i in enumerate(encoder_h_dims_target):
                        if index == (len(encoder_h_dims_target)-1):
                            dec[n_domain+3].append(['fc', i, 1, 'sigmoid', 0]) #TODO1, [model_name='drug_response', out_dim=1, norm=1, activation='sigmoid', droupt=0]
                        else:
                            dec[n_domain+3].append(['fc', i, 1, 'relu', drop])

        # init model
        model = VAE(enc, dec, ref_id=ref_id, n_domain=n_domain,
                    batch_size=batch_size,
                    lambda_recon=lambda_recon,
                    lambda_kl=lambda_kl,
                    lambda_ot=lambda_ot,
                    lambda_response=lambda_response)
        if model_info:
            log.info('model\n'+model.__repr__())
        if cell_regularization:
            print(f'####function.py###, use cell regularization')
        if unshared_decoder: #use two data-specfic decoders, unshared_decoder=True
            if unshared_encoder: #use two data-specfic encoders， unshared_encoder=True
                print(f'####function.py##, use two data-specfic encoders，two data-specfic decoders')
                if sampler=='smote' or sampler=='weight':
                    model.fit2_unshared_encoder_decoder(
                        Ctrainloader,
                        Ptrainloader,
                        tran,
                        num_cell=num_cell_copy,
                        save_OT=save_OT,
                        lambda_recon=lambda_recon,
                        lambda_kl=lambda_kl,
                        lambda_ot=lambda_ot,
                        lambda_response=lambda_response,
                        lambda_cell=lambda_cell,
                        reg=reg,
                        reg_m=reg_m,
                        lr=lr,
                        device=device,
                        verbose=verbose,
                        loss_type=loss_type,
                        drug_response=drug_response,
                        cell_regularization=cell_regularization,
                        adata_cm=adata_cm,
                        adatas=adatas, 
                        unshared_decoder=unshared_decoder,
                        n_epoch=n_epoch
                    )
                else:
                    model.fit2_1_unshared_encoder_decoder(
                        trainloader,
                        tran,
                        num_cell,
                        num_gene,
                        save_OT=save_OT,
                        lambda_recon=lambda_recon,
                        lambda_kl=lambda_kl,
                        lambda_ot=lambda_ot,
                        lambda_response=lambda_response,
                        lambda_cell=lambda_cell,
                        reg=reg,
                        reg_m=reg_m,
                        lr=lr,
                        device=device,
                        verbose=verbose,
                        loss_type=loss_type,
                        drug_response=drug_response,
                        cell_regularization=cell_regularization,
                        adata_cm=adata_cm,
                        adatas=adatas,
                        n_epoch=n_epoch
                    )
            else:
                print(f'####function.py##, use shared encoder, and two data-specfic decoder')
                if sampler=='smote' or sampler=='weight':
                    model.fit2_unshared_decoder(
                        Ctrainloader,
                        Ptrainloader,
                        tran,
                        num_cell=num_cell_copy,
                        save_OT=save_OT,
                        lambda_recon=lambda_recon,
                        lambda_kl=lambda_kl,
                        lambda_ot=lambda_ot,
                        lambda_response=lambda_response,
                        lambda_cell=lambda_cell,
                        reg=reg,
                        reg_m=reg_m,
                        lr=lr,
                        device=device,
                        verbose=verbose,
                        loss_type=loss_type,
                        drug_response=drug_response,
                        cell_regularization=cell_regularization,
                        adata_cm=adata_cm,
                        adatas=adatas, 
                        unshared_decoder=unshared_decoder,
                        n_epoch=n_epoch,
                        mmd_match=mmd_match,
                        mmd_GAMMA=mmd_GAMMA,
                        lambda_mmd=lambda_mmd,
                        optimal_transmission=optimal_transmission,
                    )
                else:
                    model.fit2_1_unshared_decoder( # use shared encoder, and two data-specific decoder
                        trainloader,
                        tran,
                        num_cell,
                        num_gene,
                        save_OT=save_OT,
                        lambda_recon=lambda_recon,
                        lambda_kl=lambda_kl,
                        lambda_ot=lambda_ot,
                        lambda_response=lambda_response,
                        lambda_cell=lambda_cell,
                        reg=reg,
                        reg_m=reg_m,
                        lr=lr,
                        device=device,
                        verbose=verbose,
                        loss_type=loss_type,
                        drug_response=drug_response,
                        cell_regularization=cell_regularization,
                        adata_cm=adata_cm,
                        n_epoch=n_epoch,
                        mmd_match=mmd_match,
                        mmd_GAMMA=mmd_GAMMA,
                        lambda_mmd=lambda_mmd,
                        optimal_transmission=optimal_transmission,
                    )
        else: #use shared encoder and shared decoder, unshared_decoder=False
            print(f'####function.py###, use shared encoder and shared decoder')
            if sampler=='smote' or sampler=='weight':
                model.fit2( # use shared encoder and shared decoder
                    Ctrainloader,
                    Ptrainloader,
                    tran,
                    num_cell=num_cell_copy,
                    save_OT=save_OT,
                    lambda_recon=lambda_recon,
                    lambda_kl=lambda_kl,
                    lambda_ot=lambda_ot,
                    lambda_response=lambda_response,
                    lambda_cell=lambda_cell, 
                    reg=reg,
                    reg_m=reg_m,
                    lr=lr,
                    device=device,
                    verbose=verbose,
                    loss_type=loss_type,
                    drug_response=drug_response,
                    cell_regularization=cell_regularization, 
                    adata_cm=adata_cm,
                    n_epoch=n_epoch,
                    mmd_match=mmd_match,
                    mmd_GAMMA=mmd_GAMMA, 
                    lambda_mmd=lambda_mmd, 
                )
            else:
                model.fit2_1( # use shared encoder and shared decoder
                    trainloader,
                    tran,
                    num_cell,
                    num_gene,
                    save_OT=save_OT,
                    lambda_recon=lambda_recon,
                    lambda_kl=lambda_kl,
                    lambda_ot=lambda_ot,
                    lambda_response=lambda_response,
                    lambda_cell=lambda_cell,
                    reg=reg,
                    reg_m=reg_m,
                    lr=lr,
                    device=device,
                    verbose=verbose,
                    loss_type=loss_type,
                    drug_response=drug_response,
                    cell_regularization=cell_regularization, 
                    adata_cm=adata_cm,
                    n_epoch=n_epoch,
                    mmd_match=mmd_match,
                    mmd_GAMMA=mmd_GAMMA, 
                    lambda_mmd=lambda_mmd, 
                )
        torch.save({'enc':enc, 'dec':dec, 'n_domain':n_domain, 'ref_id':ref_id, 'num_gene':num_gene, 'batch_size':batch_size,'lambda_recon':lambda_recon, 'lambda_kl':lambda_kl, 'lambda_ot':lambda_ot, 'lambda_response':lambda_response, 'sampler':sampler, 'unshared_decoder':unshared_decoder, 'unshared_encoder':unshared_encoder, 'mmd_match':mmd_match, 'cell_regularization':cell_regularization}, outdir+'/checkpoint/'+DRUG+'_config.pt')
        torch.save(model.state_dict(), outdir+'/checkpoint/'+DRUG+'_model.pt')
    # predict
    else:
        print(f'####function.py##, out={out},load model with checkpoint')
        state = torch.load(outdir+'/checkpoint/'+DRUG+'_config.pt')
        enc, dec, n_domain, ref_id, num_gene, batch_size, lambda_recon, lambda_kl, lambda_ot, lambda_response, sampler, unshared_decoder, unshared_encoder, mmd_match, cell_regularization = state['enc'], state['dec'], state['n_domain'], state['ref_id'], state['num_gene'], state['batch_size'], state['lambda_recon'], state['lambda_kl'], state['lambda_ot'], state['lambda_response'], state['sampler'], state['unshared_decoder'], state['unshared_encoder'], state['mmd_match'], state['cell_regularization']
        model = VAE(enc, dec, ref_id=ref_id, n_domain=n_domain,batch_size=batch_size,lambda_recon=lambda_recon,lambda_kl=lambda_kl,lambda_ot=lambda_ot,lambda_response=lambda_response)
        model.load_model(outdir+'/checkpoint/'+DRUG+'_model.pt')
        model.to(device)
        if mmd_match:
            print(f'####function.py###, both use MMD+OT')
        else:
            print(f'####function.py##, only use OT')
        if cell_regularization:
            print(f'####function.py##, use cell regularization ')
        if unshared_decoder: #use two data-specific decoder
            if unshared_encoder: #use two data-specific encoder
                print(f'####function.py##, use two data-specific encoder and decoder')
            else:
                print(f'####function.py##, use shared encoder and two data-specific decoders')
        else:
            print(f'####function.py##, use shared encoder and decoder')

        if sampler=='smote':
            print(f'####function.py##, sampler={sampler}')
            if unshared_encoder: #use two data-specific encoder
                Ctrainloader, Ptrainloader, testloader = load_data_smote_unshared_encoder(
                    num_cell_copy,
                    adatas=adatas,
                    source_batch=source_batch,
                    target_batch=target_batch,
                    drop_last=False,
                    shuffle=True,
                    num_workers=num_workers,
                    over_sampling_strategy=over_sampling_strategy,
                    under_sampling_strategy=under_sampling_strategy)
            else:
                Ctrainloader, Ptrainloader, testloader = load_data_smote(
                    num_cell_copy,
                    adatas=adatas,
                    source_batch=source_batch,
                    target_batch=target_batch,
                    drop_last=False,
                    shuffle=True,
                    num_workers=num_workers,
                    over_sampling_strategy=over_sampling_strategy,
                    under_sampling_strategy=under_sampling_strategy)
        elif sampler=='weight':
            print(f'####function.py##, sampler={sampler}')
            if unshared_encoder: #use two data-specific encoder
                Ctrainloader, Ptrainloader, testloader = load_data_weight_unshared_encoder(
                    adatas=adatas,
                    source_batch=source_batch,
                    target_batch=target_batch,
                    drop_last=False,
                    shuffle=True,
                    num_workers=num_workers)
            else:
                Ctrainloader, Ptrainloader, testloader = load_data_weight(
                    adatas=adatas,
                    source_batch=source_batch,
                    target_batch=target_batch,
                    drop_last=False,
                    shuffle=True,
                    num_workers=num_workers)
        else:
            print(f'####function.py##, sampler={sampler}')
            trainloader, testloader = load_data_unshared_encoder(
                num_cell=num_cell,
                adata_cm=adata_cm,
                domain_name=batch_key,
                batch_size=batch_size,
                num_workers=num_workers
            )
            if unshared_encoder: #use two data-specific encoder
                pass
            else:
                trainloader, testloader = load_data(
                    num_cell=num_cell,
                    adata_cm=adata_cm,
                    domain_name=batch_key,
                    batch_size=batch_size,
                    num_workers=num_workers
                )
    
    if out == 'latent':
        if sampler=='smote' or sampler=='weight':
            model.encodeBatch(adata_cm, adatas, dataloader=testloader, num_cell=num_cell, num_gene=num_gene, device=device, out=out, DRUG=DRUG,source_batch=source_batch,target_batch=target_batch,sampler=sampler, unshared_encoder=unshared_encoder) # save latent rep
        else:
            model.encodeBatch_1(adata_cm, adatas, dataloader=testloader, num_gene=num_gene, num_cell=num_cell,device=device,out=out, DRUG=DRUG,source_batch=source_batch,target_batch=target_batch,sampler=sampler, unshared_encoder=unshared_encoder) # save latent rep
    elif out=='predict':
        if sampler=='smote' or sampler=='weight':
            model.encodeBatch(adata_cm, adatas, dataloader=testloader, num_cell=num_cell, num_gene=num_gene, device=device,  out=out,DRUG=DRUG,source_batch=source_batch,target_batch=target_batch,sampler=sampler, unshared_encoder=unshared_encoder) # save latent rep
        else:
            model.encodeBatch_1(adata_cm, adatas, dataloader=testloader, num_gene=num_gene, num_cell=num_cell,device=device, out=out,  DRUG=DRUG,source_batch=source_batch,target_batch=target_batch,sampler=sampler, unshared_encoder=unshared_encoder) # save latent rep
    
    #IntegratedGradients
    if printgene:
        print(f'####function.py##, study critical genes')
        target_model = TargetModel(model.decoder,model.encoder)
        ig = IntegratedGradients(target_model)
        x_tar = torch.FloatTensor(adatas[1].X).to(device)
        scattr, delta =  ig.attribute(x_tar,target=0, return_convergence_delta=True,internal_batch_size=x_tar.shape[0])
        scattr = scattr.detach().cpu().numpy()
        # Save integrated gradient
        igadata= sc.AnnData(scattr)
        igadata.var.index = adatas[1].var.index
        igadata.obs.index = adatas[1].obs.index
        sc_gra = "./drug/"+str(DRUG)+ "/" + DRUG +"sc_gradient.txt"
        sc_gen = "./drug/"+str(DRUG)+ "/" + DRUG +"sc_gene.csv"
        sc_lab = "./drug/"+str(DRUG)+ "/" + DRUG +"sc_label.csv"
        np.savetxt(sc_gra,scattr,delimiter = " ")
        pd.DataFrame(adatas[1].var.index).to_csv(sc_gen)
        pd.DataFrame(adatas[1].obs["response"]).to_csv(sc_lab)
    
    torch.cuda.empty_cache()
    if save_OT and out == 'latent':
        print(f'####funtion.py##,save OT matrix')
        return adata_cm, tran
    return adata_cm
