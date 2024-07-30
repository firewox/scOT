
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


def Run(
        adatas=None,     
        adata_cm=None,   
        lambda_recon=1.0,
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
                    lambda_response=lambda_response,
                    drop=drop)
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
        torch.save({'enc':enc, 'dec':dec, 'n_domain':n_domain, 'ref_id':ref_id, 'num_gene':num_gene, 'batch_size':batch_size,'lambda_recon':lambda_recon, 'lambda_kl':lambda_kl, 'lambda_ot':lambda_ot, 'lambda_response':lambda_response, 'sampler':sampler, 'unshared_decoder':unshared_decoder, 'unshared_encoder':unshared_encoder, 'mmd_match':mmd_match, 'cell_regularization':cell_regularization, 'drop':drop}, outdir+'/checkpoint/'+DRUG+'_config.pt')
        torch.save(model.state_dict(), outdir+'/checkpoint/'+DRUG+'_model.pt')
    # predict
    else:
        print(f'####function.py##, out={out},load model with checkpoint')
        state = torch.load(outdir+'/checkpoint/'+DRUG+'_config.pt')
        enc, dec, n_domain, ref_id, num_gene, batch_size, lambda_recon, lambda_kl, lambda_ot, lambda_response, sampler, unshared_decoder, unshared_encoder, mmd_match, cell_regularization, drop = state['enc'], state['dec'], state['n_domain'], state['ref_id'], state['num_gene'], state['batch_size'], state['lambda_recon'], state['lambda_kl'], state['lambda_ot'], state['lambda_response'], state['sampler'], state['unshared_decoder'], state['unshared_encoder'], state['mmd_match'], state['cell_regularization'], state['drop']
        model = VAE(enc, dec, ref_id=ref_id, n_domain=n_domain,batch_size=batch_size,lambda_recon=lambda_recon,lambda_kl=lambda_kl,lambda_ot=lambda_ot,lambda_response=lambda_response,drop=drop)
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
        from captum.attr import IntegratedGradients
        print(f'####function.py##, study critical genes')
        target_model = TargetModel(model.decoder,model.encoder)
        ig = IntegratedGradients(target_model)
        x_tar = torch.FloatTensor(adatas[1].X).to(device)
        scattr, delta =  ig.attribute(x_tar, return_convergence_delta=True,internal_batch_size=x_tar.shape[0])
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
