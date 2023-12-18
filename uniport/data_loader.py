#!/usr/bin/env 
"""
# Author: Kai Cao
# Modified from SCALEX
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler
from torch.utils.data import DataLoader
from torch.utils.data.dataset import ConcatDataset
import scanpy as sc
from scipy.sparse import issparse
import scipy


class BatchSampler(Sampler):
    """
    Batch-specific Sampler
    sampled data of each batch is from the same dataset.
    """
    def __init__(self, batch_size, batch_id, drop_last=False):
        """
        create a BatchSampler object
        
        Parameters
        ----------
        batch_size
            batch size for each sampling
        batch_id
            batch id of samples
        drop_last
            drop the last samples that not up to one batch        
        """
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.batch_id = batch_id

    def __iter__(self):#数据集x的取样成一个batch，数据集y的取样成一个batch
        batch = {}
        sampler = np.random.permutation(len(self.batch_id))
        for idx in sampler:
            c = self.batch_id[idx]
            if c not in batch:
                batch[c] = []
            batch[c].append(idx)

            if len(batch[c]) == self.batch_size:
                yield batch[c]
                batch[c] = []

        for c in batch.keys():
            if len(batch[c]) > 0 and not self.drop_last:
                yield batch[c]
            
    def __len__(self):
        if self.drop_last:
            return len(self.batch_id) // self.batch_size
        else:
            return (len(self.batch_id)+self.batch_size-1) // self.batch_size


class BatchSampler_balance(Sampler):
    """
    Balanced batch-specific Sampler， 实现了一个平衡的批次采样器，通过生成平衡的批次样本来提高模型训练的效果。
    """
    def __init__(self, batch_size, num_cell, batch_id, drop_last=False):
        """
        create a Balanced BatchSampler object
        
        Parameters
        ----------
        batch_size
            batch size for each sampling
        number_cell
            number of cells for each cell types
        batch_id
            batch id of all samples
        drop_last
            drop the last samples that not up to one batch
        """
        self.batch_size = batch_size
        self.num_cell = num_cell
        self.drop_last = drop_last
        self.batch_id = batch_id

    def __iter__(self):
        batch = {}
        max_num = max(self.num_cell)
        balance_list = []
        for i in self.num_cell:
            if i < 0.02*max_num:
                balance_list.append(self.num_cell.index(i))

        sampler = np.random.permutation(len(self.batch_id))
        sample_idx = []
        balance_idx = {}

        for i in balance_list:
            balance_idx[i] = []
            for j in sampler:
                if self.batch_id[j]==i:
                    balance_idx[i].append(j)

        for idx in sampler:

            sample_idx.append(idx)
            if len(sample_idx) == self.batch_size:

                for key in balance_idx:
                    balance_idx[key] = np.random.choice(balance_idx[key], min(self.num_cell[key], int(0.02*self.batch_size)))
                    sample_idx.extend(balance_idx[key])
                yield sample_idx
                sample_idx = []

        if len(sample_idx) > 0 and not self.drop_last:
            yield sample_idx
            
    def __len__(self):
        if self.drop_last:
            return len(self.batch_id) // self.batch_size
        else:
            return (len(self.batch_id)+self.batch_size-1) // self.batch_size

    
class SingleCellDataset(Dataset):

    def __init__(self, data, batch):
        
        self.data = data
        self.batch = batch
        self.shape = data.shape
        
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, idx):

        domain_id = self.batch[idx]
        # x = self.data[idx].toarray().squeeze()#TODO 修改
        x = self.data[idx]#TODO 修改

        return x, domain_id, idx#TODO,@see:vae.py,415rows

class SingleCellDataset_vertical(Dataset):

    def __init__(self, adatas):

        self.adatas = adatas
        
    def __len__(self):
        return self.adatas[0].shape[0]
    
    def __getitem__(self, idx):

        x = self.adatas[0].X[idx].toarray().squeeze()

        for i in range(1, len(self.adatas)):

            x = np.concatenate((x, self.adatas[i].X[idx].toarray().squeeze()))

        return x, idx
    
def load_data(adatas, mode='h', use_rep=['X','X'], num_cell=None, max_gene=None, adata_cm=None, use_specific=False, domain_name='domain_id', batch_size=256, drop_last=True, shuffle=True, num_workers=4):

    '''
    Load data for training.

    Parameters
    ----------
    adatas
        A list of AnnData matrice.
    mode
        training mode. Choose between ['h', 'd', 'v'].
    use_rep
        use '.X' or '.obsm'.
    num_cell
        numbers of cells of each adata in adatas.
    max_gene
        maximum number of genes of each adata in adatas.
    adata_cm
        adata with common genes of adatas.
    use_specific
        use dataset-specific genes.
    domain_name
        domain name of each adata in adatas.
    batch_size
        size of each mini batch for training.
    drop_last
        drop the last samples that not up to one batch.
    shuffle
        shuffle the data
    num_workers
        number parallel load processes according to cpu cores.

    Returns
    -------
    trainloader
        data loader for training
    testloader
        data loader for testing
    '''

    if mode == 'd':

        for i, adata in enumerate(adatas):

            if use_rep[i]=='X':
                tmp = adata.X
            else:
                tmp = adata.obsm[use_rep[i]]
            if tmp.shape[1] < max_gene:
                tmp =  scipy.sparse.hstack((tmp, scipy.sparse.coo_matrix(np.zeros((tmp.shape[0], max_gene-tmp.shape[1])))))
            if i == 0:
                x = tmp
                batches = adata.obs[domain_name].astype(int).tolist()
            else:
                x = scipy.sparse.vstack((x, tmp))
                batches.extend(adata.obs[domain_name])
        
        x = x.tocsr()

        scdata = SingleCellDataset(x, batches)
    
    elif mode == 'h':
        # batches = int(adata_cm.obs[domain_name]).tolist()
        batches = adata_cm.obs[domain_name].cat.categories.tolist()

        if use_specific:

            for i, adata in enumerate(adatas):

                adata_tmp = adata_cm[adata_cm.obs[domain_name]==batches[i],]

                x_c = adata_tmp.X #对应domain_name的共同基因的高表达基因

                x_s = adata.X #特异性高表达基因

                if x_s.shape[1] < max_gene:

                    x_s = scipy.sparse.hstack((x_s, scipy.sparse.coo_matrix(np.zeros((x_s.shape[0], max_gene-x_s.shape[1])))))

                if i == 0:
                    x = scipy.sparse.hstack((x_c, x_s))
                else:
                    x = scipy.sparse.vstack((x, scipy.sparse.hstack((x_c, x_s))))
                #TODO，看一下x的shape，x.shape=(11259, 4000)，x.shape=(22518, 4000)
                #print(f'###data_loader.py#249rows#################x.shape={x.shape}')
                x = x.tocsr()

        else:
            #if not issparse(adata_cm.X):#TODO 修改
            #    adata_cm.X = scipy.sparse.csr_matrix(adata_cm.X)#TODO 修改
            x = adata_cm.X
            
        #TODO,adata_cm.obs[domain_name].astype(int).tolist(),里面都是[0,1]
        #print(f'###data_loader.py#258rows#################len(adata_cm.obs[domain_name].astype(int).tolist())==={len(adata_cm.obs[domain_name].astype(int).tolist())}')
        scdata = SingleCellDataset(x, adata_cm.obs[domain_name].astype(int).tolist())
        #TODO,scdata是什么？,scdata.shape=(22518, 4000)
        #print(f'###data_loader.py#261rows#################scdata.shape={scdata.shape}')
    else:
        scdata = SingleCellDataset_vertical(adatas)


    # scdata = SingleCellDataset(adata_cm.X, adata_cm.obs['batch'].cat.codes.values.astype(int), mode='batchcorrect')

    if min(num_cell)<0.02*max(num_cell):  # if samples in one domain is too less, use the balanced batch sampler.

        balance_sampler = BatchSampler_balance(batch_size, num_cell, adata_cm.obs[domain_name].astype(int).tolist(), drop_last=True)
        trainloader = DataLoader(
            scdata, 
            batch_sampler=balance_sampler,
            num_workers=num_workers,
        )

    else:
        trainloader = DataLoader(
            scdata, 
            batch_size=batch_size, 
            drop_last=drop_last, # default true
            shuffle=shuffle, # default true
            num_workers=num_workers,
        )

    '''if mode == 'h':
        batch_sampler = BatchSampler(batch_size, adata_cm.obs[domain_name], drop_last=False)
        testloader = DataLoader(scdata, batch_sampler=batch_sampler) #testloader.dataset.shape=(22518, 4000)
        #TODO,testloader？,testloader.dataset.shape=(22518, 4000)
        #print(f'###data_loader.py#290rows#################testloader.dataset.shape={testloader.dataset.shape}')
    else:
        testloader = DataLoader(scdata, batch_size=batch_size, drop_last=False, shuffle=False)'''
    #TODO new feature
    testloader = DataLoader(scdata, batch_size=adata_cm.X.shape[0], drop_last=False, shuffle=False)
    return trainloader, testloader

def load_data_unshared_encoder(adatas, mode='h', use_rep=['X','X'], num_cell=None, max_gene=None, adata_cm=None, domain_name='domain_id', batch_size=256, drop_last=True, shuffle=True, num_workers=4):

    '''
    Load data for training.

    Parameters
    ----------
    adatas
        A list of AnnData matrice.
    mode
        training mode. Choose between ['h', 'd', 'v'].
    use_rep
        use '.X' or '.obsm'.
    num_cell
        numbers of cells of each adata in adatas.
    max_gene
        maximum number of genes of each adata in adatas.
    adata_cm
        adata with common genes of adatas.
    use_specific
        use dataset-specific genes.
    domain_name
        domain name of each adata in adatas.
    batch_size
        size of each mini batch for training.
    drop_last
        drop the last samples that not up to one batch.
    shuffle
        shuffle the data
    num_workers
        number parallel load processes according to cpu cores.

    Returns
    -------
    trainloader
        data loader for training
    testloader
        data loader for testing
    '''

    if mode == 'h':
        # batches = int(adata_cm.obs[domain_name]).tolist()
        batches = adata_cm.obs[domain_name].cat.categories.tolist()

        #if not issparse(adata_cm.X):#TODO 修改
        #    adata_cm.X = scipy.sparse.csr_matrix(adata_cm.X)#TODO 修改
        x = adata_cm.X
            
        
        scdata = SingleCellDataset(x, adata_cm.obs[domain_name].astype(int).tolist())


    if min(num_cell)<0.02*max(num_cell):  # if samples in one domain is too less, use the balanced batch sampler.
        balance_sampler = BatchSampler_balance(batch_size, num_cell, adata_cm.obs[domain_name].astype(int).tolist(), drop_last=True)
        trainloader = DataLoader(
            scdata, 
            batch_sampler=balance_sampler,
            num_workers=num_workers,
        )
    else:
        trainloader = DataLoader(
            scdata, 
            batch_size=batch_size, 
            drop_last=drop_last, # default true
            shuffle=shuffle, # default true
            num_workers=num_workers,
        )

    #TODO new feature
    testloader = DataLoader(scdata, batch_size=adata_cm.X.shape[0], drop_last=False, shuffle=False)
    return trainloader, testloader


#TODO new feature
def load_data_smote(num_cell_copy, adatas, source_batch=190, target_batch=128, drop_last=False, shuffle=True, num_workers=0,over_sampling_strategy=0.5, under_sampling_strategy=0.5):

    '''
    Load data for training.

    Parameters
    ----------
    adatas
        A list of AnnData matrice.
    mode
        training mode. Choose between ['h', 'd', 'v'].
    use_rep
        use '.X' or '.obsm'.
    num_cell
        numbers of cells of each adata in adatas.
    max_gene
        maximum number of genes of each adata in adatas.
    adata_cm
        adata with common genes of adatas.
    use_specific
        use dataset-specific genes.
    domain_name
        domain name of each adata in adatas.
    batch_size
        size of each mini batch for training.
    drop_last
        drop the last samples that not up to one batch.
    shuffle
        shuffle the data
    num_workers
        number parallel load processes according to cpu cores.

    Returns
    -------
    trainloader
        data loader for training
    testloader
        data loader for testing
    '''
    ####use SMOTE
    #print(f'####data_loader.py#339row, 测试1')
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline
    from imblearn.under_sampling import RandomUnderSampler
    #print(f'####data_loader.py#343row, 测试2')
    over = SMOTE(sampling_strategy=over_sampling_strategy)
    under = RandomUnderSampler(sampling_strategy=under_sampling_strategy)
    steps = [('o', over), ('u', under)]
    pipeline = Pipeline(steps=steps)
    XTrainGDSC_smote, YTrainGDSC_smote = pipeline.fit_resample(adatas[0].X, adatas[0].obs['response'].astype(int).tolist())
    from collections import Counter
    print(f"####data_loader.py#351row, nagative V.S positive=={Counter(YTrainGDSC_smote)}")
    num_cell_copy[0] = len(YTrainGDSC_smote)
    print(f"####data_loader.py#353row, num_cell_copy=={num_cell_copy}")
    
    # 1. 加载bulk数据
    scdata_bulk_smote = SingleCellDataset(XTrainGDSC_smote, YTrainGDSC_smote)
    scdata_bulk_origin = SingleCellDataset(adatas[0].X, adatas[0].obs['response'].astype(int).tolist())
    Ctrainloader = DataLoader(scdata_bulk_smote, batch_size=source_batch, shuffle=shuffle,drop_last=drop_last,num_workers=num_workers)
    # 2. 加载sc数据
    scdata_sc = SingleCellDataset(adatas[1].X, adatas[1].obs['response'].astype(int).tolist())
    Ptrainloader = DataLoader(scdata_sc, batch_size=target_batch, shuffle=shuffle,drop_last=drop_last,num_workers=num_workers)
    # 3. 把bulk数据和sc数据压缩到一起,组成trainloader
    from itertools import cycle
    trainloader = zip(Ctrainloader,cycle(Ptrainloader))
    # 4. 合成testloader
    Ctrainloader_test = DataLoader(scdata_bulk_origin, batch_size=adatas[0].X.shape[0], shuffle=False,drop_last=False)
    Ptrainloader_test = DataLoader(scdata_sc, batch_size=adatas[1].X.shape[0], shuffle=False,drop_last=False)
    #testloader = zip(Ctrainloader_test,cycle(Ptrainloader_test))
    testloader = zip(Ctrainloader_test,Ptrainloader_test)
    return Ctrainloader, Ptrainloader, testloader

#TODO new feature
def load_data_smote_unshared_encoder(num_cell_copy, adatas, source_batch=190, target_batch=128, drop_last=False, shuffle=True, num_workers=0,over_sampling_strategy=0.5, under_sampling_strategy=0.5):

    '''
    Load data for training.

    Parameters
    ----------
    adatas
        A list of AnnData matrice.
    mode
        training mode. Choose between ['h', 'd', 'v'].
    use_rep
        use '.X' or '.obsm'.
    num_cell
        numbers of cells of each adata in adatas.
    max_gene
        maximum number of genes of each adata in adatas.
    adata_cm
        adata with common genes of adatas.
    use_specific
        use dataset-specific genes.
    domain_name
        domain name of each adata in adatas.
    batch_size
        size of each mini batch for training.
    drop_last
        drop the last samples that not up to one batch.
    shuffle
        shuffle the data
    num_workers
        number parallel load processes according to cpu cores.

    Returns
    -------
    trainloader
        data loader for training
    testloader
        data loader for testing
    '''
    ####use SMOTE
    #print(f'####data_loader.py#339row, 测试1')
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline
    from imblearn.under_sampling import RandomUnderSampler
    #print(f'####data_loader.py#343row, 测试2')
    over = SMOTE(sampling_strategy=over_sampling_strategy)
    under = RandomUnderSampler(sampling_strategy=under_sampling_strategy)
    steps = [('o', over), ('u', under)]
    pipeline = Pipeline(steps=steps)
    XTrainGDSC_smote, YTrainGDSC_smote = pipeline.fit_resample(adatas[0].X, adatas[0].obs['response'].astype(int).tolist())
    from collections import Counter
    print(f"####data_loader.py#351row, nagative V.S positive=={Counter(YTrainGDSC_smote)}")
    num_cell_copy[0] = len(YTrainGDSC_smote)
    print(f"####data_loader.py#353row, num_cell_copy=={num_cell_copy}")
    
    # 1. 加载bulk数据
    scdata_bulk_smote = SingleCellDataset(XTrainGDSC_smote, YTrainGDSC_smote)
    scdata_bulk_origin = SingleCellDataset(adatas[0].X, adatas[0].obs['response'].astype(int).tolist())
    Ctrainloader = DataLoader(scdata_bulk_smote, batch_size=source_batch, shuffle=shuffle,drop_last=drop_last,num_workers=num_workers)
    # 2. 加载sc数据
    scdata_sc = SingleCellDataset(adatas[1].X, adatas[1].obs['response'].astype(int).tolist())
    Ptrainloader = DataLoader(scdata_sc, batch_size=target_batch, shuffle=shuffle,drop_last=drop_last,num_workers=num_workers)
    # 3. 把bulk数据和sc数据压缩到一起,组成trainloader
    from itertools import cycle
    trainloader = zip(Ctrainloader,cycle(Ptrainloader))
    # 4. 合成testloader
    Ctrainloader_test = DataLoader(scdata_bulk_origin, batch_size=adatas[0].X.shape[0], shuffle=False,drop_last=False)
    Ptrainloader_test = DataLoader(scdata_sc, batch_size=adatas[1].X.shape[0], shuffle=False,drop_last=False)
    #testloader = zip(Ctrainloader_test,cycle(Ptrainloader_test))
    testloader = zip(Ctrainloader_test,Ptrainloader_test)
    return Ctrainloader, Ptrainloader, testloader



#TODO new feature
def load_data_weight(adatas, source_batch=190, target_batch=128, drop_last=False, shuffle=True, num_workers=4):

    '''
    Load data for training.

    Parameters
    ----------
    adatas
        A list of AnnData matrice.
    mode
        training mode. Choose between ['h', 'd', 'v'].
    use_rep
        use '.X' or '.obsm'.
    num_cell
        numbers of cells of each adata in adatas.
    max_gene
        maximum number of genes of each adata in adatas.
    adata_cm
        adata with common genes of adatas.
    use_specific
        use dataset-specific genes.
    domain_name
        domain name of each adata in adatas.
    batch_size
        size of each mini batch for training.
    drop_last
        drop the last samples that not up to one batch.
    shuffle
        shuffle the data
    num_workers
        number parallel load processes according to cpu cores.

    Returns
    -------
    trainloader
        data loader for training
    testloader
        data loader for testing
    '''
    ####use WEIGHT
    from collections import Counter
    Counter(adatas[0].obs['response'])[0]/len(adatas[0].obs['response'])
    Counter(adatas[0].obs['response'])[1]/len(adatas[0].obs['response'])
    class_sample_count = np.array([Counter(adatas[0].obs['response'])[0]/len(adatas[0].obs['response']),Counter(adatas[0].obs['response'])[1]/len(adatas[0].obs['response'])])
    weight = 1. / class_sample_count
    #upsampling source domain training set which is unbalanced##
    samples_weight = np.array([weight[t] for t in adatas[0].obs['response'].values])

    samples_weight = torch.from_numpy(samples_weight)
    samples_weight = samples_weight.reshape(-1) # Flatten out the weights so it's a 1-D tensor of weights
    from torch.utils.data.sampler import WeightedRandomSampler
    sampler = WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'), len(samples_weight), replacement=True)

    # 1. 加载bulk数据
    scdata_bulk = SingleCellDataset(adatas[0].X, adatas[0].obs['response'])
    Ctrainloader = DataLoader(scdata_bulk, batch_size=source_batch, shuffle=False,sampler=sampler,drop_last=drop_last,num_workers=num_workers)
    # 2. 加载sc数据
    scdata_sc = SingleCellDataset(adatas[1].X, adatas[1].obs['response'].astype(int).tolist())
    Ptrainloader = DataLoader(scdata_sc, batch_size=target_batch, shuffle=shuffle,drop_last=drop_last,num_workers=num_workers)
    # 3. 把bulk数据和sc数据压缩到一起,组成trainloader
    from itertools import cycle
    trainloader = zip(Ctrainloader,cycle(Ptrainloader))

    # 4. 合成testloader
    Ctrainloader_test = DataLoader(scdata_bulk, batch_size=adatas[0].X.shape[0], shuffle=False,drop_last=False)
    Ptrainloader_test = DataLoader(scdata_sc, batch_size=adatas[1].X.shape[0], shuffle=False,drop_last=False)
    #testloader = zip(Ctrainloader_test,cycle(Ptrainloader_test))
    testloader = zip(Ctrainloader_test,Ptrainloader_test)
    return Ctrainloader, Ptrainloader, testloader
    
    

#TODO new feature
def load_data_weight_unshared_encoder(adatas, source_batch=190, target_batch=128, drop_last=False, shuffle=True, num_workers=4):

    '''
    Load data for training.

    Parameters
    ----------
    adatas
        A list of AnnData matrice.
    mode
        training mode. Choose between ['h', 'd', 'v'].
    use_rep
        use '.X' or '.obsm'.
    num_cell
        numbers of cells of each adata in adatas.
    max_gene
        maximum number of genes of each adata in adatas.
    adata_cm
        adata with common genes of adatas.
    use_specific
        use dataset-specific genes.
    domain_name
        domain name of each adata in adatas.
    batch_size
        size of each mini batch for training.
    drop_last
        drop the last samples that not up to one batch.
    shuffle
        shuffle the data
    num_workers
        number parallel load processes according to cpu cores.

    Returns
    -------
    trainloader
        data loader for training
    testloader
        data loader for testing
    '''
    ####use WEIGHT
    from collections import Counter
    Counter(adatas[0].obs['response'])[0]/len(adatas[0].obs['response'])
    Counter(adatas[0].obs['response'])[1]/len(adatas[0].obs['response'])
    class_sample_count = np.array([Counter(adatas[0].obs['response'])[0]/len(adatas[0].obs['response']),Counter(adatas[0].obs['response'])[1]/len(adatas[0].obs['response'])])
    weight = 1. / class_sample_count
    #upsampling source domain training set which is unbalanced##
    samples_weight = np.array([weight[t] for t in adatas[0].obs['response'].values])

    samples_weight = torch.from_numpy(samples_weight)
    samples_weight = samples_weight.reshape(-1) # Flatten out the weights so it's a 1-D tensor of weights
    from torch.utils.data.sampler import WeightedRandomSampler
    sampler = WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'), len(samples_weight), replacement=True)

    # 1. 加载bulk数据
    scdata_bulk = SingleCellDataset(adatas[0].X, adatas[0].obs['response'])
    Ctrainloader = DataLoader(scdata_bulk, batch_size=source_batch, shuffle=False,sampler=sampler,drop_last=drop_last,num_workers=num_workers)
    # 2. 加载sc数据
    scdata_sc = SingleCellDataset(adatas[1].X, adatas[1].obs['response'].astype(int).tolist())
    Ptrainloader = DataLoader(scdata_sc, batch_size=target_batch, shuffle=shuffle,drop_last=drop_last,num_workers=num_workers)
    # 3. 把bulk数据和sc数据压缩到一起,组成trainloader
    from itertools import cycle
    trainloader = zip(Ctrainloader,cycle(Ptrainloader))

    # 4. 合成testloader
    Ctrainloader_test = DataLoader(scdata_bulk, batch_size=adatas[0].X.shape[0], shuffle=False,drop_last=False)
    Ptrainloader_test = DataLoader(scdata_sc, batch_size=adatas[1].X.shape[0], shuffle=False,drop_last=False)
    #testloader = zip(Ctrainloader_test,cycle(Ptrainloader_test))
    testloader = zip(Ctrainloader_test,Ptrainloader_test)
    return Ctrainloader, Ptrainloader, testloader
    
    
