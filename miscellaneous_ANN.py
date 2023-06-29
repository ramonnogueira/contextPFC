import torch
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import scipy
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
import itertools
nan=float('nan')


def create_input(n_trials,t_steps,coh_uq,input_noise,scale_ctx,ctx_noise):
    dic={}
    input_vec_pre=nan*np.zeros((n_trials*len(coh_uq),t_steps,2))
    target_vec_pre=nan*np.zeros(n_trials*len(coh_uq))
    coherence_pre=nan*np.zeros(n_trials*len(coh_uq))

    # Input is random noise with different means (coherence). Last dimension is context, same for all time steps and +1 -1. 
    # Target is 1 when mean is positive and 0 otherwise
    # Create Stimulus
    for i in range(len(coh_uq)):
        input_vec_pre[i*n_trials:(i+1)*n_trials,:,0]=np.random.normal(loc=coh_uq[i],scale=input_noise,size=(n_trials,t_steps))
        target_vec_pre[i*n_trials:(i+1)*n_trials]=np.sign(coh_uq[i])
        coherence_pre[i*n_trials:(i+1)*n_trials]=coh_uq[i]
    try:
        target_vec_pre[target_vec_pre==0]=np.sign(np.random.normal(loc=0,scale=1,size=(n_trials)))
    except:
        print ('No 0 coherence')
    target_vec_pre[target_vec_pre==-1]=0
    # Add context
    context_pre=scale_ctx*np.sign(np.random.normal(loc=0,scale=1,size=(len(coh_uq)*n_trials)))
    context_noise=(context_pre+np.random.normal(loc=0,scale=ctx_noise,size=len(coh_uq)*n_trials))
    #context[context==-1]=0
    for i in range(t_steps):
        input_vec_pre[:,i,1]=context_noise
    
    # Shuffle indeces
    index_def=np.random.permutation(np.arange(len(coh_uq)*n_trials))
    #index_def=np.arange(len(coh_uq)*n_trials)
    input_vec=input_vec_pre[index_def]
    target_vec=target_vec_pre[index_def]
    coherence=coherence_pre[index_def]
    context=context_pre[index_def]

    # The motion direction at t=0 is always 0 to simulate the latency
    input_vec[:,0,0]=0
    
    # Return input and target in torch format
    dic['input_rec']=Variable(torch.from_numpy(np.array(input_vec,dtype=np.float32)),requires_grad=False)
    dic['target_vec']=Variable(torch.from_numpy(np.array(target_vec,dtype=np.int16)),requires_grad=False)
    dic['coherence']=coherence
    dic['context']=context
    return dic

