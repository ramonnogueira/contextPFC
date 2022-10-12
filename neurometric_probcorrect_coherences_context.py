import os
import matplotlib.pylab as plt
import numpy as np
import scipy.io
import math
import sys
import tables
import pandas
import pickle as pkl
from scipy.stats import sem
from scipy.stats import pearsonr
from numpy.random import permutation
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.svm import LinearSVC
from scipy.stats import ortho_group 
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import KFold,StratifiedKFold,StratifiedShuffleSplit
from sklearn.neural_network import MLPClassifier
#from numba import jit
import miscellaneous

nan=float('nan')
minf=float('-inf')
pinf=float('inf')
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

# pre targets: context, prev choice, prev rew
# post dots: choice, context, diff
# pre-saccade: choice, context, diff?
# post-saccade: choice, context, reward

# target onset: 'targ_on', dots onset: 'dots_on', dots offset: 'dots_off', saccade: 'response_edf'
talig_vec=np.array(['targ_on','dots_on','response_edf'])
dic_time={} # same number of steps for all time locks
dic_time['targ_on']=np.array([1000,1000,500,200]) # time pre, time post, bin size, step size
dic_time['dots_on']=np.array([1000,1000,500,200])
dic_time['response_edf']=np.array([1000,1000,500,200])
steps=int((dic_time['dots_on'][0]+dic_time['dots_on'][1])/dic_time['dots_on'][3])# Careful here!
xx_dic={}
for i in range(len(talig_vec)):
    xx_dic[talig_vec[i]]=np.linspace(-dic_time[talig_vec[i]][0]/1000,dic_time[talig_vec[i]][1]/1000,steps) 

nt=200
n_rand=10
perc_tr=0.8
thres=0
reg=1

for k in range(len(monkeys)):
    print (monkeys[k])
    abs_path='/home/ramon/Dropbox/Esteki_Kiani/data/late/%s/'%(phase,monkeys[k]) 
    files=os.listdir(abs_path)
    
    # print ('  ',talig_vec[kkk])
    # for kk in range(len(quant_vec)):
    #     print ('    ',quant_vec[kk])
    #     pseudo=miscellaneous.pseudopopulation_2(abs_path,files,quant_vec[kk],talig_vec[kkk],dic_time[talig_vec[kkk]],steps,thres,nt,n_rand,perc_tr)
    #     pseudo_all=pseudo['pseudo_all']
    #     pseudo_tr=pseudo['pseudo_tr']
    #     pseudo_te=pseudo['pseudo_te']
    #     clase_all=pseudo['clase_all']
    #     for i in range(steps):
    #             for ii in range(n_rand):
    #                 dim[kkk,kk,i,ii]=miscellaneous.dim_pseudo_2(pseudo_tr[i,ii],pseudo_te[i,ii],clase_all,reg)
    #                 abstr[kkk,kk,i,ii]=miscellaneous.abstraction_2D(pseudo_all[i,ii],clase_all,reg)[:,:,1]
                    
    # dim_m=np.mean(dim,axis=3)
    # dim_std=np.nanstd(dim,axis=3)
    # abs_m=np.mean(abstr,axis=(3,5))
    # abs_std=np.std(abstr,axis=(3,5))

    # #####################
    # # Plots
    # tl_vec=['targets on','dots on','saccade']


   
