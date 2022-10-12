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
talig_vec=np.array(['dots_on']) #'targ_on',,'response_edf
dic_time={} # same number of steps for all time locks
dic_time['targ_on']=np.array([1000,1000,500,200]) # time pre, time post, bin size, step size
dic_time['dots_on']=np.array([0,1000,500,200])
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
n_dim=100

quant_vec=['choice','context','difficulty']
#col=['green','blue','brown','purple','red','lime','royalblue','orange','pink','salmon']
monkeys=['Niels']#,'Galileo']

for k in range(len(monkeys)):
    print (monkeys[k])
    abs_path='/home/ramon/Dropbox/Esteki_Kiani/data/%s/'%monkeys[k] 
    files=os.listdir(abs_path)
    dim=nan*np.zeros((len(talig_vec),steps,n_rand))
    abstraction=nan*np.zeros((len(talig_vec),steps,n_rand,3))
    for kkk in range(len(talig_vec)):
        print ('  ',talig_vec[kkk])
        pseudo=miscellaneous.pseudopopulation_3(abs_path,files,quant_vec,talig_vec[kkk],dic_time[talig_vec[kkk]],steps,thres,nt,n_rand,perc_tr)
        pseudo_all=pseudo['pseudo_all']
        pseudo_tr=pseudo['pseudo_tr']
        pseudo_te=pseudo['pseudo_te']
        clase_all=pseudo['clase_all']
        for i in range(steps):
            for ii in range(n_rand):
                dim[kkk,i,ii]=miscellaneous.dim_pseudo_3(pseudo_tr[i,ii],pseudo_te[i,ii],clase_all,n_dim,reg)
                abstraction[kkk,i,ii]=np.mean(miscellaneous.abstraction_3D(pseudo_all[i,ii],clase_all)[:,:,1],axis=1)
            print (np.mean(dim[kkk,i]))
            print (np.mean(abstraction[kkk,i],axis=0))
                    
    # perf_m=np.nanmean(perf,axis=3)
    # perf_std=np.nanstd(perf,axis=3)

    # #####################
    # if monkeys[k]=='Niels':
    #     n_coh=7
    # if monkeys[k]=='Galileo':
    #     n_coh=8
    # # Plots
    # tl_vec=['targets on','dots on','saccade']
    # fig=plt.figure(figsize=(len(tl_vec)*4,len(quant_vec)*3))
    # for kk in range(len(quant_vec)):
    #     for kkk in range(len(talig_vec)):
    #         ax=fig.add_subplot(len(quant_vec),len(tl_vec),kk*3+kkk+1)
    #         miscellaneous.adjust_spines(ax,['left','bottom'])
    #         if kk==0:
    #             ax.set_title('Time lock %s'%tl_vec[kkk])
    #         if kkk==0:
    #             ax.set_ylabel('Decoding Performance \n %s'%quant_vec[kk])
    #         if kk==(len(quant_vec)-1):
    #             ax.set_xlabel('Time from %s (sec)'%tl_vec[kkk])
            
    #         ax.plot(xx_dic[talig_vec[kkk]],perf_m[kkk,kk,:,1],color=col[kk])
    #         ax.fill_between(xx_dic[talig_vec[kkk]],perf_m[kkk,kk,:,1]-perf_std[kkk,kk,:,1],perf_m[kkk,kk,:,1]+perf_std[kkk,kk,:,1],color=col[kk],alpha=0.7)
    #         ax.axvline(0,color='black',linestyle='--')
    #         ax.set_ylim([0.4,1.0])
    #         ax.plot(xx_dic[talig_vec[kkk]],0.5*np.ones(steps),color='black',linestyle='--')
    # fig.savefig('/home/ramon/Dropbox/Esteki_Kiani/plots/decoding_%s_bin_%i_thres_%.1f_pseudo.pdf'%(monkeys[k],dic_time['dots_on'][2],thres),dpi=500,bbox_inches='tight')

   
