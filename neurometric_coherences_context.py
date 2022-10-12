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

# target onset: 'targ_on', dots onset: 'dots_on', dots offset: 'dots_off', saccade: 'response_edf'
talig_vec=np.array(['response_edf'])#'targ_on','dots_on'
dic_time={} # same number of steps for all time locks
dic_time['targ_on']=np.array([0,1000,200,100]) # time pre, time post, bin size, step size
dic_time['dots_on']=np.array([0,1000,200,100])
dic_time['response_edf']=np.array([1000,200,200,200])
steps_dic={}
xx_dic={}
for i in range(len(talig_vec)):
    steps_dic[talig_vec[i]]=int((dic_time[talig_vec[i]][0]+dic_time[talig_vec[i]][1])/dic_time[talig_vec[i]][3])
    xx_dic[talig_vec[i]]=np.linspace(-dic_time[talig_vec[i]][0]/1000,dic_time[talig_vec[i]][1]/1000,steps_dic[talig_vec[i]],endpoint=False)

monkeys=['Niels']#,'Galileo']
phase='late'

thres=0
n_cv=4
steps=steps_dic[talig_vec[0]]
xx=xx_dic[talig_vec[0]]
reg=1

for k in range(len(monkeys)):
    print (monkeys[k])
    abs_path='/home/ramon/Dropbox/Esteki_Kiani/data/%s/%s/'%(phase,monkeys[k])
    files=os.listdir(abs_path)

    neurometric_pre=nan*np.zeros((len(files),steps,15,3,n_cv,2))
    pairwise_coh_pre=nan*np.zeros((len(files),steps,15,15,3,n_cv,2)) 
    
    for kk in range(len(files)):
        #Load data
        print ('  ',files[kk])
        data=scipy.io.loadmat(abs_path+'%s'%(files[kk]),struct_as_record=False,simplify_cells=True)
        behavior=miscellaneous.behavior(data)
        index_nonan=behavior['index_nonan']
        reward=behavior['reward']
        ind_corr=np.where(reward==1)[0]
        
        choice=behavior['choice'][ind_corr]
        stimulus=behavior['stimulus'][ind_corr]
        coh_num=behavior['coh_num'][ind_corr]
        coh_num_uq=np.unique(coh_num)
        context=behavior['context'][ind_corr]
        
        firing_rate_pre=miscellaneous.getRasters(data,talig_vec[0],dic_time[talig_vec[0]],index_nonan,thres)[ind_corr] #trials, neurons, time
        firing_rate=miscellaneous.normalize_fr(firing_rate_pre)

        for i in range(steps):
            for j in range(len(coh_num_uq)):
                for jj in range(j+1,len(coh_num_uq)):
                    ind_coh=np.where((coh_num==coh_num_uq[j])|(coh_num==coh_num_uq[jj]))[0]
                    ind_coh1=np.where((((coh_num==coh_num_uq[j])|(coh_num==coh_num_uq[jj]))&(context==1)))[0]
                    ind_coh0=np.where((((coh_num==coh_num_uq[j])|(coh_num==coh_num_uq[jj]))&(context==0)))[0]

                    firing_coh=firing_rate[ind_coh]
                    firing_coh1=firing_rate[ind_coh1]
                    firing_coh0=firing_rate[ind_coh0]

                    choice_coh=choice[ind_coh]
                    choice_coh1=choice[ind_coh1]
                    choice_coh0=choice[ind_coh0]
                    
                    coh_num_coh=coh_num[ind_coh]
                    coh_num_coh1=coh_num[ind_coh1]
                    coh_num_coh0=coh_num[ind_coh0]

                    print (j,jj,len(coh_num_coh),np.mean(coh_num_coh))

                    # All
                    cv=StratifiedKFold(n_splits=n_cv)
                    g=-1
                    for train_index, test_index in cv.split(firing_coh,coh_num_coh):
                        g=(g+1)
                        cl=LogisticRegression(C=1/reg,class_weight='balanced')
                        cl.fit(firing_coh[train_index][:,:,i],coh_num_coh[train_index])
                        pairwise_coh_pre[kk,i,j,jj,0,g,0]=cl.score(firing_coh[train_index][:,:,i],coh_num_coh[train_index])
                        pairwise_coh_pre[kk,i,j,jj,0,g,1]=cl.score(firing_coh[test_index][:,:,i],coh_num_coh[test_index])

                    # Context 1
                    cv=StratifiedKFold(n_splits=n_cv)
                    g=-1
                    for train_index, test_index in cv.split(firing_coh1,coh_num_coh1):
                        g=(g+1)
                        cl=LogisticRegression(C=1/reg,class_weight='balanced')
                        cl.fit(firing_coh1[train_index][:,:,i],coh_num_coh1[train_index])
                        pairwise_coh_pre[kk,i,j,jj,1,g,0]=cl.score(firing_coh1[train_index][:,:,i],coh_num_coh1[train_index])
                        pairwise_coh_pre[kk,i,j,jj,1,g,1]=cl.score(firing_coh1[test_index][:,:,i],coh_num_coh1[test_index])

                    # Context 0
                    cv=StratifiedKFold(n_splits=n_cv)
                    g=-1
                    for train_index, test_index in cv.split(firing_coh0,coh_num_coh0):
                        g=(g+1)
                        cl=LogisticRegression(C=1/reg,class_weight='balanced')
                        cl.fit(firing_coh0[train_index][:,:,i],coh_num_coh0[train_index])
                        pairwise_coh_pre[kk,i,j,jj,2,g,0]=cl.score(firing_coh0[train_index][:,:,i],coh_num_coh0[train_index])
                        pairwise_coh_pre[kk,i,j,jj,2,g,1]=cl.score(firing_coh0[test_index][:,:,i],coh_num_coh0[test_index])
                        
    pairwise_coh=np.mean(pairwise_coh_pre,axis=(0,5))
                   
    vmin=0.4
    vmax=1.0
    for o in range(steps):
        print (xx[o])
        plt.imshow(pairwise_coh[o,1:-1,1:-1,0,1],vmin=vmin,vmax=vmax)
        plt.colorbar()
        plt.title('Perf All')
        plt.show()

        plt.imshow(pairwise_coh[o,1:-1,1:-1,1,1],vmin=vmin,vmax=vmax)
        plt.colorbar()
        plt.title('Perf Ctx1')
        plt.show()

        plt.imshow(pairwise_coh[o,1:-1,1:-1,2,1],vmin=vmin,vmax=vmax)
        plt.colorbar()
        plt.title('Perf Ctx0')
        plt.show()
                    



                
