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

def calc_dp(feat,clase,wei):
    ind1=np.where(clase==1)[0]
    ind0=np.where(clase==0)[0]
    mean1=np.mean(feat[ind1],axis=0)
    mean0=np.mean(feat[ind0],axis=0)
    deltaf=(mean1-mean0)
    cov=0.5*(np.cov(feat[ind1],rowvar=False)+np.cov(feat[ind0],rowvar=False))
    num=np.dot(deltaf,wei)
    denom=np.sqrt(np.dot(wei,np.dot(cov,wei)))
    return 0.5*num/denom
#####################################

# target onset: 'targ_on', dots onset: 'dots_on', dots offset: 'dots_off', saccade: 'response_edf'
talig_vec=np.array(['dots_on','response_edf'])
dic_time={} 
dic_time['dots_on']=np.array([0,1000,200,200])# time pre, time post, bin size, step size
dic_time['response_edf']=np.array([900,100,200,200])
steps_dic={}
xx_dic={}
for i in range(len(talig_vec)):
    steps_dic[talig_vec[i]]=int((dic_time[talig_vec[i]][0]+dic_time[talig_vec[i]][1])/dic_time[talig_vec[i]][3])
    xx_dic[talig_vec[i]]=np.linspace(-dic_time[talig_vec[i]][0]/1000,dic_time[talig_vec[i]][1]/1000,steps_dic[talig_vec[i]],endpoint=False)

print (xx_dic)

nt=100
n_rand=20
perc_tr=0.8
thres=0
reg=1e-5

quant_vec=['reward_m1','choice_0','context_0']#
tback=1
# Choice0, Choice-1, Context0, Reward-1, maybe Stimulus0?
#col=['blue','blue','brown','red']
#alph=[0.7,1,1,0.7]
#col=['green','blue','brown']
#alph=[1,1,1]
col=['red','blue','brown']
alph=[0.7,1,1]


monkeys=['Niels','Galileo']
phase='late'

for k in range(len(monkeys)):
    print (monkeys[k])
    abs_path='/home/ramon/Dropbox/Esteki_Kiani/data/sorted/%s/%s/'%(phase,monkeys[k]) 
    files=os.listdir(abs_path)
    for kkk in range(len(talig_vec)):
        steps=steps_dic[talig_vec[kkk]]
        xx=xx_dic[talig_vec[kkk]]
        perf=nan*np.zeros((len(quant_vec),steps,n_rand,2))
        print ('  ',talig_vec[kkk])
        pseudo=miscellaneous.pseudopopulation_nvar(abs_path,files,quant_vec,talig_vec[kkk],dic_time[talig_vec[kkk]],steps,thres,nt,n_rand,perc_tr,tback)
        pseudo_tr=pseudo['pseudo_tr']
        pseudo_te=pseudo['pseudo_te']
        for q in range(len(quant_vec)):
            clase=pseudo['clase_var'][:,q]
            for i in range(steps):
                for ii in range(n_rand):
                    try:
                        cl=LogisticRegression(C=1/reg,class_weight='balanced')
                        cl.fit(pseudo_tr[i,ii],clase)
                        perf[q,i,ii,0]=cl.score(pseudo_tr[i,ii],clase)
                        perf[q,i,ii,1]=cl.score(pseudo_te[i,ii],clase)
                    except:
                        print (i,ii)

        perf_m=np.nanmean(perf,axis=2)
        perf_std=np.nanstd(perf,axis=2)
        #print (perf_m[:,:,:,1])

        for ii in range(len(quant_vec)):
            fig=plt.figure(figsize=(6,4))
            ax=fig.add_subplot(111)
            miscellaneous.adjust_spines(ax,['left','bottom'])
            ax.plot(xx,perf_m[ii,:,1],color=col[ii],alpha=alph[ii])
            ax.fill_between(xx,perf_m[ii,:,1]-perf_std[ii,:,1],perf_m[ii,:,1]+perf_std[ii,:,1],color=col[ii],alpha=0.7*alph[ii])
            ax.axvline(0,color='black',linestyle='--')
            ax.set_ylim([0.4,1.0])
            ax.plot(xx,0.5*np.ones(steps),color='black',linestyle='--')
            fig.savefig('/home/ramon/Dropbox/Esteki_Kiani/plots/decoding_%s_pseudo_decorrelated_%s_%s_bin_%i_3.png'%(monkeys[k],talig_vec[kkk],quant_vec[ii],dic_time['dots_on'][2]),dpi=500,bbox_inches='tight')
