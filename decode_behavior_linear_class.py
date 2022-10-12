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
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import KFold,StratifiedKFold,StratifiedShuffleSplit
from sklearn.neural_network import MLPClassifier
import miscellaneous

nan=float('nan')
minf=float('-inf')
pinf=float('inf')
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

#################################################

monkeys=['Niels','Galileo']#,'Galileo']

n_cv=100
reg=1e2

perf_ch_pre=nan*np.zeros((len(monkeys),3,8,n_cv))
wei_ch_pre=nan*np.zeros((len(monkeys),3,8,n_cv,5))
perf_rt_pre=nan*np.zeros((len(monkeys),3,8,n_cv))
wei_rt_pre=nan*np.zeros((len(monkeys),3,8,n_cv,5))
for k in range(len(monkeys)):
    abs_path='/home/ramon/Dropbox/Esteki_Kiani/data/sorted/late/%s/'%monkeys[k] 
    files=os.listdir(abs_path)
    for kk in range(len(files)):
        print (files[kk])
        #Load data
        data=scipy.io.loadmat(abs_path+'%s'%(files[kk]),struct_as_record=False,simplify_cells=True)
        beha=miscellaneous.behavior(data)
        index_nonan=beha['index_nonan']
        reward=beha['reward']
        coh=beha['coherence']
        coh_unique=np.unique(coh)
        context=beha['context']
        stimulus=beha['stimulus']
        difficulty=beha['difficulty']
        choice=beha['choice']
        rt=beha['reaction_time']
       
        # Create Features for Behavior Linear and Nonlinear Classifiers
        features=nan*np.zeros((len(choice)-1,5))
        features[:,0]=stimulus[0:-1] 
        features[:,1]=stimulus[1:]
        #features[:,2]=difficulty[0:-1]
        #features[:,3]=difficulty[1:]
        features[:,2]=reward[0:-1] 
        features[:,3]=context[1:]
        features[:,4]=choice[0:-1]
        target=choice[1:]
        #
        feat_norm=nan*np.zeros(np.shape(features))
        for i in range(len(features[0])):
            feat_m=np.mean(features[:,i])
            feat_s=np.std(features[:,i])
            feat_norm[:,i]=(features[:,i]-feat_m)/feat_s

        for i in range(len(coh_unique)):
            #print (i)
            ind_coh=np.where(coh[1:]==coh_unique[i])[0]
            cv=StratifiedShuffleSplit(n_splits=n_cv,test_size=0.2)
            g=-1
            for train_index, test_index in cv.split(feat_norm[ind_coh],target[ind_coh]):
                g=(g+1)
                # Linear Classifier
                cl=LogisticRegression(C=1/reg,class_weight='balanced')
                cl.fit(feat_norm[ind_coh][train_index],target[ind_coh][train_index])
                perf_ch_pre[k,kk,i,g]=cl.score(feat_norm[ind_coh][test_index],target[ind_coh][test_index])
                wei_ch_pre[k,kk,i,g]=cl.coef_[0]

                # Linear Regression
                cl=LinearRegression()
                ind_nonan_tr=~np.isnan(rt[ind_coh][train_index])
                ind_nonan_te=~np.isnan(rt[ind_coh][test_index])
                #print (len(rt[ind_coh][train_index][ind_nonan_tr]),len(rt[ind_coh][test_index][ind_nonan_te]))
                cl.fit(feat_norm[ind_coh][train_index][ind_nonan_tr],rt[ind_coh][train_index][ind_nonan_tr])
                perf_rt_pre[k,kk,i,g]=cl.score(feat_norm[ind_coh][test_index][ind_nonan_te],rt[ind_coh][test_index][ind_nonan_te])
                wei_rt_pre[k,kk,i,g]=cl.coef_

                
#################################################################
# Classifier Choice 
perf_ch=np.mean(perf_ch_pre,axis=(3))
wei_ch=np.mean(wei_ch_pre,axis=(3))
perf_ch_m=np.mean(perf_ch,axis=1)
perf_ch_sem=sem(perf_ch,axis=1)
wei_ch_m=np.mean(wei_ch,axis=1)
wei_ch_sem=sem(wei_ch,axis=1)

# Max coherence set to nan for Niels
wei_ch_m[0,7]=nan

fig=plt.figure(figsize=(10,2*3.5))
#col=['lime','green','violet','purple','salmon','orange','brown','blue']
col=['lime','green','salmon','orange','brown','blue']
alph=[0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
width=0.12
for i in range(len(monkeys)):
    ax=fig.add_subplot(len(monkeys),1,i+1)
    miscellaneous.adjust_spines(ax,['left','bottom'])
    ax.set_title('%s'%monkeys[i])
    for ii in range(len(perf_ch_m[0])):
        ax.bar(np.arange(len(feat_norm[0]))+width*ii-0.4,wei_ch_m[i,ii],yerr=wei_ch_sem[i,ii],color=col,alpha=alph[ii],width=width)
    ax.set_ylabel('Weights classifier')
    if i==1:
        #plt.legend(loc='best')
        #plt.xticks(np.arange(len(feat_norm[0])),['Prev. Stimulus','Curr. Stimulus','Prev. Diff','Curr. Diff','Prev. Reward','Prev. Context','Prev. Choice'])
        plt.xticks(np.arange(len(feat_norm[0])),['Prev. Stimulus','Curr. Stimulus','Prev. Reward','Prev. Context','Prev. Choice'])
fig.savefig('/home/ramon/Dropbox/Esteki_Kiani/plots/weights_coherences_choice.pdf',dpi=500,bbox_inches='tight')

# Classifier Choice 
perf_rt=np.mean(perf_rt_pre,axis=(3))
wei_rt=np.mean(wei_rt_pre,axis=(3))
perf_rt_m=np.mean(perf_rt,axis=1)
perf_rt_sem=sem(perf_rt,axis=1)
wei_rt_m=np.mean(wei_rt,axis=1)
wei_rt_sem=sem(wei_rt,axis=1)

# Max coherence set to nan for Niels
wei_rt_m[0,7]=nan

fig=plt.figure(figsize=(10,2*3.5))
for i in range(len(monkeys)):
    ax=fig.add_subplot(len(monkeys),1,i+1)
    miscellaneous.adjust_spines(ax,['left','bottom'])
    ax.set_title('%s'%monkeys[i])
    for ii in range(len(perf_ch_m[0])):
        ax.bar(np.arange(len(feat_norm[0]))+width*ii-0.4,wei_rt_m[i,ii],yerr=wei_rt_sem[i,ii],color=col,alpha=alph[ii],width=width)
    ax.set_ylabel('Weights classifier')
    if i==1:
        #plt.legend(loc='best')
        #plt.xticks(np.arange(len(feat_norm[0])),['Prev. Stimulus','Curr. Stimulus','Prev. Diff','Curr. Diff','Prev. Reward','Prev. Context','Prev. Choice'])
        plt.xticks(np.arange(len(feat_norm[0])),['Prev. Stimulus','Curr. Stimulus','Prev. Reward','Prev. Context','Prev. Choice'])
fig.savefig('/home/ramon/Dropbox/Esteki_Kiani/plots/weights_coherences_rt.pdf',dpi=500,bbox_inches='tight')


