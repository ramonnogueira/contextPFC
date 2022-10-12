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
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import KFold,StratifiedKFold,StratifiedShuffleSplit

import miscellaneous

nan=float('nan')
minf=float('-inf')
pinf=float('inf')
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

def adjust_spines(ax, spines):
    for loc, spine in ax.spines.items():
        if loc in spines: 
            if loc=='left':
                spine.set_position(('outward', 10))  # outward by 10 points
            if loc=='bottom':
                spine.set_position(('outward', 0))  # outward by 10 points
         #   spine.set_smart_bounds(True)
        else:
            spine.set_color('none')  # don't draw spine
    # turn off ticks where there is no spine
    if 'left' in spines:
        ax.yaxis.set_ticks_position('left')
    else:
        # no yaxis ticks
        ax.yaxis.set_ticks([])
    if 'bottom' in spines:
        ax.xaxis.set_ticks_position('bottom')
    else:
        # no xaxis ticks
        ax.xaxis.set_ticks([])

#################################################

n_cv=10
reg=1e-3
lr=0.005
test_size=0.2
#models_vec=[(),(200),(200,200),(200,200,200)]
models_vec=[(),(100),(100,100),(100,100,100)]
#models_vec=[(),(20),(20,20),(20,20,20)]

monkeys=['Niels','Galileo']

for k in range(len(monkeys)):
    #abs_path='/home/ramon/Dropbox/Esteki_Kiani/data/sorted/late/%s/'%monkeys[k]
    abs_path='/home/ramon/Dropbox/Esteki_Kiani/data/unsorted/%s/'%monkeys[k] 
    files=os.listdir(abs_path)
    perf_choice_pre=nan*np.zeros((len(files),n_cv,len(models_vec)+1))
    perf_rt_pre=nan*np.zeros((len(files),n_cv,len(models_vec)+1))
    for kk in range(len(files)):
        #Load data
        print (files[kk])
        data=scipy.io.loadmat(abs_path+'%s'%(files[kk]),struct_as_record=False,simplify_cells=True)
        beha=miscellaneous.behavior(data)
        index_nonan=beha['index_nonan']
        reward=beha['reward']
        coh_signed=beha['coherence_signed']
        context=beha['context']
        stimulus=beha['stimulus']
        difficulty=beha['difficulty']
        choice=beha['choice']
        coh_uq=np.unique(coh_signed)
        rt=np.log(beha['reaction_time'])
        
        # Create Features for Behavior Linear and Nonlinear Classifiers
        features=nan*np.zeros((len(choice)-1,7))
        features[:,0]=coh_signed[0:-1] 
        features[:,1]=coh_signed[1:]
        features[:,2]=difficulty[0:-1]
        features[:,3]=difficulty[1:]
        features[:,4]=reward[0:-1] 
        features[:,5]=context[1:]
        features[:,6]=choice[0:-1]
        # cm1=choice[0:-1].copy()
        # cm1[cm1==0]=-1
        # rm1=reward[0:-1].copy()
        # rm1[rm1==0]=-1
        # ctx0=context[1:].copy()
        # ctx0[ctx0==0]=-1
        # features[:,7]=(cm1*rm1)
        # features[:,7][(cm1*rm1)==-1]=0
        # features[:,8]=(cm1*ctx0)
        # features[:,8][(cm1*ctx0)==-1]=0
        target=choice[1:]
        print ('Balanced? ',np.mean(target))
        #
        feat_norm=nan*np.zeros(np.shape(features))
        for i in range(len(features[0])):
            feat_m=np.mean(features[:,i])
            feat_s=np.std(features[:,i])
            feat_norm[:,i]=(features[:,i]-feat_m)/feat_s

        # Classifier for all trials
        cv=StratifiedShuffleSplit(n_splits=n_cv,test_size=test_size)
        g=-1
        for train_index, test_index in cv.split(feat_norm,target):
            g=(g+1)
            cl=LogisticRegression(C=1/reg,class_weight='balanced')
            cl.fit(feat_norm[train_index],target[train_index])
            perf_choice_pre[kk,g,0]=cl.score(feat_norm[test_index],target[test_index])

            cl=LinearRegression()
            ind_nonan_tr=~np.isnan(rt[train_index])
            ind_nonan_te=~np.isnan(rt[test_index])
            cl.fit(feat_norm[train_index][ind_nonan_tr],rt[train_index][ind_nonan_tr])
            perf_rt_pre[kk,g,0]=cl.score(feat_norm[test_index][ind_nonan_te],rt[test_index][ind_nonan_te])

            for j in range(len(models_vec)):
                nn=MLPClassifier(hidden_layer_sizes=models_vec[j],activation='relu',alpha=reg,learning_rate_init=lr)
                nn.fit(feat_norm[train_index],target[train_index])
                perf_choice_pre[kk,g,j+1]=nn.score(feat_norm[test_index],target[test_index])
                
                nn=MLPRegressor(hidden_layer_sizes=models_vec[j],activation='relu',alpha=reg,learning_rate_init=lr)
                nn.fit(feat_norm[train_index][ind_nonan_tr],rt[train_index][ind_nonan_tr])
                perf_rt_pre[kk,g,j+1]=nn.score(feat_norm[test_index][ind_nonan_te],rt[test_index][ind_nonan_te])

    perf_choice=np.mean(perf_choice_pre,axis=1)
    perf_choice_m=np.mean(perf_choice,axis=0)
    perf_choice_sem=sem(perf_choice,axis=0)
    print (perf_choice)
    print (perf_choice_m)
    print (perf_choice_sem)

    perf_rt=np.mean(perf_rt_pre,axis=1)
    perf_rt_m=np.mean(perf_rt,axis=0)
    perf_rt_sem=sem(perf_rt,axis=0)
    print (perf_rt)
    print (perf_rt_m)
    print (perf_rt_sem)
    
            # for i in range(len(feat_norm[0])):
    #             feat_used=feat_norm.copy()
    #             feat_used[:,i]=np.zeros(len(feat_norm))
    #             delta_perf[kk,g,i]=(perf-nn.score(feat_used[test_index],target[test_index]))

    # delta_m=np.mean(delta_perf,axis=(0,1))
    # print (delta_m)

        # # Classifier conditioned on coherence
        # print ('Classifier per coherence')
        # lr=0.01
        # arch=(100,100,100)
        # reg=1e-1
        # for i in range(len(coh_unique)):
        #     #print (i)
        #     ind_coh=np.where(coh[1:]==coh_unique[i])[0]
        #     cv=StratifiedShuffleSplit(n_splits=n_cv,test_size=0.2)
        #     g=-1
        #     for train_index, test_index in cv.split(feat_norm[ind_coh],target[ind_coh]):
        #         g=(g+1)
        #         # Linear Classifier
        #         cl=LogisticRegression(C=1/reg,class_weight='balanced')
        #         cl.fit(feat_norm[ind_coh][train_index],target[ind_coh][train_index])
        #         perf_lin[kk,i,g,1]=cl.score(feat_norm[ind_coh][test_index],target[ind_coh][test_index])
        #         wei[kk,i,g,:,1]=cl.coef_[0]
        #         # Nonlinear Classifier
        #         nn=MLPClassifier(hidden_layer_sizes=arch,activation='relu',batch_size=batch,alpha=reg,learning_rate_init=lr)
        #         nn.fit(feat_norm[ind_coh][train_index],target[ind_coh][train_index])
        #         perf_nlin[kk,i,g,1]=nn.score(feat_norm[ind_coh][test_index],target[ind_coh][test_index])
            
    # # Classifiers Behavior linear and nonlinear
    # perf_lin_pre=np.mean(perf_lin,axis=(2))
    # perf_nlin_pre=np.mean(perf_nlin,axis=(2))
    # wei_pre=np.mean(wei,axis=(2))
    # perf_lin_m=np.mean(perf_lin_pre,axis=0)
    # perf_lin_sem=sem(perf_lin_pre,axis=0)
    # perf_nlin_m=np.mean(perf_nlin_pre,axis=0)
    # perf_nlin_sem=sem(perf_nlin_pre,axis=0)
    # wei_m=np.mean(wei_pre,axis=0)
    # wei_sem=sem(wei_pre,axis=0)
    # fig=plt.figure(figsize=(4,3))
    # ax=fig.add_subplot(1,1,1)
    # adjust_spines(ax,['left','bottom'])
    # ax.plot(coh_unique,perf_lin_m[:,0],color='blue',label='Linear All',linestyle='--')
    # ax.fill_between(coh_unique,perf_lin_m[:,0]-perf_lin_sem[:,0],perf_lin_m[:,0]+perf_lin_sem[:,0],color='blue',alpha=0.4)
    # ax.plot(coh_unique,perf_lin_m[:,1],color='blue',label='Linear')
    # ax.fill_between(coh_unique,perf_lin_m[:,1]-perf_lin_sem[:,1],perf_lin_m[:,1]+perf_lin_sem[:,1],color='blue',alpha=0.7)
    # ax.plot(coh_unique,perf_nlin_m[:,0],color='green',label='NonLinear All',linestyle='--')
    # ax.fill_between(coh_unique,perf_nlin_m[:,0]-perf_nlin_sem[:,0],perf_nlin_m[:,0]+perf_nlin_sem[:,0],color='green',alpha=0.4)
    # ax.plot(coh_unique,perf_nlin_m[:,1],color='green',label='NonLinear')
    # ax.fill_between(coh_unique,perf_nlin_m[:,1]-perf_nlin_sem[:,1],perf_nlin_m[:,1]+perf_nlin_sem[:,1],color='green',alpha=0.7)
    # ax.set_xlabel('Motion Coherence')
    # ax.set_ylabel('DP Choice')
    # ax.plot(coh_unique,0.5*np.ones(len(coh_unique)),color='black',linestyle='--')
    # plt.legend(loc='best')
    # fig.savefig('/home/ramon/Dropbox/Esteki_Kiani/plots/classifier_behavior_linear_nonlinear_%s.pdf'%monkeys[k],dpi=500,bbox_inches='tight')




