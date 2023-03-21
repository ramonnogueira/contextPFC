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
import miscellaneous
import datetime
from scipy.stats import ortho_group
from scipy.stats import special_ortho_group

nan=float('nan')
minf=float('-inf')
pinf=float('inf')
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn


def clase_resolution(group_coh,coherence):
    clase_coh=nan*np.zeros(len(coherence))
    coh_uq=np.unique(coherence)
    for i in range(len(coh_uq)):
        ind=np.where(coherence==coh_uq[i])[0]
        clase_coh[ind]=group_coh[i]
    return clase_coh
    

def shuffle_distr(pseudo_tr,pseudo_te,clase_all,rot_mat_vec):
    clase_uq=np.unique(clase_all)
    pseudo_tr_sh=nan*np.zeros(np.shape(pseudo_tr))
    pseudo_te_sh=nan*np.zeros(np.shape(pseudo_te))
    for i in range(len(clase_uq)):
        ind_cl=np.where(clase_all==clase_uq[i])[0]
        rot_mat=rot_mat_vec[i]
        #print (rot_mat)
        #pseudo_tr_sh[ind_cl]=np.dot(pseudo_tr[ind_cl],rot_mat)
        #pseudo_te_sh[ind_cl]=np.dot(pseudo_te[ind_cl],rot_mat)
        pseudo_tr_sh[ind_cl]=pseudo_tr[ind_cl][:,rot_mat]
        pseudo_te_sh[ind_cl]=pseudo_te[ind_cl][:,rot_mat]
    return pseudo_tr_sh,pseudo_te_sh

def index_shuffle(num_neu,clase_all):
    clase_uq=np.unique(clase_all)
    #rot_mat_vec=nan*np.zeros((len(clase_uq),num_neu,num_neu))
    rot_mat_vec=[]#nan*np.zeros((len(clase_uq),num_neu))
    for i in range(len(clase_uq)):
        #print (i)
        #rot_mat_vec[i]=ortho_group.rvs(num_neu)
        rot_mat_vec.append(np.random.permutation(np.arange(num_neu)))
    rot_mat_vec=np.array(rot_mat_vec)
    return rot_mat_vec

def null_model_coh(repr_tr,repr_te,pert_std,n_coh,nt):
    n_neu=len(repr_tr[0])
    repr_tr_pert=nan*np.zeros(np.shape(repr_tr))
    repr_te_pert=nan*np.zeros(np.shape(repr_te))
    for i in range(2*n_coh):
        pert=np.random.normal(0,pert_std,n_neu)
        for ii in range(nt):
            repr_tr_pert[i*nt+ii]=(repr_tr[i*nt+ii]+pert)
            repr_te_pert[i*nt+ii]=(repr_te[i*nt+ii]+pert)
    return repr_tr_pert,repr_te_pert

def classifier(data,var):
    n_cv=5
    reg=1
    perf=nan*np.zeros((n_cv,2))
    skf=StratifiedKFold(n_splits=n_cv)
    g=-1
    for train, test in skf.split(data,var):
        g=(g+1)
        cl=LogisticRegression(C=1/reg)
        cl.fit(data[train],var[train])
        perf[g,0]=cl.score(data[train],var[train])
        perf[g,1]=cl.score(data[test],var[test])
    return np.mean(perf,axis=0)

def rotation_ccgp(pseudo,clase_all):
    clase_uq=np.unique(clase_all)
    num_neu=pseudo.shape[-1]
    pseudo_sh=nan*np.zeros(np.shape(pseudo))
    for i in range(len(clase_uq)):
        ind_cl=np.where(clase_all==clase_uq[i])[0]
        rot_mat=np.random.permutation(np.arange(num_neu))
        pseudo_sh[:,ind_cl]=pseudo[:,ind_cl][:,:,rot_mat]
    return pseudo_sh


def flat_time_pseudo(pseudo):
    steps=pseudo.shape[0]
    neu_total=pseudo.shape[-1]
    pseudo_flat=nan*np.zeros((pseudo.shape[1],pseudo.shape[2],steps*neu_total))
    for p in range(pseudo.shape[1]):
        for pp in range(steps):
            pseudo_flat[p,:,pp*neu_total:(pp+1)*neu_total]=pseudo[pp,p]
    return pseudo_flat

def abstraction_2D(feat_decod,feat_binary,bias,reg):
    exp_uq=np.unique(feat_binary,axis=0)
    feat_binary_exp=np.zeros(len(feat_binary))
    for t in range(len(feat_binary)):
        for tt in range((len(exp_uq))):
            gg=(np.sum(feat_binary[t]==exp_uq[tt])==len(feat_binary[0]))
            if gg:
                feat_binary_exp[t]=tt
    #
    #dichotomies=np.array([[0,0,1,1],[0,1,0,1],[0,1,1,0]])
    #train_dich=np.array([[[0,2],[1,3],[0,3],[1,2]],[[0,1],[2,3],[0,3],[1,2]],[[0,1],[2,3],[0,2],[1,3]]])
    #test_dich=np.array([[[1,3],[0,2],[1,2],[0,3]],[[2,3],[0,1],[1,2],[0,3]],[[2,3],[0,1],[1,3],[0,2]]])
    dichotomies=np.array([[0,0,1,1],[0,1,0,1]])
    train_dich=np.array([[[0,2],[1,3]],[[0,1],[2,3]]])
    test_dich=np.array([[[1,3],[0,2]],[[2,3],[0,1]]])
    
    perf=nan*np.zeros((len(dichotomies),len(train_dich[0])))
    inter=nan*np.zeros((len(dichotomies),len(train_dich[0])))
    for k in range(len(dichotomies)): #Loop on "dichotomies"
      for kk in range(len(train_dich[0])): #Loop on ways to train this particular "dichotomy"
         ind_train=np.where((feat_binary_exp==train_dich[k][kk][0])|(feat_binary_exp==train_dich[k][kk][1]))[0]
         ind_test=np.where((feat_binary_exp==test_dich[k][kk][0])|(feat_binary_exp==test_dich[k][kk][1]))[0]

         task=nan*np.zeros(len(feat_binary_exp))
         for i in range(4):
             ind_task=(feat_binary_exp==i)
             task[ind_task]=dichotomies[k][i]

         supp=LogisticRegression(C=1/reg,class_weight='balanced')
         #supp=LinearSVC(C=1/reg,class_weight='balanced')
         mod=supp.fit(feat_decod[ind_train],task[ind_train])
         inter[k,kk]=mod.intercept_[0]
         pred=(np.dot(feat_decod[ind_test],supp.coef_[0])+supp.intercept_+bias)>0
         perf[k,kk]=np.mean(pred==task[ind_test])
         #perf[k,kk,0]=supp.score(feat_decod[ind_train],task[ind_train])
         #perf[k,kk,1]=supp.score(feat_decod[ind_test],task[ind_test])
    return perf,inter

##############################################

monkeys=['Niels']

# target onset: 'targ_on', dots onset: 'dots_on', dots offset: 'dots_off', saccade: 'response_edf'
#talig='response_edf'
#dic_time=np.array([650,-50,200,200])# time pre, time post, bin size, step size (time pre always positive) #For Galileo use timepost 800 or 1000. For Niels use 
talig='dots_on'
dic_time=np.array([0,600,200,200])# time pre, time post, bin size, step size (time pre always positive) #For Galileo use timepost 800 or 1000. For Niels use 
steps=int((dic_time[0]+dic_time[1])/dic_time[3])
xx=np.linspace(-dic_time[0]/1000,dic_time[1]/1000,steps,endpoint=False)

nt=100 #100 for coh signed, 200 for coh unsigned, 50 for coh signed with context
n_rand=10
n_shuff=0
perc_tr=0.8
thres=0
reg=1e0
n_coh=15

group_ref=np.array([-7 ,-6 ,-5 ,-4 ,-3 ,-2 ,-1 ,0  ,1  ,2  ,3  ,4  ,5  ,6  ,7  ])
#group_coh=np.array([-7 ,-6 ,-5 ,-4 ,-3 ,-2 ,-1 ,0  ,1  ,2  ,3  ,4  ,5  ,6  ,7])
#group_coh=np.array([-3,-3,-3,-2,-2,-1,-1,0,1,1,2,2,3,3,3])
#group_coh=np.array([-3,-3,-2,-2,-2,-1,-1,0,1,1,2,2,2,3,3])
#group_coh=np.array([-2,-2,-2,-2,-1,-1,-1,0,1,1,1,2,2,2,2])
#group_coh=np.array([-2,-2,-2,-1,-1,-1,-1,0,1,1,1,1,2,2,2])
#group_coh=np.array([nan,-2 ,-2 ,-2 ,-1 ,-1 ,-1 ,nan,1  ,1  ,1  ,2  ,2  ,2  ,nan])
#group_coh=np.array([nan,nan,0  ,0  ,0  ,0  ,0  ,nan,1  ,1  ,1  ,1  ,1  ,nan,nan])
#group_coh=np.array([nan,0  ,0  ,0  ,0  ,0  ,0  ,nan,1  ,1  ,1  ,1  ,1  ,1  ,nan])
#group_coh=np.array([nan,0 ,0 ,0 ,1 ,1 ,1 ,nan,1  ,1  ,1  ,0  ,0  ,0  ,nan])
#group_coh=np.array([nan,nan,nan,nan,nan,nan,nan,nan,0  ,0  ,0  ,1  ,1  ,1  ,nan])
# group_coh_vec=np.array([[nan,nan,nan,nan,nan,nan,1  ,nan,0  ,nan,nan,nan,nan,nan,nan], #Diff
#                        [nan,nan,nan,nan,nan,1  ,nan,nan,nan,0  ,nan,nan,nan,nan,nan], #Diff
#                        [nan,nan,nan,nan,1  ,nan,nan,nan,nan,nan,0  ,nan,nan,nan,nan], #Diff
#                        [nan,nan,nan,1  ,nan,nan,nan,nan,nan,nan,nan,0  ,nan,nan,nan], #Easy
#                        [nan,nan,1  ,nan,nan,nan,nan,nan,nan,nan,nan,nan,0  ,nan,nan], #Easy
#                        [nan,1  ,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,0  ,nan]])  #Easy
#group_coh_vec=np.array([[nan,nan,nan,nan,1  ,1  ,1  ,nan,0  ,0  ,0  ,nan,nan,nan,nan], #Diff
#                        [nan,1  ,1  ,1  ,nan,nan,nan,nan,nan,nan,nan,0  ,0  ,0  ,nan]]) #Easy
group_coh_vec=np.array([[nan,0  ,0  ,0  ,0  ,0  ,0  ,nan,1  ,1  ,1  ,1  ,1  ,1  ,nan]])

tpre_sacc=50

#bias_vec=np.linspace(-3,3,31)
#bias_vec=np.linspace(-5,5,31)
#bias_vec=np.linspace(-7,7,31)
bias_vec=np.linspace(-10,10,31)
#bias_vec=np.linspace(-20,20,31) #Niels
#bias_vec=np.linspace(-15,15,31) #Galileo
#bias_vec=np.linspace(-1,1,31)


for k in range(len(monkeys)):
    print (monkeys[k])
    abs_path='/home/ramon/Dropbox/Esteki_Kiani/data/sorted/late/%s/'%(monkeys[k])
    files=miscellaneous.order_files(np.array(os.listdir(abs_path)))
    print (files)
    perf_all=nan*np.zeros((steps,len(group_coh_vec),n_rand,3))
    ccgp_all=nan*np.zeros((steps,len(group_coh_vec),n_rand,len(bias_vec),2,2))
    #inter_all=nan*np.zeros((n_rand,len(bias_vec),2,2))
    pseudo=miscellaneous.pseudopop_coherence_context_correct(abs_path,files,talig,dic_time,steps,thres,nt,n_rand,perc_tr,True,tpre_sacc,group_ref,shuff=False)
    
    for kk in range(steps):
        # Careful! in this function I am only using correct trials so that choice and stimulus are the same    
        pseudo_all=pseudo['pseudo_all'][kk]
        pseudo_tr=pseudo['pseudo_tr'][kk]
        pseudo_te=pseudo['pseudo_te'][kk]
        context=pseudo['clase_ctx']
        clase_all=pseudo['clase_all']
        coherence=pseudo['clase_coh']

        for j in range(len(group_coh_vec)):
            clase_coh=clase_resolution(group_coh_vec[j],coherence)
            indnan=~np.isnan(clase_coh) # Indices for the coherences we are going to use (True means to use, False to discard)

            feat_binary=nan*np.zeros((len(coherence),2))
            ind00=np.where((clase_coh==0)&(context==0))[0]
            ind01=np.where((clase_coh==0)&(context==1))[0]
            ind10=np.where((clase_coh==1)&(context==0))[0]
            ind11=np.where((clase_coh==1)&(context==1))[0]
            feat_binary[ind00]=np.array([0,0])
            feat_binary[ind01]=np.array([0,1])
            feat_binary[ind10]=np.array([1,0])
            feat_binary[ind11]=np.array([1,1])
    
            for ii in range(n_rand):
                #print (' ',ii)
                sum_nan=np.sum(np.isnan(pseudo_tr[ii]),axis=1)
                indnan_flat=(sum_nan==0) # True will be used, False discarded
                ind_nonan=(indnan*indnan_flat) # Index used combination of discarded from RT and discarded from group_coh
                #print ('Diff ',j,np.sum(ind_nonan))
                # Choice
                cl=LogisticRegression(C=1/reg,class_weight='balanced')
                #cl=LinearSVC(C=1/reg,class_weight='balanced')
                cl.fit(pseudo_tr[ii][ind_nonan],feat_binary[:,0][ind_nonan])
                perf_all[kk,j,ii,0]=cl.score(pseudo_te[ii][ind_nonan],feat_binary[:,0][ind_nonan])
                # Context
                cl=LogisticRegression(C=1/reg,class_weight='balanced')
                #cl=LinearSVC(C=1/reg,class_weight='balanced')
                cl.fit(pseudo_tr[ii][ind_nonan],feat_binary[:,1][ind_nonan])
                perf_all[kk,j,ii,1]=cl.score(pseudo_te[ii][ind_nonan],feat_binary[:,1][ind_nonan])
                # XOR
                cl=LogisticRegression(C=1/reg,class_weight='balanced')
                #cl=LinearSVC(C=1/reg,class_weight='balanced')
                xor=np.sum(feat_binary,axis=1)%2
                cl.fit(pseudo_tr[ii][ind_nonan],xor[ind_nonan])
                perf_all[kk,j,ii,2]=cl.score(pseudo_te[ii][ind_nonan],xor[ind_nonan])
                # CCGP
                for f in range(len(bias_vec)):
                    print (f)
                    ccgp=abstraction_2D(pseudo_all[ii][ind_nonan],feat_binary[ind_nonan],bias=bias_vec[f],reg=reg)
                    ccgp_all[kk,j,ii,f]=ccgp[0]
                    #inter_all[ii,f]=ccgp[1]


    perf_all_m=np.nanmean(perf_all,axis=2)
    perf_all_std=np.std(perf_all,axis=2)
    ccgp_all_m=np.nanmean(ccgp_all,axis=2)
    # inter_all_m=np.nanmean(inter_all,axis=(0,1))
    # ccgp_all_std=np.nanstd(ccgp_all,axis=0)
    #print (ccgp_all_m[15])
    #print (np.max(ccgp_all_m,axis=0))
    #print (inter_all_m)
    print (perf_all_m)
    
    # Plot performance Tasks and XOR vs time
    fig=plt.figure(figsize=(2.3,2))
    ax=fig.add_subplot(111)
    miscellaneous.adjust_spines(ax,['left','bottom'])
    ax.plot(xx,perf_all_m[:,0,0],color='blue',label='Choice')
    ax.fill_between(xx,perf_all_m[:,0,0]-perf_all_std[:,0,0],perf_all_m[:,0,0]+perf_all_std[:,0,0],color='blue',alpha=0.5)
    ax.plot(xx,perf_all_m[:,0,1],color='brown',label='Context')
    ax.fill_between(xx,perf_all_m[:,0,1]-perf_all_std[:,0,1],perf_all_m[:,0,1]+perf_all_std[:,0,1],color='brown',alpha=0.5)
    ax.plot(xx,perf_all_m[:,0,2],color='black',label='XOR')
    ax.fill_between(xx,perf_all_m[:,0,2]-perf_all_std[:,0,2],perf_all_m[:,0,2]+perf_all_std[:,0,2],color='black',alpha=0.5)
    ax.plot(xx,0.5*np.ones(len(xx)),color='black',linestyle='--')
    ax.set_ylim([0.4,1])
    ax.set_xlabel('Time (sec)')
    ax.set_ylabel('Decoding Performance')
    plt.legend(loc='best')
    #fig.savefig('/home/ramon/Dropbox/Esteki_Kiani/plots/choice_ctx_xor_time_pseudo_tl_%s_%s.pdf'%(talig,monkeys[k]),dpi=500,bbox_inches='tight')
 
    # # Plot Shifted CCGP
    # fig=plt.figure(figsize=(2.3,2))
    # ax=fig.add_subplot(111)
    # miscellaneous.adjust_spines(ax,['left','bottom'])
    # ax.plot(bias_vec,ccgp_all_m[:,0,0],color='blue')
    # ax.plot(bias_vec,ccgp_all_m[:,0,1],color='royalblue')
    # ax.plot(bias_vec,ccgp_all_m[:,1,0],color='brown')
    # ax.plot(bias_vec,ccgp_all_m[:,1,1],color='orange')
    # ax.plot(bias_vec,0.5*np.ones(len(bias_vec)),color='black',linestyle='--')
    # ax.axvline(0,color='black')
    # ax.set_ylim([0.4,1])
    # ax.set_ylabel('Shifted-CCGP')
    # ax.set_xlabel('Shift')
    # #plt.xticks(xx[indnan0],coh_plot[k][indnan0])
    # #plt.xticks(xx,coh_plot[1])
    # fig.savefig('/home/ramon/Dropbox/Esteki_Kiani/figures/figure_neuro_pseudo_shifted_ccgp_%s_reg1em3.pdf'%(monkeys[k]),dpi=500,bbox_inches='tight')
 
       
