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
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import KFold,StratifiedKFold,StratifiedShuffleSplit
from sklearn.neural_network import MLPClassifier
from scipy.optimize import curve_fit
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

def create_input(disp,pert,n_neu,n_trials,sigma):
    a_pre=np.array([[0,0],[0,1],[1,0],[1,1]],dtype=np.float64)
    a=a_pre.copy()
    a[1,0]=(a[1,0]+disp)
    a[3,0]=(a[3,0]+disp)
    #wei=np.random.normal(0,1,(2,n_neu))
    b_pre=np.zeros((4,n_neu))
    b_pre[:,0:2]=a
    wei=ortho_group.rvs(dim=n_neu)
    b=np.dot(b_pre,wei)

    # plt.scatter(a_pre[0:2,0],a_pre[0:2,1],color='black')
    # plt.scatter(a[0:2,0],a[0:2,1],color='blue')
    # plt.scatter(a[2:,0],a[2:,1],color='blue',alpha=0.5)
    # plt.scatter(b[0:2,0],b[0:2,1],color='green')
    # plt.scatter(b[2:,0],b[2:,1],color='green',alpha=0.5)
    # plt.xlim([-5,5])
    # plt.ylim([-5,5])
    # plt.show()

    b_pert=np.random.normal(b,pert,np.shape(b))
   
    inp_tr=nan*np.zeros((4*n_trials,n_neu))
    inp_te=nan*np.zeros((4*n_trials,n_neu))
    feat_binary=nan*np.zeros((4*n_trials,2))
    for i in range(4):
        inp_tr[i*n_trials:(i+1)*n_trials]=np.random.normal(b_pert[i],sigma,(n_trials,n_neu))
        inp_te[i*n_trials:(i+1)*n_trials]=np.random.normal(b_pert[i],sigma,(n_trials,n_neu))
        feat_binary[i*n_trials:(i+1)*n_trials]=np.array(a_pre[i],dtype=np.int16)
        
    return inp_tr,inp_te,feat_binary

def abstraction_2D(feat_decod,feat_binary,bias,reg):
    exp_uq=np.unique(feat_binary,axis=0)
    feat_binary_exp=np.zeros(len(feat_binary))
    for t in range(len(feat_binary)):
        for tt in range((len(exp_uq))):
            gg=(np.sum(feat_binary[t]==exp_uq[tt])==len(feat_binary[0]))
            if gg:
                feat_binary_exp[t]=tt
                
    dichotomies=np.array([[0,0,1,1],[0,1,0,1]])
    train_dich=np.array([[[0,2],[1,3]],[[0,1],[2,3]]])
    test_dich=np.array([[[1,3],[0,2]],[[2,3],[0,1]]])
    
    perf=nan*np.zeros((len(dichotomies),len(train_dich[0])))
    for k in range(len(dichotomies)): #Loop on "dichotomies"
      for kk in range(len(train_dich[0])): #Loop on ways to train this particular "dichotomy"
         ind_train=np.where((feat_binary_exp==train_dich[k][kk][0])|(feat_binary_exp==train_dich[k][kk][1]))[0]
         ind_test=np.where((feat_binary_exp==test_dich[k][kk][0])|(feat_binary_exp==test_dich[k][kk][1]))[0]

         task=nan*np.zeros(len(feat_binary_exp))
         for i in range(4):
             ind_task=(feat_binary_exp==i)
             task[ind_task]=dichotomies[k][i]

         supp=LogisticRegression(C=1/reg,class_weight='balanced')
         mod=supp.fit(feat_decod[ind_train],task[ind_train])
         pred=(np.dot(feat_decod[ind_test],supp.coef_[0])+supp.intercept_+bias)>0
         perf[k,kk]=np.mean(pred==task[ind_test])
         
    return perf

############################################

n_neu=10
n_trials=10
sigma=0.4
#disp=0
disp_vec=np.linspace(0,1,10)
#pert_vec=np.array(np.linspace(0,5,20))
#pert_vec=[0]
pert=0.1
n_files=50
reg=1e0
n_bias=50
bias_vec=np.linspace(-10,10,n_bias)

# perf_pre=nan*np.zeros((len(pert_vec),n_files,3))
# ccgp_pre=nan*np.zeros((len(pert_vec),n_files,len(bias_vec),2,2))
# shccgp_pre=nan*np.zeros((len(pert_vec),n_files,2,2))

perf_pre=nan*np.zeros((len(disp_vec),n_files,3))
ccgp_pre=nan*np.zeros((len(disp_vec),n_files,len(bias_vec),2,2))
shccgp_pre=nan*np.zeros((len(disp_vec),n_files,2,2))

for j in range(len(disp_vec)):
    print (disp_vec[j])
    for i in range(n_files):
        inp_tr,inp_te,feat_binary=create_input(disp_vec[j],pert,n_neu,n_trials,sigma)
        
        # Var 1
        cl=LogisticRegression(C=1/reg,class_weight='balanced')
        cl.fit(inp_tr,feat_binary[:,0])
        perf_pre[j,i,0]=cl.score(inp_te,feat_binary[:,0])
        # Var 2
        cl=LogisticRegression(C=1/reg,class_weight='balanced')
        cl.fit(inp_tr,feat_binary[:,1])
        perf_pre[j,i,1]=cl.score(inp_te,feat_binary[:,1])
        # XOR
        cl=LogisticRegression(C=1/reg,class_weight='balanced')
        xor=np.sum(feat_binary,axis=1)%2
        cl.fit(inp_tr,xor)
        perf_pre[j,i,2]=cl.score(inp_te,xor)

        #print (perf_pre[j,i])

        for f in range(len(bias_vec)):
            ccgp_pre[j,i,f]=abstraction_2D(inp_te,feat_binary,bias=bias_vec[f],reg=reg)

        shccgp_pre[j,i,0,0]=np.max(ccgp_pre[j,i,:,0,0])
        shccgp_pre[j,i,0,1]=np.max(ccgp_pre[j,i,:,0,1])
        shccgp_pre[j,i,1,0]=np.max(ccgp_pre[j,i,:,1,0])
        shccgp_pre[j,i,1,1]=np.max(ccgp_pre[j,i,:,1,1])

        # plt.plot(bias_vec,ccgp_pre[j,i,:,0,0],color='blue')
        # plt.plot(bias_vec,ccgp_pre[j,i,:,0,1],color='royalblue')
        # plt.plot(bias_vec,ccgp_pre[j,i,:,1,0],color='brown')
        # plt.plot(bias_vec,ccgp_pre[j,i,:,1,1],color='orange')
        # plt.ylim([0,1])
        # plt.show()

perf_m=np.mean(perf_pre,axis=1)
ccgp_m=np.mean(ccgp_pre,axis=1)
shccgp_m=np.mean(shccgp_pre,axis=1)

print (perf_m)

# plt.plot(pert_vec,perf_m[:,0],color='blue')
# plt.plot(pert_vec,perf_m[:,1],color='brown')
# plt.plot(pert_vec,perf_m[:,2],color='black')
# plt.plot(pert_vec,np.mean(shccgp_m[:,0],axis=1),color='blue',linestyle='--')
# plt.plot(pert_vec,np.mean(shccgp_m[:,1],axis=1),color='brown',linestyle='--')
# plt.plot(pert_vec,np.mean(ccgp_m[:,15,0],axis=1),color='royalblue',linestyle='--')
# plt.plot(pert_vec,np.mean(ccgp_m[:,15,1],axis=1),color='orange',linestyle='--')
# plt.show()

plt.plot(disp_vec,perf_m[:,0],color='blue')
plt.plot(disp_vec,perf_m[:,1],color='brown')
plt.plot(disp_vec,perf_m[:,2],color='black')
plt.plot(disp_vec,np.mean(shccgp_m[:,0],axis=1),color='blue',linestyle='--')
plt.plot(disp_vec,np.mean(shccgp_m[:,1],axis=1),color='brown',linestyle='--')
plt.plot(disp_vec,np.mean(ccgp_m[:,int(n_bias/2),0],axis=1),color='royalblue',linestyle='--')
plt.plot(disp_vec,np.mean(ccgp_m[:,int(n_bias/2),1],axis=1),color='orange',linestyle='--')
plt.ylim([0.4,1])
plt.show()


    
            
