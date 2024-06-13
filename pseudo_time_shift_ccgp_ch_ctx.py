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

def abstraction_2D_bias(feat_decod,feat_binary,bias_vec,reg):
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
    
    perf=nan*np.zeros((len(bias_vec),len(dichotomies),len(train_dich[0])))
    parallel=nan*np.zeros(len(dichotomies))
    for k in range(len(dichotomies)): #Loop on "dichotomies"
        para=nan*np.zeros((len(train_dich[0]),len(feat_decod[0])))
        for kk in range(len(train_dich[0])): #Loop on ways to train this particular "dichotomy"
            ind_train=np.where((feat_binary_exp==train_dich[k][kk][0])|(feat_binary_exp==train_dich[k][kk][1]))[0]
            ind_test=np.where((feat_binary_exp==test_dich[k][kk][0])|(feat_binary_exp==test_dich[k][kk][1]))[0]

            task=nan*np.zeros(len(feat_binary_exp))
            for i in range(4):
                ind_task=(feat_binary_exp==i)
                task[ind_task]=dichotomies[k][i]

            supp=LogisticRegression(C=1/reg,class_weight='balanced')
            mod=supp.fit(feat_decod[ind_train],task[ind_train])
            para[kk]=supp.coef_[0]
            for j in range(len(bias_vec)):
                pred=(np.dot(feat_decod[ind_test],supp.coef_[0])+supp.intercept_+bias_vec[j])>0
                perf[j,k,kk]=np.mean(pred==task[ind_test])
        parallel[k]=np.dot(para[0],para[1])/(np.linalg.norm(para[0])*np.linalg.norm(para[1]))
    return perf,parallel

def rotation_indices(n_rot,n_cat,n_neu):
    ind=nan*np.zeros((n_rot,n_cat,n_neu))
    for k in range(n_rot):
        for i in range(n_cat):
            ind[k,i]=np.random.permutation(np.arange(n_neu))
    return np.array(ind,dtype=np.int16)

def create_rot(pseudo,ind_cat,ind_rot):
    pseudo_rot=nan*np.zeros(pseudo.shape) # trials x neurons
    n_cat=len(ind_rot)
    for i in range(n_cat):
        pseudo_rot[ind_cat[i]]=pseudo[ind_cat[i]][:,ind_rot[i]]
    return pseudo_rot

##############################################

def calculate_everything(monkey,group_coh_vec,bias_vec,abs_path,files,talig,dic_time,steps,thres,nt,n_rand,n_rot,perc_tr,tpre_sacc,group_ref,shuff):
    perf_all=nan*np.zeros((steps,n_rand,3))
    ccgp_all=nan*np.zeros((steps,n_rand,len(bias_vec),2,2))
    ccgp_rot_all=nan*np.zeros((n_rot,steps,n_rand,len(bias_vec),2,2))
    
    pseudo=miscellaneous.pseudopop_coherence_context_correct(abs_path,files,talig,dic_time,steps,thres,nt,n_rand,perc_tr,tpre_sacc,group_ref,shuff,learning=True)
    for kk in range(steps):
        print (kk)
        # Careful! in this function I am only using correct trials so that choice and stimulus are the same    
        pseudo_all=pseudo['pseudo_all'][kk]
        pseudo_tr=pseudo['pseudo_tr'][kk]
        pseudo_te=pseudo['pseudo_te'][kk]
        context=pseudo['clase_ctx']
        clase_all=pseudo['clase_all']
        coherence=pseudo['clase_coh']
        neu_total=pseudo_tr.shape[-1]
        
        clase_coh=clase_resolution(group_coh_vec,coherence)
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
       
        index_cat=np.array([ind00,ind01,ind10,ind11])
        index_rot=rotation_indices(n_rot=n_rot,n_cat=len(index_cat),n_neu=neu_total)
        
        for ii in range(n_rand):
            #print (' ',ii)
            sum_nan=np.sum(np.isnan(pseudo_tr[ii]),axis=1)
            #np.set_printoptions(threshold=sys.maxsize)
            #print (sum_nan)
            indnan_flat=(sum_nan==0) # True will be used, False discarded
            ind_nonan=(indnan*indnan_flat) # Index used combination of discarded from RT and discarded from group_coh
            
            if monkey=='Niels':
                neu_rnd=np.arange(neu_total)
            if monkey=='Galileo':
                n_max=96*4 # 96 channels, 4 files
                neu_rnd=np.sort(np.random.choice(np.arange(neu_total),n_max,replace=False)) # Careful!!!
            
            # Choice
            cl=LogisticRegression(C=1/reg,class_weight='balanced')
            #cl=LinearSVC(C=1/reg,class_weight='balanced')
            cl.fit(pseudo_tr[ii][ind_nonan][:,neu_rnd],feat_binary[:,0][ind_nonan])
            perf_all[kk,ii,0]=cl.score(pseudo_te[ii][ind_nonan][:,neu_rnd],feat_binary[:,0][ind_nonan])
            # Context
            cl=LogisticRegression(C=1/reg,class_weight='balanced')
            #cl=LinearSVC(C=1/reg,class_weight='balanced')
            cl.fit(pseudo_tr[ii][ind_nonan][:,neu_rnd],feat_binary[:,1][ind_nonan])
            perf_all[kk,ii,1]=cl.score(pseudo_te[ii][ind_nonan][:,neu_rnd],feat_binary[:,1][ind_nonan])
            # XOR
            cl=LogisticRegression(C=1/reg,class_weight='balanced')
            #cl=LinearSVC(C=1/reg,class_weight='balanced')
            xor=np.sum(feat_binary,axis=1)%2
            cl.fit(pseudo_tr[ii][ind_nonan][:,neu_rnd],xor[ind_nonan])
            perf_all[kk,ii,2]=cl.score(pseudo_te[ii][ind_nonan][:,neu_rnd],xor[ind_nonan])
            
            # CCGP
            #for f in range(len(bias_vec)):
            ccgp=abstraction_2D_bias(pseudo_all[ii][ind_nonan][:,neu_rnd],feat_binary[ind_nonan],bias_vec=bias_vec,reg=reg)
            ccgp_all[kk,ii]=ccgp[0]

            # Distribution of ccgp after breaking geometry through rotations
            for n in range(n_rot):
                #print ('rot ',n)
                pseudo_rot=create_rot(pseudo_all[ii],index_cat,index_rot[n])
                #for f in range(len(bias_vec)):
                ccgp_rot=abstraction_2D_bias(pseudo_rot[ind_nonan][:,neu_rnd],feat_binary[ind_nonan],bias_vec=bias_vec,reg=reg)
                ccgp_rot_all[n,kk,ii]=ccgp_rot[0]

    return perf_all,ccgp_all,ccgp_rot_all

##########################################################

monkey='Galileo'

talig='dots_on' #'response_edf' #dots_on

nt=100 #100 for coh signed, 200 for coh unsigned, 50 for coh signed with context
n_rand=20
n_shuff=0
perc_tr=0.8
thres=0
reg=1e2
n_coh=15
tpre_sacc=50
n_rot=0

group_ref=np.array([-7 ,-6 ,-5 ,-4 ,-3 ,-2 ,-1 ,0  ,1  ,2  ,3  ,4  ,5  ,6  ,7  ])

if monkey=='Niels':
    group_coh_vec=np.array([nan,0  ,0  ,0  ,0  ,0  ,0  ,nan,1  ,1  ,1  ,1  ,1  ,1  ,nan])
    bias_vec=np.linspace(-10,10,31) #Niels
    dic_time=np.array([0,450,200,50]) # time pre, time post, bin size, step size
    ind_l=8
    ind_u=12
if monkey=='Galileo':
    group_coh_vec=np.array([nan,0  ,0  ,0  ,0  ,0  ,0  ,nan,1  ,1  ,1  ,1  ,1  ,1  ,nan]) # Careful!
    bias_vec=np.linspace(-10,10,31) #Galileo
    dic_time=np.array([0,650,200,50]) # Careful! time pre, time post, bin size, step size
    ind_l=20
    ind_u=30

steps=int((dic_time[0]+dic_time[1])/dic_time[3])
xx=np.linspace(-dic_time[0]/1000,dic_time[1]/1000,steps,endpoint=False)
print (steps)
print (xx)

#abs_path='/home/ramon/Dropbox/Esteki_Kiani/data/sorted/late/%s/'%(monkeys[k])
abs_path='/home/ramon/Dropbox/Proyectos_Postdoc/Esteki_Kiani/data/unsorted/%s/'%(monkey)
files_pre=np.array(os.listdir(abs_path))
order=miscellaneous.order_files(files_pre)
files=np.array(files_pre[order])[ind_l:ind_u]
print (files)

# Original
perf_all,ccgp_all,ccgp_rot_all=calculate_everything(monkey,group_coh_vec,bias_vec,abs_path,files,talig,dic_time,steps,thres,nt,n_rand,n_rot,perc_tr,tpre_sacc,group_ref,shuff=False)

# Plot as a function of time
for t in range(steps):
    fig=plt.figure(figsize=(2.3,2))
    ax=fig.add_subplot(111)
    miscellaneous.adjust_spines(ax,['left','bottom'])
    ax.plot(bias_vec,np.mean(ccgp_all[t,:,:,0,0],axis=0),color='royalblue',label='Sh-CCGP Direction 1')
    ax.fill_between(bias_vec,np.mean(ccgp_all[t,:,:,0,0],axis=0)-np.std(ccgp_all[t,:,:,0,0],axis=0),np.mean(ccgp_all[t,:,:,0,0],axis=0)+np.std(ccgp_all[t,:,:,0,0],axis=0),color='royalblue',alpha=0.5)
    ax.plot(bias_vec,np.mean(ccgp_all[t,:,:,0,1],axis=0),color='blue',label='Sh-CCGP Direction 2')
    ax.fill_between(bias_vec,np.mean(ccgp_all[t,:,:,0,1],axis=0)-np.std(ccgp_all[t,:,:,0,1],axis=0),np.mean(ccgp_all[t,:,:,0,1],axis=0)+np.std(ccgp_all[t,:,:,0,1],axis=0),color='blue',alpha=0.5)
    ax.plot(bias_vec,np.mean(ccgp_all[t,:,:,1,0],axis=0),color='orange',label='Sh-CCGP Context 1')
    ax.fill_between(bias_vec,np.mean(ccgp_all[t,:,:,1,0],axis=0)-np.std(ccgp_all[t,:,:,1,0],axis=0),np.mean(ccgp_all[t,:,:,1,0],axis=0)+np.std(ccgp_all[t,:,:,1,0],axis=0),color='orange',alpha=0.5)
    ax.plot(bias_vec,np.mean(ccgp_all[t,:,:,1,1],axis=0),color='brown',label='Sh-CCGP Context 2')
    ax.fill_between(bias_vec,np.mean(ccgp_all[t,:,:,1,1],axis=0)-np.std(ccgp_all[t,:,:,1,1],axis=0),np.mean(ccgp_all[t,:,:,1,1],axis=0)+np.std(ccgp_all[t,:,:,1,1],axis=0),color='brown',alpha=0.5)
    ax.plot(bias_vec,0.5*np.ones(len(bias_vec)),color='black',linestyle='--')
    ax.set_ylim([0.4,1])
    ax.set_xlabel('Bias')
    ax.set_ylabel('Decoding Performance')
    plt.legend(loc='best')
    fig.savefig('/home/ramon/Dropbox/Proyectos_Postdoc/Esteki_Kiani/plots/shifted_ccgp_tl_%s_%s_t%i.pdf'%(talig,monkey,t),dpi=500,bbox_inches='tight')


# # Parallelism Score
# fig=plt.figure(figsize=(2.3,2))
# ax=fig.add_subplot(111)
# miscellaneous.adjust_spines(ax,['left','bottom'])
# ax.plot(xx,parallel_m[:,0],color='blue',label='Parallelism Direction')
# ax.fill_between(xx,parallel_m[:,0]-parallel_std[:,0],parallel_m[:,0]+parallel_std[:,0],color='blue',alpha=0.5)
# ax.plot(xx,parallel_m[:,1],color='brown',label='Parallelism Context')
# ax.fill_between(xx,parallel_m[:,1]-parallel_std[:,1],parallel_m[:,1]+parallel_std[:,1],color='brown',alpha=0.5)
# # Shuffled
# ax.fill_between(xx,parallel_null[ind_null,:,0],parallel_null[-ind_null,:,0],color='blue',alpha=0.3)
# ax.fill_between(xx,parallel_null[ind_null,:,1],parallel_null[-ind_null,:,1],color='brown',alpha=0.3)
# ax.plot(xx,np.zeros(len(xx)),color='black',linestyle='--')
# ax.set_ylim([-0.15,0.55])
# ax.set_xlabel('Time (sec)')
# ax.set_ylabel('Parallelism Score')
# plt.legend(loc='best')
# fig.savefig('/home/ramon/Dropbox/Esteki_Kiani/plots/pseudo_time_parallelism_ch_ctx_%s_%s.pdf'%(talig,monkey),dpi=500,bbox_inches='tight')

 
       
