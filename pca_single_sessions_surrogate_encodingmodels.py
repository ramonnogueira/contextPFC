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
from sklearn.manifold import MDS
from sklearn.decomposition import PCA
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
from sklearn.neural_network import MLPRegressor
np.set_printoptions(suppress=True)

def lin_comp_task(data,feat_bin):
    n_cv=5
    reg=1
    perf=nan*np.zeros((3,n_cv))
    # Linear Task
    for i in range(2):
        skf=StratifiedKFold(n_splits=n_cv)
        g=-1
        for train, test in skf.split(data,feat_bin[:,i]):
            g=(g+1)
            cl=LogisticRegression(C=1/reg,class_weight='balanced')
            cl.fit(data[train],feat_bin[:,i][train])
            perf[i,g]=cl.score(data[test],feat_bin[:,i][test])
    # Complex Task
    xor=np.sum(feat_bin,axis=1)%2
    skf=StratifiedKFold(n_splits=n_cv)
    g=-1
    for train, test in skf.split(data,xor):
        g=(g+1)
        cl=LogisticRegression(C=1/reg,class_weight='balanced')
        cl.fit(data[train],xor[train])
        perf[2,g]=cl.score(data[test],xor[test])
    return np.mean(perf,axis=1)

def abstraction_2D(feat_decod,feat_binary,reg):
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
    
    perf=nan*np.zeros((len(dichotomies),len(train_dich[0]),2))
    for k in range(len(dichotomies)): #Loop on "dichotomies"
      for kk in range(len(train_dich[0])): #Loop on ways to train this particular "dichotomy"
         ind_train=np.where((feat_binary_exp==train_dich[k][kk][0])|(feat_binary_exp==train_dich[k][kk][1]))[0]
         ind_test=np.where((feat_binary_exp==test_dich[k][kk][0])|(feat_binary_exp==test_dich[k][kk][1]))[0]

         task=nan*np.zeros(len(feat_binary_exp))
         for i in range(4):
             ind_task=(feat_binary_exp==i)
             task[ind_task]=dichotomies[k][i]

         supp=LogisticRegression(C=reg,class_weight='balanced')
         #supp=LinearSVC(C=reg,class_weight='balanced')
         mod=supp.fit(feat_decod[ind_train],task[ind_train])
         perf[k,kk,0]=supp.score(feat_decod[ind_train],task[ind_train])
         perf[k,kk,1]=supp.score(feat_decod[ind_test],task[ind_test])
    return perf

def class_twovars(data,var1,var2):
    n_rand=10
    n_cv=5
    reg=1
    perf=nan*np.zeros((n_rand,n_cv,2))
    uq1=np.unique(var1)
    uq2=np.unique(var2)
    for i in range(n_rand):
        ind11=np.where((var1==uq1[0])&(var2==uq2[0]))[0]
        ind12=np.where((var1==uq1[0])&(var2==uq2[1]))[0]
        ind21=np.where((var1==uq1[1])&(var2==uq2[0]))[0]
        ind22=np.where((var1==uq1[1])&(var2==uq2[1]))[0]
        mint=np.min(np.array([len(ind11),len(ind12),len(ind21),len(ind22)]))
        ind_all=[ind11,ind12,ind21,ind22]
        class_all=np.array([[0,0],[0,1],[1,0],[1,1]])
        # Create dataset
        data_r=nan*np.zeros((4*mint,len(data[0])))
        clas_r=np.zeros((4*mint,2),dtype=np.int16)
        for ii in range(4):
            ind_r=np.random.choice(ind_all[ii],mint,replace=False)
            data_r[ii*(mint):(ii+1)*mint]=data[ind_r]
            clas_r[ii*(mint):(ii+1)*mint]=class_all[ii]
        # Decode Var1
        skf=StratifiedKFold(n_splits=n_cv)
        g=-1
        for train, test in skf.split(data_r,clas_r[:,0]):
            g=(g+1)
            cl=LogisticRegression(C=1/reg)
            cl.fit(data_r[train],clas_r[train][:,0])
            perf[i,g,0]=cl.score(data_r[test],clas_r[test][:,0])
        # Decode Var2
        skf=StratifiedKFold(n_splits=n_cv)
        g=-1
        for train, test in skf.split(data_r,clas_r[:,1]):
            g=(g+1)
            cl=LogisticRegression(C=1/reg)
            cl.fit(data_r[train],clas_r[train][:,1])
            perf[i,g,1]=cl.score(data_r[test],clas_r[test][:,1])
    return np.mean(perf,axis=(0,1))


def score(y_pred,y_true):
    n_neu=len(y_pred[0])
    var=np.var(y_true,axis=0)
    llh=np.mean((y_true-y_pred)**2,axis=0)
    return np.ones(n_neu)-llh/var

def normalize_feat(feat):
    feat_norm=nan*np.zeros(np.shape(feat))
    for i in range(len(feat[0])):
        mean=np.mean(feat[:,i])
        std=np.std(feat[:,i])
        feat_norm[:,i]=(feat[:,i]-mean)/std
    return feat_norm

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

################################################
nan=float('nan')
minf=float('-inf')
pinf=float('inf')
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

monkey='Galileo'
aligned='dots_on'
if monkey=='Niels':
    dic_time=np.array([0,600,200,200]) # time pre, time post, bin size, step size
if monkey=='Galileo':
    dic_time=np.array([0,800,200,200]) # time pre, time post, bin size, step size
steps=int((dic_time[0]+dic_time[1])/dic_time[3])
xx=np.linspace(-dic_time[0]/1000,dic_time[1]/1000,steps,endpoint=False)
print (xx)

# Main figure
# PCA in main figure was done with surrogate reprs
# Niels session 2, all trials and threshold firing rate of 1

# Supplementary figure
# Only correct trials
# Thershold =1, surrogate data with model 2

tpre_sacc=50 # To avoid RT-contaminated trials this should be positive

# parameters that work great: tpre_sacc 50, time bin 100, reg 1e-3, lr 1e-3, surrogate from model 1hidden layer 100, per_tr > 0.25.

group_coh=np.array([-7 ,-6 ,-5 ,-4 ,-3 ,-2 ,-1 ,0  ,1  ,2  ,3  ,4  ,5  ,6  ,7  ]) #15

col=np.array(['darkgreen','darkgreen','darkgreen','darkgreen','darkgreen','darkgreen','darkgreen','black','darkgoldenrod','darkgoldenrod','darkgoldenrod','darkgoldenrod','darkgoldenrod','darkgoldenrod','darkgoldenrod','purple','purple','purple','purple','purple','purple','purple','black','darkblue','darkblue','darkblue','darkblue','darkblue','darkblue','darkblue'])
alph=np.array([0.7,0.6,0.5,0.4,0.3,0.2,0.1,1,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.7,0.6,0.5,0.4,0.3,0.2,0.1,1,0.1,0.2,0.3,0.4,0.5,0.6,0.7])

abs_path='/home/ramon/Dropbox/Proyectos_Postdoc/Esteki_Kiani/data/sorted/late/%s/'%(monkey)
#abs_path='/home/ramon/Dropbox/Proyectos_Postdoc/Esteki_Kiani/data/unsorted/%s/'%(monkey) 
files=os.listdir(abs_path)

for kk in range(len(files)):
    print (files[kk])
    data=scipy.io.loadmat(abs_path+'%s'%(files[kk]),struct_as_record=False,simplify_cells=True)
    beha=miscellaneous.behavior(data,group_coh)
    index_nonan=beha['index_nonan']
    reward=beha['reward']
    ind_corr=np.where(reward==1)[0]
    coherence=beha['coherence_signed'][ind_corr] #Cuidado!!
    coh_num=beha['coh_num'][ind_corr] #Cuidado!!
    context=beha['context'][ind_corr] #Cuidado!!
    stimulus=beha['stimulus'][ind_corr] #Cuidado!!
    choice=beha['choice'][ind_corr] #Cuidado!!
    choice_uq=np.unique(choice)
    stim_uq=np.unique(stimulus)
    ctx_uq=np.unique(context)
    coh_uq=np.unique(coherence)
    rt=beha['reaction_time'][ind_corr] #Cuidado!!
    firing_rate_pre=miscellaneous.getRasters(data,aligned,dic_time,index_nonan,threshold=1) #Careful with threshold!
    #firing_rate_pre=miscellaneous.getRasters_unsorted(data,aligned,dic_time,index_nonan,threshold=1) #Careful with threshold!
    firing_rate=miscellaneous.normalize_fr(firing_rate_pre)[ind_corr] #Cuidado!!
    print (np.shape(firing_rate))
    fr_norm=miscellaneous.normalize_fr(firing_rate) # Trials x neurons x time_steps
    num_neu=len(firing_rate[0])

    ######################################################
    # Encoding Models
    features=nan*np.zeros((len(fr_norm),2))
    features[:,0]=coh_num
    features[:,1]=context
    feat_norm=normalize_feat(features)
    
    n_cv=10
    tr_size=0.75
    activation='relu'
    lr=1e-3
    reg=1e-3
    models_vec=[(),(100),(100,100),(100,100,100)]
    print ('LR ',lr,'REG ',reg)
    
    r2_vec=nan*np.zeros((len(models_vec)+1,steps,n_cv,2))
    fr_surr_pre=nan*np.zeros((len(models_vec)+1,len(fr_norm),num_neu,steps))
    for j in range(steps):
        print (j)
        # Linear Regression
        #cv=KFold(n_splits=n_cv)
        cv=ShuffleSplit(n_splits=n_cv,train_size=tr_size) #Try KFold instead??
        g=-1
        for train_index, test_index in cv.split(fr_norm[:,:,j]):
            g=(g+1)
            cl=LinearRegression()
            cl.fit(feat_norm[train_index],fr_norm[:,:,j][train_index])
            y_pred_tr=cl.predict(feat_norm[train_index])
            sc_tr=score(y_pred_tr,fr_norm[:,:,j][train_index])
            r2_vec[0,j,g,0]=np.mean(sc_tr[abs(sc_tr)<1000])
            y_pred=cl.predict(feat_norm[test_index])
            fr_surr_pre[0,test_index,:,j]=y_pred
            sc=score(y_pred,fr_norm[:,:,j][test_index])
            r2_vec[0,j,g,1]=np.mean(sc[abs(sc)<1000])
    
        for l in range(len(models_vec)):
            #cv=KFold(n_splits=n_cv)
            cv=ShuffleSplit(n_splits=n_cv,train_size=tr_size)
            g=-1
            for train_index, test_index in cv.split(fr_norm[:,:,j]):
                g=(g+1)
                cl=MLPRegressor(hidden_layer_sizes=models_vec[l],activation=activation,learning_rate_init=lr,alpha=reg)
                cl.fit(feat_norm[train_index],fr_norm[:,:,j][train_index])
                y_pred_tr=cl.predict(feat_norm[train_index])
                sc_tr=score(y_pred_tr,fr_norm[:,:,j][train_index]) 
                r2_vec[l+1,j,g,0]=np.mean(sc_tr[abs(sc_tr)<1000])
                y_pred=cl.predict(feat_norm[test_index])
                fr_surr_pre[l,test_index,:,j]=y_pred # no hace falta CV porque va rellenando solo los huecos del test-set
                sc=score(y_pred,fr_norm[:,:,j][test_index]) 
                r2_vec[l+1,j,g,1]=np.mean(sc[abs(sc)<1000])
    print ('Train ',np.mean(r2_vec,axis=2)[:,:,0],np.mean(r2_vec,axis=(1,2))[:,0])
    print ('Test ',np.mean(r2_vec,axis=2)[:,:,1],np.mean(r2_vec,axis=(1,2))[:,1])
    
    # Create surrogate data
    fr_surr=fr_surr_pre[3] # Careful!!!
    fr_surr[np.isnan(fr_surr)]=0 #Due to the ShuffleSplit sometimes a particular trial is not filled, but they are few

    ##########################################################
    # USE THIS FOR PCA IN THE FIGURES
    # Train PCA all coherences
    # mean_coh_pre=nan*np.zeros((steps*2*len(coh_uq),num_neu))
    # len_tr=nan*np.zeros((steps*2*len(coh_uq)))
    # per_tr=nan*np.zeros((steps*2*len(coh_uq)))
    
    # for j in range(steps):
    #     min_t=(j*dic_time[3]+dic_time[2]-dic_time[0]+tpre_sacc) # for each time step temporal threshold at which the RT needs to be bigger than that
    #     for jj in range(len(coh_uq)):
    #         ind_ctx0_pre=np.where((coherence==coh_uq[jj])&(context==0))[0]
    #         ind_ctx1_pre=np.where((coherence==coh_uq[jj])&(context==1))[0]
    #         ind_ctx0=np.where((coherence==coh_uq[jj])&(context==0)&(rt>min_t))[0]
    #         ind_ctx1=np.where((coherence==coh_uq[jj])&(context==1)&(rt>min_t))[0]
    #         per_tr[jj*steps+j]=len(ind_ctx0)/len(ind_ctx0_pre)
    #         per_tr[jj*steps+j+len(coh_uq)*steps]=len(ind_ctx1)/len(ind_ctx1_pre)
    #         len_tr[jj*steps+j]=len(ind_ctx0)
    #         len_tr[jj*steps+j+len(coh_uq)*steps]=len(ind_ctx1)
    #         #mean_coh_pre[jj*steps+j]=np.mean(firing_rate[ind_ctx0][:,:,j],axis=0)
    #         #mean_coh_pre[jj*steps+j+len(coh_uq)*steps]=np.mean(firing_rate[ind_ctx1][:,:,j],axis=0)
    #         #mean_coh_pre[jj*steps+j]=np.mean(fr_norm[ind_ctx0][:,:,j],axis=0)
    #         #mean_coh_pre[jj*steps+j+len(coh_uq)*steps]=np.mean(fr_norm[ind_ctx1][:,:,j],axis=0)
    #         mean_coh_pre[jj*steps+j]=np.mean(fr_surr[ind_ctx0][:,:,j],axis=0)
    #         mean_coh_pre[jj*steps+j+len(coh_uq)*steps]=np.mean(fr_surr[ind_ctx1][:,:,j],axis=0)
                
    # ind_tr=(per_tr>=0.25)*(len_tr>=10)
    # print (per_tr)
    # mean_coh=mean_coh_pre[ind_tr]
    
    # embedding=PCA(n_components=3)
    # fitPCA=embedding.fit(mean_coh)
    # print (fitPCA.explained_variance_ratio_)

    # # Plot as a function of time
    # for j in range(steps)[::-1]:
    #     print (j)
    #     plt.rcParams.update({'font.size': 15})
    #     fig = plt.figure()
    #     ax = fig.add_subplot(111, projection='3d')
    #     ind_step=(np.arange(0,steps*2*len(coh_uq),steps)+j)
    #     for jj in np.arange(2*len(coh_uq))[ind_tr[ind_step]]:
    #         pseudo_mds_ctx=embedding.transform(mean_coh_pre[ind_step][jj:(jj+1)])
    #         ax.scatter(pseudo_mds_ctx[0,0],pseudo_mds_ctx[0,1],pseudo_mds_ctx[0,2],color=col[jj],alpha=alph[jj],s=100)
    #     ax.set_xlabel('PC1')
    #     ax.set_ylabel('PC2')
    #     ax.set_zlabel('PC3')
    #     ax.set_xlim([-3,3])
    #     ax.set_ylim([-3,3])
    #     ax.set_zlim([-3,3])
    #     plt.show()
    #     plt.close(fig)

    #########################################################
    # # Train PCA on only 4 conditions
    # mean_coh_pre=nan*np.zeros((steps*2*len(stim_uq),num_neu))
    # len_tr=nan*np.zeros((steps*2*len(stim_uq)))
    # per_tr=nan*np.zeros((steps*2*len(stim_uq)))
    
    # for j in range(steps):
    #     min_t=(j*dic_time[3]+dic_time[2]-dic_time[0]+tpre_sacc) # for each time step temporal threshold at which the RT needs to be bigger than that
    #     print (min_t)
    #     for jj in range(len(stim_uq)):
    #         ind_ctx0_pre=np.where((choice==jj)&(context==0)&(abs(coherence)!=0.75))[0]
    #         ind_ctx1_pre=np.where((choice==jj)&(context==1)&(abs(coherence)!=0.75))[0]
    #         ind_ctx0=np.where((choice==jj)&(context==0)&(rt>min_t)&(abs(coherence)!=0.75))[0]
    #         ind_ctx1=np.where((choice==jj)&(context==1)&(rt>min_t)&(abs(coherence)!=0.75))[0]
    #         # ind_ctx0_pre=np.where((stimulus==jj)&(context==0)&(abs(coherence)!=0.75))[0]
    #         # ind_ctx1_pre=np.where((stimulus==jj)&(context==1)&(abs(coherence)!=0.75))[0]
    #         # ind_ctx0=np.where((stimulus==jj)&(context==0)&(rt>min_t)&(abs(coherence)!=0.75))[0]
    #         # ind_ctx1=np.where((stimulus==jj)&(context==1)&(rt>min_t)&(abs(coherence)!=0.75))[0]
    #         per_tr[jj*steps+j]=len(ind_ctx0)/len(ind_ctx0_pre)
    #         per_tr[jj*steps+j+len(stim_uq)*steps]=len(ind_ctx1)/len(ind_ctx1_pre)
    #         len_tr[jj*steps+j]=len(ind_ctx0)
    #         len_tr[jj*steps+j+len(stim_uq)*steps]=len(ind_ctx1)
    #         #mean_coh_pre[jj*steps+j]=np.mean(firing_rate[ind_ctx0][:,:,j],axis=0)
    #         #mean_coh_pre[jj*steps+j+len(stim_uq)*steps]=np.mean(firing_rate[ind_ctx1][:,:,j],axis=0)
    #         mean_coh_pre[jj*steps+j]=np.mean(fr_norm[ind_ctx0][:,:,j],axis=0) 
    #         mean_coh_pre[jj*steps+j+2*steps]=np.mean(fr_norm[ind_ctx1][:,:,j],axis=0)
    #         #mean_coh_pre[jj*steps+j]=np.mean(fr_surr[ind_ctx0][:,:,j],axis=0)
    #         #mean_coh_pre[jj*steps+j+2*steps]=np.mean(fr_surr[ind_ctx1][:,:,j],axis=0)
            
    # ind_tr=(per_tr>=0.25)*(len_tr>=10)
    # print (per_tr)
    # mean_coh=mean_coh_pre[ind_tr]
    
    # embedding=PCA(n_components=3)
    # fitPCA=embedding.fit(mean_coh)
    # print (fitPCA.explained_variance_ratio_)
    
    # mean_coh_col=np.array([np.nanmean(mean_coh_pre[l*steps:(l+1)*steps][ind_tr[l*steps:(l+1)*steps]],axis=0) for l in range(2*len(coh_uq))])
    # if monkeys[k]=='Niels':
    #     mean_coh_col=np.delete(mean_coh_col,[0,14,15,29],axis=0)
    
    # # Plot collapsing time only left coherences
    # pseudo_mds_ctx=embedding.transform(mean_coh_col)        
    
    # plt.rcParams.update({'font.size': 15})
    # fig = plt.figure()#figsize=(2,2)
    # ax = fig.add_subplot(111, projection='3d')
    
    # for jj in range(len(mean_coh_col)):
    #     ax.scatter(pseudo_mds_ctx[jj,0],pseudo_mds_ctx[jj,1],pseudo_mds_ctx[jj,2],color=col[jj],alpha=alph[jj],s=20)
    # ax.set_xlabel('PC1')
    # ax.set_ylabel('PC2')
    # ax.set_zlabel('PC3')
    # ax.set_xlim([-3,3])
    # ax.set_ylim([-3,3])
    # ax.set_zlim([-3,3])
    # plt.show()
    # plt.close(fig)
    
    # Plot as a function of time 2 conditions
    # col2=np.array(['green','green','blue','blue'])
    # alph2=np.array([1,0.3,0.3,1])
    # for j in range(steps)[::-1]:
    #     print (j)
    #     plt.rcParams.update({'font.size': 15})
    #     fig = plt.figure()
    #     ax = fig.add_subplot(111, projection='3d')
    #     ind_step=(np.arange(0,steps*2*len(stim_uq),steps)+j)
    #     pseudo_mds_ctx=embedding.transform(mean_coh_pre[ind_step][ind_tr[ind_step]])
    #     for jj in range(2*len(stim_uq)):
    #         ax.scatter(pseudo_mds_ctx[jj,0],pseudo_mds_ctx[jj,1],pseudo_mds_ctx[jj,2],color=col2[jj],alpha=alph2[jj],s=100)
    #     ax.set_xlabel('PC1')
    #     ax.set_ylabel('PC2')
    #     ax.set_zlabel('PC3')
    #     # ax.set_xlim([-3,3])
    #     # ax.set_ylim([-3,3])
    #     # ax.set_zlim([-3,3])
    #     ax.set_xlim([-2,2])
    #     ax.set_ylim([-2,2])
    #     ax.set_zlim([-2,2])
    #     plt.show()
    #     plt.close(fig)
