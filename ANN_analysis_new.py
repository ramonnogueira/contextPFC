import torch
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from scipy.stats import sem
import scipy
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import itertools
import pickle as pkl
import nn_pytorch
import miscellaneous_ANN
import miscellaneous
from sklearn.model_selection import KFold,StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.svm import LinearSVC
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
nan=float('nan')
minf=float('-inf')
pinf=float('inf')
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

# Currently we have
#     - Fixed duration
#     - Context is an explicit signal on each trial
#     - No context change nor learning
# What do we want
#     - Trials with variable trial lengths
#     - Context is an internal variable
#     - Context Change is uncued

# Geometry Analysis
# DONE  - Analyze Dimensionality of Curve overall:
#     DONE - Global: Shattering Dim for different resolutions
#     DONE - Local: sliding window 4 points to see if dim is lower at the end than center

# Are the two contexts parallel?
# - Do CCGP tricks
# - DO XOR for 4 points: 2 coherences x 2 contexts (sliding window)

# RESULT:
# Context is more decodable when it's behaviorally relevant: same strength of context (input signal) is much more present (DP) when there is an asymmetry (NEED CONFIRM). 

def resolution_dim(clase,resol):
    if resol=='1':
        clase_vec=[0,1,2,3,4,5,6,7,8,9,10]    
    if resol=='2':
       #clase_vec=[0,1,2,3,4,5,6,7,8,9,10]    
        clase_vec=[0,0,1,1,2,2,2,3,3,4,4]
    if resol=='3':
       #clase_vec=[0,1,2,3,4,5,6,7,8,9,10]    
        clase_vec=[0,0,0,1,1,2,2,2,3,3,3]
    if resol=='4':
       #clase_vec=[0,1,2,3,4,5,6,7,8,9,10]    
        clase_vec=[0,0,0,0,1,1,1,1,3,3,3]
    if resol=='5':
       #clase_vec=[0,1,2,3,4,5,6,7,8,9,10]    
        clase_vec=[0,0,0,0,0,0,1,1,1,1,1]
    #    
    clase_uq=np.unique(clase)
    clase_def=nan*np.zeros(len(clase))
    for i in range(len(clase_uq)):
        clase_def[clase==clase_uq[i]]=clase_vec[i]
    return clase_def


def dimensionality(data,clase,n_dim):
    clase_uq=np.unique(clase)
    len_1=len(clase_uq) # length or number of diferent categories or clases 
    clase_dim_pre=np.zeros(len_1)
    clase_dim_pre[int(0.5*len_1):]=1 # vector pre of coloring for the different clases
    dim_vec=nan*np.zeros(n_dim) # vector pre of performance across random colorings
    for d in range(n_dim):
        clase_dim=np.random.permutation(clase_dim_pre) # random coloring on the loop
        clase_d=nan*np.zeros(len(clase)) # random class asigned to each trial
        for l in range(len(clase_uq)):
            ind_l=np.where(clase==clase_uq[l])[0] 
            clase_d[ind_l]=clase_dim[l]
        # Classifier CV
        n_cv=5
        reg=1
        skf=StratifiedKFold(n_splits=n_cv)
        g=-1
        perf_cv=nan*np.zeros(n_cv)
        for train, test in skf.split(data,clase_d):
            g=(g+1)
            cl=LogisticRegression(C=1/reg,class_weight='balanced')
            cl.fit(data[train],clase_d[train])
            perf_cv[g]=cl.score(data[test],clase_d[test])
        dim_vec[d]=np.mean(perf_cv)
    return np.mean(dim_vec)

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

         supp=LogisticRegression(C=reg,class_weight='balanced',solver='lbfgs')
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

#######################################################
# Parameters       
n_trials_train=200
n_trials_test=200
t_steps=20
xx=np.arange(t_steps)/10

batch_size=1100
n_hidden=10
sigma_train=1
sigma_test=1
input_noise=1
scale_ctx=1

disp_vec=np.array([0,1,2,3,4,5])

reg=1e-5
lr=0.01 
n_files=5

coh_uq=np.linspace(-1,1,11)
#coh_uq=np.array([-1,-0.5,-0.25,-0.1,-0.05,0,0.05,0.1,0.25,0.5,1])
coh_uq_abs=coh_uq[coh_uq>=0]
print (coh_uq_abs)
wei_ctx=[1,1] # first: respond same choice from your context, second: respond opposite choice from your context. For unbalanced contexts increase first number. You don't want to make mistakes on choices on congruent contexts. 

perf_task=nan*np.zeros((n_files,2,len(coh_uq),t_steps))
perf_task_abs=nan*np.zeros((n_files,2,len(coh_uq_abs),t_steps))
psycho=nan*np.zeros((n_files,len(coh_uq),t_steps,3))
perf_bias=nan*np.zeros((n_files,len(coh_uq),t_steps,3))
shatter_dim=nan*np.zeros((n_files,t_steps))
dim_local=nan*np.zeros((n_files,t_steps,8))

lin_comp_local=nan*np.zeros((n_files,t_steps,len(disp_vec),10,3))
ccgp_local=nan*np.zeros((n_files,t_steps,len(disp_vec),10,2))
perf_dec_ctx=nan*np.zeros((n_files,t_steps,2))
for hh in range(n_files):
    print (hh)
    # Def variables
    all_train=miscellaneous_ANN.create_input(n_trials_train,t_steps,coh_uq,input_noise,scale_ctx=scale_ctx)
    all_test=miscellaneous_ANN.create_input(n_trials_test,t_steps,coh_uq,input_noise,scale_ctx=scale_ctx)
    context=all_test['input_rec'].detach().numpy()[:,0,1]
    ctx_uq=np.unique(context)
    stimulus=all_test['target_vec'].detach().numpy()
    coherence=all_test['coherence']

    # Train RNN
    rec=nn_pytorch.nn_recurrent(reg=reg,lr=lr,output_size=2,hidden_dim=n_hidden)
    rec.fit(input_seq=all_train['input_rec'],target_seq=all_train['target_vec'],batch_size=batch_size,sigma_noise=sigma_train,wei_ctx=wei_ctx)

    # Indices trials
    index0=np.where(all_test['target_vec']==0)[0]
    index1=np.where(all_test['target_vec']==1)[0]
    # Hidden units' activity
    ut_train=rec.model(all_train['input_rec'],sigma_noise=sigma_train)[2].detach().numpy()
    ut_test=rec.model(all_test['input_rec'],sigma_noise=sigma_test)[2].detach().numpy()
    # Decision units activity
    zt_train=rec.model(all_train['input_rec'],sigma_noise=sigma_train)[3].detach().numpy()
    zt_test=rec.model(all_test['input_rec'],sigma_noise=sigma_test)[3].detach().numpy()

    # Classifier weights
    w1=rec.model.fc.weight.detach().numpy()[0]
    w2=rec.model.fc.weight.detach().numpy()[1]
    weights=(w1-w2)
    b1=rec.model.fc.bias.detach().numpy()[0]
    b2=rec.model.fc.bias.detach().numpy()[1]
    bias=(b1-b2)

    # Info Choice and Context
    for j in range(t_steps):
        perf_dec_ctx[hh,j]=class_twovars(ut_test[:,j],stimulus,context)
    print (perf_dec_ctx[hh])    
    
    # Plot performance
    dec_train=np.argmax(zt_train,axis=2)
    dec_test=np.argmax(zt_test,axis=2)
    corr_train=all_train['target_vec'].detach().numpy()
    corr_test=all_test['target_vec'].detach().numpy()
    for j in range(t_steps):
        for jj in range(len(coh_uq)):
            ind_coh_tr=np.where(all_train['coherence']==coh_uq[jj])[0]
            ind_coh_te=np.where(all_test['coherence']==coh_uq[jj])[0]
            perf_task[hh,0,jj,j]=np.mean(dec_train[ind_coh_tr][:,j]==corr_train[ind_coh_tr])
            perf_task[hh,1,jj,j]=np.mean(dec_test[ind_coh_te][:,j]==corr_test[ind_coh_te])

    # Plot performance Absolute coherence
    dec_train=np.argmax(zt_train,axis=2)
    dec_test=np.argmax(zt_test,axis=2)
    corr_train=all_train['target_vec'].detach().numpy()
    corr_test=all_test['target_vec'].detach().numpy()
    for j in range(t_steps):
        for jj in range(len(coh_uq_abs)):
            ind_coh_tr=np.where(abs(all_train['coherence'])==coh_uq_abs[jj])[0]
            ind_coh_te=np.where(abs(all_test['coherence'])==coh_uq_abs[jj])[0]
            perf_task_abs[hh,0,jj,j]=np.mean(dec_train[ind_coh_tr][:,j]==corr_train[ind_coh_tr])
            perf_task_abs[hh,1,jj,j]=np.mean(dec_test[ind_coh_te][:,j]==corr_test[ind_coh_te])

    # Psychometric
    for j in range(t_steps):
        for i in range(len(coh_uq)):
            psycho[hh,i,j,0]=np.mean(dec_test[:,j][(coherence==coh_uq[i])])
            psycho[hh,i,j,1]=np.mean(dec_test[:,j][(coherence==coh_uq[i])&(context==ctx_uq[0])])
            psycho[hh,i,j,2]=np.mean(dec_test[:,j][(coherence==coh_uq[i])&(context==ctx_uq[1])])

    # Perf Bias (similar to chronometric?)
    for j in range(t_steps):
        for i in range(len(coh_uq)):
            perf_bias[hh,i,j,0]=np.mean(dec_test[:,j][coherence==coh_uq[i]]==stimulus[coherence==coh_uq[i]])
            perf_bias[hh,i,j,1]=np.mean(dec_test[:,j][(coherence==coh_uq[i])&(context==ctx_uq[0])]==stimulus[(coherence==coh_uq[i])&(context==ctx_uq[0])])
            perf_bias[hh,i,j,2]=np.mean(dec_test[:,j][(coherence==coh_uq[i])&(context==ctx_uq[1])]==stimulus[(coherence==coh_uq[i])&(context==ctx_uq[1])])


######################################################
# Figure psychometric
psycho_m=np.mean(psycho,axis=0)
psycho_sem=sem(psycho,axis=0)

fig=plt.figure(figsize=(2.3,2))
ax=fig.add_subplot(111)
miscellaneous.adjust_spines(ax,['left','bottom'])
#print (psycho_m)

t_plot=19
ax.plot(coh_uq*100,psycho_m[:,t_plot,0],color='black')
ax.fill_between(coh_uq*100,psycho_m[:,t_plot,0]-psycho_sem[:,t_plot,0],psycho_m[:,t_plot,0]+psycho_sem[:,t_plot,0],color='black',alpha=0.6)
ax.plot(coh_uq*100,psycho_m[:,t_plot,1],color='green')
ax.fill_between(coh_uq*100,psycho_m[:,t_plot,1]-psycho_sem[:,t_plot,1],psycho_m[:,t_plot,1]+psycho_sem[:,t_plot,1],color='green',alpha=0.6)
ax.plot(coh_uq*100,psycho_m[:,t_plot,2],color='blue')
ax.fill_between(coh_uq*100,psycho_m[:,t_plot,2]-psycho_sem[:,t_plot,2],psycho_m[:,t_plot,2]+psycho_sem[:,t_plot,2],color='blue',alpha=0.6)
ax.plot(coh_uq*100,0.5*np.ones(len(coh_uq)),color='black',linestyle='--')
ax.axvline(0,color='black',linestyle='--')
ax.set_ylim([0,1])
ax.set_ylabel('Probability Right Response')
ax.set_xlabel('Evidence Right Choice (%)')
fig.savefig('/home/ramon/Dropbox/Esteki_Kiani/figures/figure_rnn_psychometric.pdf',dpi=500,bbox_inches='tight')

# Probability Correct
perfbias_m=np.mean(perf_bias,axis=0)
perfbias_sem=sem(perf_bias,axis=0)

fig=plt.figure(figsize=(2.3,2))
ax=fig.add_subplot(111)
miscellaneous.adjust_spines(ax,['left','bottom'])

t_plot=19
ax.plot(coh_uq*100,perfbias_m[:,t_plot,0],color='black')
ax.fill_between(coh_uq*100,perfbias_m[:,t_plot,0]-perfbias_sem[:,t_plot,0],perfbias_m[:,t_plot,0]+perfbias_sem[:,t_plot,0],color='black',alpha=0.6)
ax.plot(coh_uq*100,perfbias_m[:,t_plot,1],color='green')
ax.fill_between(coh_uq*100,perfbias_m[:,t_plot,1]-perfbias_sem[:,t_plot,1],perfbias_m[:,t_plot,1]+perfbias_sem[:,t_plot,1],color='green',alpha=0.6)
ax.plot(coh_uq*100,perfbias_m[:,t_plot,2],color='blue')
ax.fill_between(coh_uq*100,perfbias_m[:,t_plot,2]-perfbias_sem[:,t_plot,2],perfbias_m[:,t_plot,2]+perfbias_sem[:,t_plot,2],color='blue',alpha=0.6)
ax.plot(coh_uq*100,0.5*np.ones(len(coh_uq)),color='black',linestyle='--')
ax.axvline(0,color='black',linestyle='--')
ax.set_ylim([0,1])
ax.set_ylabel('Probability Correct')
ax.set_xlabel('Evidence Right Choice (%)')
fig.savefig('/home/ramon/Dropbox/Esteki_Kiani/figures/figure_rnn_perf_bias.pdf',dpi=500,bbox_inches='tight')        
