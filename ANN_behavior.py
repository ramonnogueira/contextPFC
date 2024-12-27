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
from scipy.stats import spearmanr
from numpy.random import permutation
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

def class_twovars(data,var1,var2):
    n_rand=10
    n_cv=5
    reg=1
    perf=nan*np.zeros((n_rand,n_cv,2))
    wei_vec=nan*np.zeros((n_rand,n_cv,2,len(data[0])))
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
            wei_vec[i,g,0]=cl.coef_[0]
        # Decode Var2
        skf=StratifiedKFold(n_splits=n_cv)
        g=-1
        for train, test in skf.split(data_r,clas_r[:,1]):
            g=(g+1)
            cl=LogisticRegression(C=1/reg)
            cl.fit(data_r[train],clas_r[train][:,1])
            perf[i,g,1]=cl.score(data_r[test],clas_r[test][:,1])
            wei_vec[i,g,1]=cl.coef_[0]
    return np.mean(perf,axis=(0,1)),np.mean(wei_vec,axis=(0,1))

def rt_func(diff_zt,ind,zt_ref):
    ztcoh=np.mean(diff_zt[ind],axis=0)
    rt_pre=np.where(abs(ztcoh)>zt_ref)[0]
    if len(rt_pre)==0:
        rt=20
    else:
        rt=rt_pre[0]+1
    return rt

def rt_dec_func(zt_test,diff_zt,dec_bound):
    # Calculate reaction time
    rt_vec=nan*np.zeros(len(diff_zt))
    for tt in range(len(diff_zt)):
        rt_pre=np.where(abs(diff_zt[tt])>dec_bound)[0]
        if len(rt_pre)==0:
            rt=20
        else:
            rt=rt_pre[0]+1
        rt_vec[tt]=rt
    # Calculate decision at t = reaction time
    dec_vec=nan*np.zeros(len(diff_zt))
    for tt in range(len(diff_zt)):
        dec_vec[tt]=np.argmax(zt_test[tt,int(rt_vec[tt])-1])
    return rt_vec,dec_vec

def null_spearmanr(x,y,n):
    null_corr=nan*np.zeros(n)
    for i in range(n):
        x_sh=permutation(x)
        y_sh=permutation(y)
        null_corr[i]=spearmanr(x_sh,y_sh)[0]
    return null_corr

def algo_clustering(x,y):
    ind_asc=np.argsort(x)
    y_sort=y[ind_asc[::-1]]
    x_sort=x[ind_asc]
    return x_sort,y_sort

def evol_eigen(act_orig,act,t_steps,n_neu):
    eigen_vec=nan*np.zeros((2,t_steps,n_neu))
    for i in range(t_steps):
        eigen_i=np.linalg.eigh(np.cov(act_orig[:,i],rowvar=False))[0]
        eigen=np.linalg.eigh(np.cov(act[:,i],rowvar=False))[0]
        eigen_vec[0,i]=(eigen_i[::-1]/np.sum(eigen_i))[0:n_neu]
        eigen_vec[1,i]=(eigen[::-1]/np.sum(eigen))[0:n_neu]
    return eigen_vec


#######################################################
# Parameters       
n_trials_train=200
n_trials_test=200
t_steps=20
xx=np.arange(t_steps)/10

batch_size=1000
n_hidden=10
sigma_train=1
sigma_test=1
input_noise=1
scale_ctx=1 #smaller than 0.01 there is no effect in psycho curves.
ctx_noise=1

reg=1e-5
lr=0.01
n_epochs=200#1000
n_files=10

zt_ref=0.7#Cut-off on decision variable for reaction time (threshold or the decision bound). We used 0.7 for [1,1] and [2,1] and 0.9 for [4,1].

save_fig=False

coh_uq=np.linspace(-1,1,11)
#coh_uq=np.linspace(-0.5,0.5,11)
#coh_uq=np.array([-1,-0.5,-0.25,-0.1,-0.05,0,0.05,0.1,0.25,0.5,1])
coh_uq_abs=coh_uq[coh_uq>=0]
print (coh_uq_abs)
wei_ctx=[4,1] # first: respond same choice from your context, second: respond opposite choice from your context. For unbalanced contexts increase first number. You don't want to make mistakes on choices on congruent contexts.
beta=0
b_exp=1

perf_task=nan*np.zeros((n_files,2,len(coh_uq),t_steps))
perf_task_abs=nan*np.zeros((n_files,2,len(coh_uq_abs),t_steps))
psycho=nan*np.zeros((n_files,len(coh_uq),t_steps,3))
perf_bias=nan*np.zeros((n_files,len(coh_uq),t_steps,3))
perf_dec_ctx=nan*np.zeros((n_files,t_steps,2))
wei_dec_ctx=nan*np.zeros((n_files,t_steps,2,n_hidden))
rt=nan*np.zeros((n_files,len(coh_uq),3))
eigen_all=nan*np.zeros((n_files,2,t_steps,n_hidden))
for hh in range(n_files):
    print (hh)
    # Def variables
    all_train=miscellaneous_ANN.create_input(n_trials_train,t_steps,coh_uq,input_noise,scale_ctx=scale_ctx,ctx_noise=ctx_noise)
    all_test=miscellaneous_ANN.create_input(n_trials_test,t_steps,coh_uq,input_noise,scale_ctx=scale_ctx,ctx_noise=ctx_noise)
    context=all_test['context']
    ctx_uq=np.unique(context)
    stimulus=all_test['target_vec'].detach().numpy()
    coherence=all_test['coherence']

    # Train RNN
    rec=nn_pytorch.nn_recurrent_sparse(reg=reg,lr=lr,output_size=2,hidden_dim=n_hidden)
    act_orig=rec.model(all_test['input_rec'],sigma_noise=sigma_test)[2].detach().numpy()
    
    rec.fit(input_seq=all_train['input_rec'],target_seq=all_train['target_vec'],context=all_train['context'],batch_size=batch_size,n_epochs=n_epochs,sigma_noise=sigma_train,wei_ctx=wei_ctx,beta=beta,b_exp=b_exp)

    # Indices trials
    index0=np.where(all_test['target_vec']==0)[0]
    index1=np.where(all_test['target_vec']==1)[0]
    # Hidden units' activity
    ut_train=rec.model(all_train['input_rec'],sigma_noise=sigma_train)[2].detach().numpy()
    ut_test=rec.model(all_test['input_rec'],sigma_noise=sigma_test)[2].detach().numpy()
    # Decision units activity
    zt_train=rec.model(all_train['input_rec'],sigma_noise=sigma_train)[3].detach().numpy()
    zt_test=rec.model(all_test['input_rec'],sigma_noise=sigma_test)[3].detach().numpy()
    # Network Choice
    dec_train=np.argmax(zt_train,axis=2)
    dec_test=np.argmax(zt_test,axis=2)
    # Reaction time and network choice
    diff_zt=(zt_test[:,:,0]-zt_test[:,:,1])
    for uu in range(len(coh_uq)):
        ind=np.where(coherence==coh_uq[uu])[0]
        ind0=np.where((coherence==coh_uq[uu])&(context==ctx_uq[0]))[0]
        ind1=np.where((coherence==coh_uq[uu])&(context==ctx_uq[1]))[0]
        rt[hh,uu,0]=rt_func(diff_zt,ind,zt_ref)
        rt[hh,uu,1]=rt_func(diff_zt,ind0,zt_ref)
        rt[hh,uu,2]=rt_func(diff_zt,ind1,zt_ref)
        
    # Classifier weights
    w1=rec.model.fc.weight.detach().numpy()[0]
    w2=rec.model.fc.weight.detach().numpy()[1]
    weights=(w1-w2)
    b1=rec.model.fc.bias.detach().numpy()[0]
    b2=rec.model.fc.bias.detach().numpy()[1]
    bias=(b1-b2)
  
    eigen_all[hh]=evol_eigen(act_orig,ut_test,t_steps,n_hidden)
    
    # # Info Choice and Context
    # for j in range(t_steps):
    #     aa=class_twovars(ut_test[:,j],stimulus,context)
    #     perf_dec_ctx[hh,j]=aa[0]
    #     wei_dec_ctx[hh,j]=aa[1]

    # n_sh=1000
    # for t in range(t_steps):
    #     print (t)
    #     x=abs(wei_dec_ctx[hh,t,0])
    #     y=abs(wei_dec_ctx[hh,t,1])

    #     print (len(x[x==0]),len(y[y==0]))
        
    #     corr=spearmanr(x,y)
    #     print (corr)
    #     #corr_sh_distr=null_spearmanr(x,y,n=n_sh)
    #     plt.scatter(x,y,s=1)
    #     plt.show()        
    #     #plt.hist(corr_sh_distr,color='blue',bins=100,alpha=0.5)
    #     #plt.axvline(corr[0],color='blue')
    #     #plt.show()

    #     x,y=algo_clustering(x,y)
    #     corr=spearmanr(x,y)
    #     print (corr)
    #     # corr_sh_distr=null_spearmanr(x,y,n=n_sh)
    #     plt.scatter(x,y,s=1)
    #     plt.show()        
    #     # plt.hist(corr_sh_distr,color='blue',bins=100,alpha=0.5)
    #     # plt.axvline(corr[0],color='blue')
    #     # plt.show()

    
    # Plot performance
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
# Eigenvalues
eigen_m=np.nanmean(eigen_all,axis=0)
eigen_sem=sem(eigen_all,axis=0,nan_policy='omit')
for i in range(t_steps):
    fig=plt.figure(figsize=(2.3,2))
    ax=fig.add_subplot(111)
    miscellaneous.adjust_spines(ax,['left','bottom'])
    ax.plot(eigen_m[0,i],color='red',label='Pre-trained')
    ax.plot(eigen_m[1,i],color='blue',label='Trained')
    ax.fill_between(np.arange(n_hidden),eigen_m[0,i]-eigen_sem[0,i],eigen_m[0,i]+eigen_sem[0,i],color='red',alpha=0.5)
    ax.fill_between(np.arange(n_hidden),eigen_m[1,i]-eigen_sem[1,i],eigen_m[1,i]+eigen_sem[1,i],color='blue',alpha=0.5)
    ax.set_ylim([0,1])
    ax.set_xlabel('Eigenvalues')
    ax.set_ylabel('Variance Explained')
    plt.legend(loc='best')
    fig.savefig('/home/ramon/Dropbox/Proyectos_Postdoc/Esteki_Kiani/plots/eigenspectrum_rnn_t_%i.png'%i,dpi=500,bbox_inches='tight')

# Plot performance vs time for different coherences
perf_abs_m=np.mean(perf_task_abs,axis=0)
perf_abs_sem=sem(perf_task_abs,axis=0)

fig=plt.figure(figsize=(2.3,2))
ax=fig.add_subplot(111)
miscellaneous.adjust_spines(ax,['left','bottom'])
for i in range(len(coh_uq_abs)):
    ax.plot(np.arange(t_steps),perf_abs_m[1,i],color='black',alpha=(i+1)/len(coh_uq_abs))
    ax.fill_between(np.arange(t_steps),perf_abs_m[1,i]-perf_abs_sem[1,i],perf_abs_m[1,i]+perf_abs_sem[1,i],color='black',alpha=(i+1)/len(coh_uq_abs))
ax.plot(np.arange(t_steps),0.5*np.ones(t_steps),color='black',linestyle='--')
ax.set_ylim([0.4,1])
ax.set_ylabel('Probability Correct')
ax.set_xlabel('Time')
if save_fig:
    #fig.savefig('/home/ramon/Dropbox/Esteki_Kiani/plots/figure_rnn_prob_correct_coh_rr%i%i_prueba2.pdf'%(wei_ctx[0],wei_ctx[1]),dpi=500,bbox_inches='tight')
    fig.savefig('/home/ramon/Dropbox/Esteki_Kiani/plots/figure_rnn_prob_correct_coh_rr%i%i_prueba2.png'%(wei_ctx[0],wei_ctx[1]),dpi=500,bbox_inches='tight')

########################################################
#print (np.mean(perf_dec_ctx,axis=0))

psycho_m=np.mean(psycho,axis=0)
psycho_sem=sem(psycho,axis=0)
perfbias_m=np.mean(perf_bias,axis=0)
perfbias_sem=sem(perf_bias,axis=0)
rt_m=np.mean(rt,axis=0)
rt_sem=sem(rt,axis=0)

# Plot Reaction Time
fig=plt.figure(figsize=(2.3,2))
ax=fig.add_subplot(111)
miscellaneous.adjust_spines(ax,['left','bottom'])
ax.plot(100*coh_uq,rt_m[:,0],color='black',label='All trials')
ax.fill_between(100*coh_uq,rt_m[:,0]-rt_sem[:,0],rt_m[:,0]+rt_sem[:,0],color='black',alpha=0.5)
ax.plot(100*coh_uq,rt_m[:,1],color='green',label='Context Left')
ax.fill_between(100*coh_uq,rt_m[:,1]-rt_sem[:,1],rt_m[:,1]+rt_sem[:,1],color='green',alpha=0.5)
ax.plot(100*coh_uq,rt_m[:,2],color='blue',label='Context Right')
ax.fill_between(100*coh_uq,rt_m[:,2]-rt_sem[:,2],rt_m[:,2]+rt_sem[:,2],color='blue',alpha=0.5)
ax.set_ylabel('Reaction time (steps)')
ax.set_xlabel('Evidence Right (Coherence)')
ax.set_ylim([1,20])
if save_fig:
    #fig.savefig('/home/ramon/Dropbox/Esteki_Kiani/plots/figure_reaction_time_rr%i%i_prueba2.pdf'%(wei_ctx[0],wei_ctx[1]),dpi=500,bbox_inches='tight')
    fig.savefig('/home/ramon/Dropbox/Esteki_Kiani/plots/figure_reaction_time_rr%i%i_prueba2.png'%(wei_ctx[0],wei_ctx[1]),dpi=500,bbox_inches='tight')

for t_plot in range(t_steps):
    # Figure psychometric
    fig=plt.figure(figsize=(2.3,2))
    ax=fig.add_subplot(111)
    miscellaneous.adjust_spines(ax,['left','bottom'])
    #print (psycho_m)
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
    ax.set_xlabel('Evidence Right (Coherence)')
    if save_fig:
        fig.savefig('/home/ramon/Dropbox/Esteki_Kiani/plots/figure_rnn_psychometric_t%i_rr%i%i_prueba2.png'%(t_plot,wei_ctx[0],wei_ctx[1]),dpi=500,bbox_inches='tight')

    # # Probability Correct
    # fig=plt.figure(figsize=(2.3,2))
    # ax=fig.add_subplot(111)
    # miscellaneous.adjust_spines(ax,['left','bottom'])
    # ax.plot(coh_uq*100,perfbias_m[:,t_plot,0],color='black')
    # ax.fill_between(coh_uq*100,perfbias_m[:,t_plot,0]-perfbias_sem[:,t_plot,0],perfbias_m[:,t_plot,0]+perfbias_sem[:,t_plot,0],color='black',alpha=0.6)
    # ax.plot(coh_uq*100,perfbias_m[:,t_plot,1],color='green')
    # ax.fill_between(coh_uq*100,perfbias_m[:,t_plot,1]-perfbias_sem[:,t_plot,1],perfbias_m[:,t_plot,1]+perfbias_sem[:,t_plot,1],color='green',alpha=0.6)
    # ax.plot(coh_uq*100,perfbias_m[:,t_plot,2],color='blue')
    # ax.fill_between(coh_uq*100,perfbias_m[:,t_plot,2]-perfbias_sem[:,t_plot,2],perfbias_m[:,t_plot,2]+perfbias_sem[:,t_plot,2],color='blue',alpha=0.6)
    # ax.plot(coh_uq*100,0.5*np.ones(len(coh_uq)),color='black',linestyle='--')
    # ax.axvline(0,color='black',linestyle='--')
    # ax.set_ylim([0,1])
    # ax.set_ylabel('Probability Correct')
    # ax.set_xlabel('Evidence Right Choice (%)')
    # if save_fig:
    #     fig.savefig('/home/ramon/Dropbox/Esteki_Kiani/plots/figure_rnn_perf_bias_t%i_rr%i%i.pdf'%(t_plot,wei_ctx[0],wei_ctx[1]),dpi=500,bbox_inches='tight')        
