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
input_noise=2
scale_ctx=1

reg=1e-10
lr=0.001
n_epochs=200
n_files=10

zt_ref=1.2 #Cut-off on decision variable for reaction time (threshold or the decision bound). Something between 1 and 2
#m_bound=0.5
#dec_bound=np.array([t_steps*m_bound-i*m_bound for i in range(t_steps)])

save_fig=True

coh_uq=np.linspace(-1,1,11)
#coh_uq=np.array([-1,-0.5,-0.25,-0.1,-0.05,0,0.05,0.1,0.25,0.5,1])
coh_uq_abs=coh_uq[coh_uq>=0]
print (coh_uq_abs)
wei_ctx=[2,1] # first: respond same choice from your context, second: respond opposite choice from your context. For unbalanced contexts increase first number. You don't want to make mistakes on choices on congruent contexts. 

perf_task=nan*np.zeros((n_files,2,len(coh_uq),t_steps))
perf_task_abs=nan*np.zeros((n_files,2,len(coh_uq_abs),t_steps))
psycho=nan*np.zeros((n_files,len(coh_uq),t_steps,3))
perf_bias=nan*np.zeros((n_files,len(coh_uq),t_steps,3))
perf_dec_ctx=nan*np.zeros((n_files,t_steps,2))
rt=nan*np.zeros((n_files,len(coh_uq),3))
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
    rec.fit(input_seq=all_train['input_rec'],target_seq=all_train['target_vec'],batch_size=batch_size,n_epochs=n_epochs,sigma_noise=sigma_train,wei_ctx=wei_ctx)

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

    # Info Choice and Context
    for j in range(t_steps):
        perf_dec_ctx[hh,j]=class_twovars(ut_test[:,j],stimulus,context)
    #print (perf_dec_ctx[hh])    
    
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
    fig.savefig('/home/ramon/Dropbox/Esteki_Kiani/plots/figure_rnn_prob_correct_coh_rr%i%i_new.pdf'%(wei_ctx[0],wei_ctx[1]),dpi=500,bbox_inches='tight')
    fig.savefig('/home/ramon/Dropbox/Esteki_Kiani/plots/figure_rnn_prob_correct_coh_rr%i%i_new.png'%(wei_ctx[0],wei_ctx[1]),dpi=500,bbox_inches='tight')

########################################################
print (np.mean(perf_dec_ctx,axis=0))

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
ax.plot(coh_uq,rt_m[:,0],color='black',label='All trials')
ax.fill_between(coh_uq,rt_m[:,0]-rt_sem[:,0],rt_m[:,0]+rt_sem[:,0],color='black',alpha=0.5)
ax.plot(coh_uq,rt_m[:,1],color='green',label='Context Left')
ax.fill_between(coh_uq,rt_m[:,1]-rt_sem[:,1],rt_m[:,1]+rt_sem[:,1],color='green',alpha=0.5)
ax.plot(coh_uq,rt_m[:,2],color='blue',label='Context Right')
ax.fill_between(coh_uq,rt_m[:,2]-rt_sem[:,2],rt_m[:,2]+rt_sem[:,2],color='blue',alpha=0.5)
ax.set_ylabel('Reaction time (steps)')
ax.set_xlabel('Evidence Right Choice (%)')
ax.set_ylim([1,20])
if save_fig:
    fig.savefig('/home/ramon/Dropbox/Esteki_Kiani/plots/figure_reaction_time_rr%i%i_new.pdf'%(wei_ctx[0],wei_ctx[1]),dpi=500,bbox_inches='tight')
    fig.savefig('/home/ramon/Dropbox/Esteki_Kiani/plots/figure_reaction_time_rr%i%i_new.png'%(wei_ctx[0],wei_ctx[1]),dpi=500,bbox_inches='tight')

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
    ax.set_xlabel('Evidence Right Choice (%)')
    if save_fig:
        fig.savefig('/home/ramon/Dropbox/Esteki_Kiani/plots/figure_rnn_psychometric_t%i_rr%i%i_new.png'%(t_plot,wei_ctx[0],wei_ctx[1]),dpi=500,bbox_inches='tight')

    # Probability Correct
    fig=plt.figure(figsize=(2.3,2))
    ax=fig.add_subplot(111)
    miscellaneous.adjust_spines(ax,['left','bottom'])
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
    if save_fig:
        fig.savefig('/home/ramon/Dropbox/Esteki_Kiani/plots/figure_rnn_perf_bias_t%i_rr%i%i_new.png'%(t_plot,wei_ctx[0],wei_ctx[1]),dpi=500,bbox_inches='tight')        
