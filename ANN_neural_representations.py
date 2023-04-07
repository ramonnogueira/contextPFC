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

def class_twovars(data,var1,var2,n_neu):
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
        # Select neurons
        ind_neu=np.random.choice(len(data[0]),n_neu,replace=False)
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
            cl.fit(data_r[train][:,ind_neu],clas_r[train][:,0])
            perf[i,g,0]=cl.score(data_r[test][:,ind_neu],clas_r[test][:,0])
        # Decode Var2
        skf=StratifiedKFold(n_splits=n_cv)
        g=-1
        for train, test in skf.split(data_r,clas_r[:,1]):
            g=(g+1)
            cl=LogisticRegression(C=1/reg)
            cl.fit(data_r[train][:,ind_neu],clas_r[train][:,1])
            perf[i,g,1]=cl.score(data_r[test][:,ind_neu],clas_r[test][:,1])
    return np.mean(perf,axis=(0,1))

def rt_func(diff_zt,ind,zt_ref):
    ztcoh=np.mean(diff_zt[ind],axis=0)
    rt_pre=np.where(abs(ztcoh)>zt_ref)[0]
    if len(rt_pre)==0:
        rt=20
    else:
        rt=rt_pre[0]+1
    return rt

#######################################################
# Parameters       
n_trials_train=200
n_trials_test=200
t_steps=20
xx=np.arange(t_steps)/10

batch_size=200
n_hidden=50
n_neu=10
sigma_train=1
sigma_test=1
input_noise=1
scale_ctx=1

reg=1e-5
lr=0.001
n_epochs=200
n_files=10

save_fig=True

coh_uq=np.linspace(-1,1,11)
#coh_uq=np.array([-1,-0.5,-0.25,-0.1,-0.05,0,0.05,0.1,0.25,0.5,1])
coh_uq_abs=coh_uq[coh_uq>=0]
print (coh_uq_abs)
wei_ctx=[2,1] # first: respond same choice from your context, second: respond opposite choice from your context. For unbalanced contexts increase first number. You don't want to make mistakes on choices on congruent contexts. 

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
    choice=dec_test[:,-1]
    correct=(stimulus==choice)
        
    # Classifier weights
    w1=rec.model.fc.weight.detach().numpy()[0]
    w2=rec.model.fc.weight.detach().numpy()[1]
    weights=(w1-w2)
    b1=rec.model.fc.bias.detach().numpy()[0]
    b2=rec.model.fc.bias.detach().numpy()[1]
    bias=(b1-b2)

    # Info Choice and Context
    for j in range(t_steps):
        # Only correct trials!
        perf_dec_ctx[hh,j]=class_twovars(ut_test[:,j][correct],stimulus[correct],context[correct],n_neu)
    

######################################################

print (np.mean(perf_dec_ctx,axis=0))

# Plot decoding performance for stimulus (only correct!) and context
perf_dec_ctx_m=np.mean(perf_dec_ctx,axis=0)
perf_dec_ctx_sem=sem(perf_dec_ctx,axis=0)

fig=plt.figure(figsize=(2.3,2))
ax=fig.add_subplot(111)
miscellaneous.adjust_spines(ax,['left','bottom'])
ax.plot(np.arange(t_steps),perf_dec_ctx_m[:,0],color='blue')
ax.fill_between(np.arange(t_steps),perf_dec_ctx_m[:,0]-perf_dec_ctx_sem[:,0],perf_dec_ctx_m[:,0]+perf_dec_ctx_sem[:,0],color='blue',alpha=0.5)
ax.plot(np.arange(t_steps),perf_dec_ctx_m[:,1],color='brown')
ax.fill_between(np.arange(t_steps),perf_dec_ctx_m[:,1]-perf_dec_ctx_sem[:,1],perf_dec_ctx_m[:,1]+perf_dec_ctx_sem[:,1],color='brown',alpha=0.5)
ax.plot(np.arange(t_steps),0.5*np.ones(t_steps),color='black',linestyle='--')
ax.set_ylim([0.4,1])
ax.set_ylabel('Decoding Performance')
ax.set_xlabel('Time')
if save_fig:
    fig.savefig('/home/ramon/Dropbox/Esteki_Kiani/plots/figure_decoding_dec_ctx_neu_%i_rr%i%i.pdf'%(n_neu,wei_ctx[0],wei_ctx[1]),dpi=500,bbox_inches='tight')
    fig.savefig('/home/ramon/Dropbox/Esteki_Kiani/plots/figure_decoding_dec_ctx_neu_%i_rr%i%i.png'%(n_neu,wei_ctx[0],wei_ctx[1]),dpi=500,bbox_inches='tight')

