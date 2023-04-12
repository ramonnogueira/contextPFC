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
from sklearn.manifold import MDS
nan=float('nan')
minf=float('-inf')
pinf=float('inf')
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

def classifier(data,clase,reg):
    n_splits=5
    perf=nan*np.zeros((n_splits,2))
    cv=StratifiedKFold(n_splits=n_splits,shuffle=True)
    g=-1
    for train_index, test_index in cv.split(data,clase):
        g=(g+1)
        clf = LogisticRegression(C=reg,class_weight='balanced')
        clf.fit(data[train_index],clase[train_index])
        perf[g,0]=clf.score(data[train_index],clase[train_index])
        perf[g,1]=clf.score(data[test_index],clase[test_index])
    return np.mean(perf,axis=0)

def class_twovars(data,feat_binary,bias_vec,n_neu):
    n_rand=10
    n_cv=5
    reg=1
    perf=nan*np.zeros((n_rand,n_cv,3))
    perf_abs=nan*np.zeros((n_rand,len(bias_vec),2,2))
    var1=feat_binary[:,0]
    var2=feat_binary[:,1]
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
        skf=StratifiedKFold(n_splits=n_cv,shuffle=True)
        g=-1
        for train, test in skf.split(data_r,clas_r[:,0]):
            g=(g+1)
            cl=LogisticRegression(C=1/reg)
            cl.fit(data_r[train][:,ind_neu],clas_r[train][:,0])
            perf[i,g,0]=cl.score(data_r[test][:,ind_neu],clas_r[test][:,0])
        # Decode Var2
        skf=StratifiedKFold(n_splits=n_cv,shuffle=True)
        g=-1
        for train, test in skf.split(data_r,clas_r[:,1]):
            g=(g+1)
            cl=LogisticRegression(C=1/reg)
            cl.fit(data_r[train][:,ind_neu],clas_r[train][:,1])
            perf[i,g,1]=cl.score(data_r[test][:,ind_neu],clas_r[test][:,1])
        # Decode XOR
        xor=np.sum(clas_r,axis=1)%2
        skf=StratifiedKFold(n_splits=n_cv,shuffle=True)
        g=-1
        for train, test in skf.split(data_r,xor):
            g=(g+1)
            cl=LogisticRegression(C=1/reg)
            cl.fit(data_r[train][:,ind_neu],xor[train])
            perf[i,g,2]=cl.score(data_r[test][:,ind_neu],xor[test])

        # Abstraction
        for f in range(len(bias_vec)):
            perf_abs[i,f]=abstraction_2D(data_r[:,ind_neu],clas_r,bias_vec[f],1)[0]
            
    return np.mean(perf,axis=(0,1)),np.mean(perf_abs,axis=0)

def rt_func(diff_zt,ind,zt_ref):
    ztcoh=np.mean(diff_zt[ind],axis=0)
    rt_pre=np.where(abs(ztcoh)>zt_ref)[0]
    if len(rt_pre)==0:
        rt=20
    else:
        rt=rt_pre[0]+1
    return rt

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


#######################################################
# Parameters       
n_trials_train=200
n_trials_test=200
t_steps=20
xx=np.arange(t_steps)/10

batch_size=200
n_hidden=30
n_neu=10
n_pca=10
sigma_train=1
sigma_test=1
input_noise=2#1.5
scale_ctx=1

reg=1e-10#1e-5
lr=0.001#0.001
n_epochs=200
n_files=2

save_fig=True

#coh_uq=np.linspace(-1,1,7)
coh_uq=np.linspace(-1,1,11)
#coh_uq=np.linspace(-0.5,0.5,11)
#coh_uq=np.linspace(-1,1,10)
#coh_uq=np.array([-1,-0.5,-0.25,-0.1,-0.05,0,0.05,0.1,0.25,0.5,1])
coh_uq_abs=coh_uq[coh_uq>=0]
ctx_uq=np.array([-1,1])
print (coh_uq_abs)
col=['darkgreen','darkgreen','darkgreen','darkgreen','darkgreen','black','darkgoldenrod','darkgoldenrod','darkgoldenrod','darkgoldenrod','darkgoldenrod','purple','purple','purple','purple','purple','black','darkblue','darkblue','darkblue','darkblue','darkblue']
#alf_col=[0.8,0.6,0.4,0.3,0.1,1,0.1,0.3,0.5,0.6,0.8]
alph=[0.8,0.6,0.4,0.3,0.1,1,0.1,0.3,0.5,0.6,0.8,0.8,0.6,0.4,0.3,0.1,1,0.1,0.3,0.5,0.6,0.8]
#col=['darkgreen','darkgreen','darkgreen','black','darkgoldenrod','darkgoldenrod','darkgoldenrod','purple','purple','purple','black','darkblue','darkblue','darkblue']
#alph=[1,0.5,0.2,1,0.2,0.5,1,1,0.5,0.2,1,0.2,0.5,1]


wei_ctx=[2,1] # first: respond same choice from your context, second: respond opposite choice from your context. For unbalanced contexts increase first number. You don't want to make mistakes on choices on congruent contexts.

bias_vec=np.linspace(-5,5,31) 
perf_dec_ctx=nan*np.zeros((n_files,t_steps,3))
perf_abs=nan*np.zeros((n_files,t_steps,len(bias_vec),2,2))
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

    feat_binary=nan*np.zeros((len(stimulus),2))
    feat_binary[:,0]=stimulus
    feat_binary[:,1]=context
    feat_binary[feat_binary==-1]=0

    # Info Choice and Context
    #for j in range(t_steps):
    #    print (j)
    #    # Only correct trials!

    # #############################
    # # PCA Train. Stack PSTH for each coherence one after each other
    # neu_rnd=np.sort(np.random.choice(np.arange(n_hidden),n_pca,replace=False))
    
    # mean_coh=nan*np.zeros((t_steps*2*len(coh_uq),n_pca))
    # for j in range(n_pca):
    #     for jj in range(len(coh_uq)):
    #         mean_coh[jj*t_steps:(jj+1)*t_steps,j]=np.mean(ut_test[(coherence==coh_uq[jj])&(context==ctx_uq[0])][:,:,neu_rnd[j]],axis=0)
    #         mean_coh[(jj+len(coh_uq))*t_steps:(jj+len(coh_uq)+1)*t_steps,j]=np.mean(ut_test[(coherence==coh_uq[jj])&(context==ctx_uq[1])][:,:,neu_rnd[j]],axis=0)

    # embedding=PCA(n_components=3)
    # pseudo_mds=embedding.fit(mean_coh)

    # wei_trans=embedding.transform(np.array([weights[neu_rnd]]))[0]
    # xx, yy = np.meshgrid(np.arange(20)-10,np.arange(20)-10)
    # z = (-wei_trans[0]*xx-wei_trans[1]*yy-bias)/wei_trans[2]
    
    # PCA Test
    for j in range(t_steps):
        print (j)
        
        aa=class_twovars(ut_test[:,j][correct],feat_binary[correct],bias_vec,n_neu)
        perf_dec_ctx[hh,j]=aa[0]
        perf_abs[hh,j]=aa[1]
                
        # mean_coh=nan*np.zeros((len(coh_uq),n_pca))
        # mean_coh_ctx=nan*np.zeros((2*len(coh_uq),n_pca))
        # for jj in range(len(coh_uq)):
        #     mean_coh[jj]=np.mean(ut_test[(coherence==coh_uq[jj])][:,j,neu_rnd],axis=0)
        #     mean_coh_ctx[jj]=np.mean(ut_test[(coherence==coh_uq[jj])&(context==ctx_uq[0])][:,j,neu_rnd],axis=0)
        #     mean_coh_ctx[jj+len(coh_uq)]=np.mean(ut_test[(coherence==coh_uq[jj])&(context==ctx_uq[1])][:,j,neu_rnd],axis=0)

        # pseudo_mds=embedding.transform(mean_coh)
        # pseudo_mds_ctx=embedding.transform(mean_coh_ctx)
        # #pseudo_mds=mean_coh.copy()
        # #pseudo_mds_ctx=mean_coh_ctx.copy()
        # #wei_pca=embedding.transform(np.reshape(weights,(1,len(weights))))
        # #print (wei_pca,bias)
        
        # # 3D
        # #if j==19:
        # fig = plt.figure()#figsize=(2,2)
        # ax = fig.add_subplot(111, projection='3d')
        # #for jj in range(len(mean_coh)):
        # #    ax.scatter(pseudo_mds[jj,0],pseudo_mds[jj,1],pseudo_mds[jj,2],color='black',alpha=alf_col[jj])
        # for jj in range(len(mean_coh_ctx)):
        #     ax.scatter(pseudo_mds_ctx[jj,0],pseudo_mds_ctx[jj,1],pseudo_mds_ctx[jj,2],color=col[jj],alpha=alph[jj])
        # ax.plot_surface(xx, yy, z, color='black',alpha=0.2)
        # ax.set_xlabel('PC1')
        # ax.set_ylabel('PC2')
        # ax.set_zlabel('PC3')
        # ax.set_xlim([-4,4])
        # ax.set_ylim([-4,4])
        # ax.set_zlim([-4,4])
        # plt.show()
        # plt.close(fig)



######################################################

# Plot decoding performance for stimulus (only correct!) and context
perf_dec_m=np.mean(perf_dec_ctx,axis=0)
perf_dec_sem=sem(perf_dec_ctx,axis=0)
perf_abs_m=np.mean(perf_abs,axis=0)

fig=plt.figure(figsize=(2.3,2))
ax=fig.add_subplot(111)
miscellaneous.adjust_spines(ax,['left','bottom'])
ax.plot(np.arange(t_steps),perf_dec_m[:,0],color='blue')
ax.fill_between(np.arange(t_steps),perf_dec_m[:,0]-perf_dec_sem[:,0],perf_dec_m[:,0]+perf_dec_sem[:,0],color='blue',alpha=0.5)
ax.plot(np.arange(t_steps),perf_dec_m[:,1],color='brown')
ax.fill_between(np.arange(t_steps),perf_dec_m[:,1]-perf_dec_sem[:,1],perf_dec_m[:,1]+perf_dec_sem[:,1],color='brown',alpha=0.5)
ax.plot(np.arange(t_steps),perf_dec_m[:,2],color='black')
ax.fill_between(np.arange(t_steps),perf_dec_m[:,2]-perf_dec_sem[:,2],perf_dec_m[:,2]+perf_dec_sem[:,2],color='black',alpha=0.5)
ax.plot(np.arange(t_steps),0.5*np.ones(t_steps),color='black',linestyle='--')
ax.set_ylim([0.4,1])
ax.set_ylabel('Decoding Performance')
ax.set_xlabel('Time')
if save_fig:
    fig.savefig('/home/ramon/Dropbox/Esteki_Kiani/plots/figure_decoding_dec_ctx_neu_%i_rr%i%i_new.pdf'%(n_neu,wei_ctx[0],wei_ctx[1]),dpi=500,bbox_inches='tight')
    fig.savefig('/home/ramon/Dropbox/Esteki_Kiani/plots/figure_decoding_dec_ctx_neu_%i_rr%i%i_new.png'%(n_neu,wei_ctx[0],wei_ctx[1]),dpi=500,bbox_inches='tight')

# Shifted-CCGP
perf_abs_m=np.mean(perf_abs,axis=0)
# plt.plot(bias_vec,perf_abs[hh,j,:,0,0],color='blue')
# plt.plot(bias_vec,perf_abs[hh,j,:,0,1],color='royalblue')
# plt.plot(bias_vec,perf_abs[hh,j,:,1,0],color='brown')
# plt.plot(bias_vec,perf_abs[hh,j,:,1,1],color='orange')
# plt.ylim([0.4,1])
# plt.show()
# plt.close()

for j in range(t_steps):
    plt.plot(bias_vec,perf_abs_m[j,:,0,0],color='blue')
    plt.plot(bias_vec,perf_abs_m[j,:,0,1],color='royalblue')
    plt.plot(bias_vec,perf_abs_m[j,:,1,0],color='brown')
    plt.plot(bias_vec,perf_abs_m[j,:,1,1],color='orange')
    plt.ylim([0.4,1])
    plt.show()
    plt.close()

# MDS
#pair_mat=nan*np.zeros((n_files,2*len(coh_uq),2*len(coh_uq),2))
# for o in range(2*len(coh_uq)):
#     for oo in range(2*len(coh_uq)):
#         ind1=np.where((coherence[correct]==coh_uq[int(o%len(coh_uq))])&(context[correct]==(ctx_uq[int(o/len(coh_uq))])))[0]
#         ind2=np.where((coherence[correct]==coh_uq[int(oo%len(coh_uq))])&(context[correct]==(ctx_uq[int(oo/len(coh_uq))])))[0]
#         nt_min=np.min(np.array([len(ind1),len(ind2)]))
#         nt_rnd1=np.random.choice(ind1,nt_min,replace=False)
#         nt_rnd2=np.random.choice(ind2,nt_min,replace=False)
#         nt_rnd_pre=np.concatenate((nt_rnd1,nt_rnd2))
#         clase_pre=np.zeros(2*nt_min)
#         clase_pre[0:nt_min]=1
#         ind_sh=np.random.permutation(np.arange(2*nt_min))
#         nt_rnd=nt_rnd_pre[ind_sh]
#         clase=clase_pre[ind_sh]
#         pair_mat[hh,o,oo]=classifier(ut_test[correct][nt_rnd][:,-1],clase,reg=1)

# plt.imshow(pair_mat[hh,:,:,1])
# plt.show()

# embedding=MDS(n_components=3)
# pseudo_mds_ctx=embedding.fit_transform(pair_mat[hh,:,:,1])

# col=['darkgreen','darkgreen','darkgreen','darkgreen','darkgreen','black','darkgoldenrod','darkgoldenrod','darkgoldenrod','darkgoldenrod','darkgoldenrod','purple','purple','purple','purple','purple','black','darkblue','darkblue','darkblue','darkblue','darkblue']
# alph=[1,0.8,0.6,0.4,0.2,1,0.2,0.4,0.6,0.8,1,1,0.8,0.6,0.4,0.2,1,0.2,0.4,0.6,0.8,1.0]
# fig = plt.figure()#figsize=(2,2)
# ax = fig.add_subplot(111, projection='3d')
# for jj in range(2*len(coh_uq)):
#     ax.scatter(pseudo_mds_ctx[jj,0],pseudo_mds_ctx[jj,1],pseudo_mds_ctx[jj,2],color=col[jj],alpha=alph[jj])
# #ax.plot_surface(xx, yy, z, color='black',alpha=0.2)
# ax.set_xlabel('PC1')
# ax.set_ylabel('PC2')
# ax.set_zlabel('PC3')
# #ax.set_xlim([-7,7])
# #ax.set_ylim([-4,4])
# #ax.set_zlim([-4,4])
# plt.show()
# plt.close(fig)
