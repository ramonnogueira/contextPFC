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
    
def dimensionality(pseudo_tr,pseudo_te,clase_coh,coh_use,clase_coh_uq,n_dim):
    num_neu=len(pseudo_tr[0])
    len_1=len(coh_use)
    clase_dim_pre=np.zeros(len_1)
    clase_dim_pre[int(0.5*len_1):]=1
    dim_vec=nan*np.zeros(n_dim)
    for d in range(n_dim):
        clase_dim=np.random.permutation(clase_dim_pre)
        clase_d_pre=nan*np.zeros(len(clase_coh))
        pseudo_tr_d_pre=nan*np.zeros((len(clase_d_pre),num_neu))
        pseudo_te_d_pre=nan*np.zeros((len(clase_d_pre),num_neu))
        for l in range(len(coh_use)):
            ind_l=np.where(clase_coh==coh_use[l])[0]
            clase_d_pre[ind_l]=clase_dim[l]
            pseudo_tr_d_pre[ind_l]=pseudo_tr[ind_l]
            pseudo_te_d_pre[ind_l]=pseudo_te[ind_l]                    
        ind_nonan=~np.isnan(clase_d_pre)
        clase_d=clase_d_pre[ind_nonan]
        pseudo_tr_d=pseudo_tr_d_pre[ind_nonan]
        pseudo_te_d=pseudo_te_d_pre[ind_nonan]
        cl=LogisticRegression(C=1/reg,class_weight='balanced')
        cl.fit(pseudo_tr_d,clase_d)
        dim_vec[d]=cl.score(pseudo_te_d,clase_d)
    return np.mean(dim_vec)

def ps_score(wei0,wei1):
    y1=np.dot(wei0,wei1)
    y2=(np.linalg.norm(wei0)*np.linalg.norm(wei1))
    return y1/y2

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

# def line_repre(repr_tr,repr_te,n_coh,nt):
#     n_neu=len(repr_tr[0])
#     repr_tr_pert=nan*np.zeros(np.shape(repr_tr))
#     repr_te_pert=nan*np.zeros(np.shape(repr_te))
#     for i in range(2*n_coh):
#         if ii
#         pert=np.random.normal(0,pert_std,n_neu)
#         for ii in range(nt):
#             repr_tr_pert[i*nt+ii]=(repr_tr[i*nt+ii]+pert)
#             repr_te_pert[i*nt+ii]=(repr_te[i*nt+ii]+pert)
#     return repr_tr_pert,repr_te_pert

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

def abstraction_2D(feat_decod,feat_binary,bias):
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
         pred=(np.dot(feat_decod[ind_test],supp.coef_[0])+supp.intercept_+bias)>0
         perf[k,kk]=np.mean(pred==task[ind_test])
         #perf[k,kk,0]=supp.score(feat_decod[ind_train],task[ind_train])
         #perf[k,kk,1]=supp.score(feat_decod[ind_test],task[ind_test])
    return perf

def significance_bootstrap(hist1,hist2,n_shuff):
    hist=np.concatenate((hist1,hist2))
    labels=np.zeros(len(hist))
    labels[int(len(hist)/2):]=1
    diff=(np.mean(hist[labels==0])-np.mean(hist[labels==1]))
    diff_vec=nan*np.zeros(n_shuff)
    for i in range(n_shuff):
        labels_p=permutation(labels)
        diff_vec[i]=(np.mean(hist[labels_p==0])-np.mean(hist[labels_p==1]))
    # P-value
    diff_sort=np.sort(diff_vec)
    med=np.median(diff_sort)
    if diff<med:
        pvalue=2*len(diff_sort[diff_sort<diff])/len(diff_sort)
    if diff>med:
        pvalue=2*len(diff_sort[diff_sort>diff])/len(diff_sort)
    return diff, diff_vec, pvalue


##############################################

monkeys=['Galileo']#'Niels']#,]

# target onset: 'targ_on', dots onset: 'dots_on', dots offset: 'dots_off', saccade: 'response_edf'
talig='response_edf'
dic_time=np.array([250,-50,200,200])# time pre, time post, bin size, step size (time pre always positive)
steps=int((dic_time[0]+dic_time[1])/dic_time[3])
xx=np.linspace(-dic_time[0]/1000,dic_time[1]/1000,steps,endpoint=False)

nt=100 #100 for coh signed, 200 for coh unsigned, 50 for coh signed with context
n_rand=10
n_dim=100
perc_tr=0.8
thres=0
reg=1e-3
n_coh=15

group_ref=np.array([-7 ,-6 ,-5 ,-4 ,-3 ,-2 ,-1 ,0  ,1  ,2  ,3  ,4  ,5  ,6  ,7])
#group_coh=np.array([nan,-2 ,-2 ,-2 ,-1 ,-1 ,-1 ,nan,1  ,1  ,1  ,2  ,2  ,2  ,nan])
#group_coh=np.array([nan,0 ,0 ,0 ,1 ,1 ,1 ,nan,1  ,1  ,1  ,0  ,0  ,0  ,nan])
#group_coh=np.array([nan,nan,0  ,0  ,0  ,0  ,0  ,nan,1  ,1  ,1  ,1  ,1  ,nan,nan])
#group_coh=np.array([nan,0  ,0  ,0  ,0  ,0  ,0  ,nan,1  ,1  ,1  ,1  ,1  ,1  ,nan])
group_coh=np.array([0  ,0  ,0  ,0  ,0  ,0  ,0  ,nan,1  ,1  ,1  ,1  ,1  ,1  ,1  ])

tpre_sacc=50

n_shuff_sig=10000
bias_vec=np.linspace(-3,3,31)

# Niels
#files_groups=[[0,4],[4,8],[8,12]]
#files_groups=[[0,3],[3,6],[6,9],[9,12]]
#files_groups=[[0,2],[2,4],[4,6],[6,8],[8,10],[10,12]]
#files_groups=[[0,1],[1,2],[2,3],[3,4],[4,5],[5,6],[6,7],[7,8],[8,9],[9,10],[10,11],[11,12]]

# Galileo
#files_groups=[[0,10],[10,20],[20,30]]
files_groups=[[0,5],[5,10],[10,15],[15,20],[20,25],[25,30]]
#files_groups=[[0,3],[3,6],[6,9],[9,12],[12,15],[15,18],[18,21],[21,24],[24,27],[27,30]]

for k in range(len(monkeys)):
    print (monkeys[k])
    #abs_path='/home/ramon/Dropbox/Esteki_Kiani/data/sorted/late/%s/'%(monkeys[k])
    abs_path='/home/ramon/Dropbox/Esteki_Kiani/data/unsorted/%s/'%(monkeys[k])
    files=miscellaneous.order_files(np.array(os.listdir(abs_path)))
    perf_all=nan*np.zeros((len(files_groups),n_rand,3))
    ccgp_all=nan*np.zeros((len(files_groups),n_rand,len(bias_vec),2,2))

    for i,nn in enumerate(files_groups):
        # Careful! in this function I am only using correct trials so that choice and stimulus are the same
        pseudo=miscellaneous.pseudopop_coherence_context_correct(abs_path,files[nn[0]:nn[1]],talig,dic_time,steps,thres,nt,n_rand,perc_tr,True,tpre_sacc,group_ref)
        pseudo_all=pseudo['pseudo_all']
        pseudo_tr=pseudo['pseudo_tr']
        pseudo_te=pseudo['pseudo_te']
        neu_total=len(pseudo_tr[0,0,0])
        print (np.shape(pseudo_tr))
        context=pseudo['clase_ctx']
        clase_all=pseudo['clase_all']
        coherence=pseudo['clase_coh']
        clase_coh=clase_resolution(group_coh,coherence)
        indnan=~np.isnan(clase_coh)

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
            print (' ',ii)
            pseudo_flat=nan*np.zeros((pseudo_all.shape[2],steps*neu_total))
            pseudo_tr_flat=nan*np.zeros((pseudo_tr.shape[2],steps*neu_total))
            pseudo_te_flat=nan*np.zeros((pseudo_te.shape[2],steps*neu_total))
            for pp in range(steps):
                pseudo_flat[:,pp*neu_total:(pp+1)*neu_total]=pseudo_all[pp,ii]
                pseudo_tr_flat[:,pp*neu_total:(pp+1)*neu_total]=pseudo_tr[pp,ii]
                pseudo_te_flat[:,pp*neu_total:(pp+1)*neu_total]=pseudo_te[pp,ii]

            sum_nan=np.sum(np.isnan(pseudo_tr_flat),axis=1)
            indnan_flat=(sum_nan==0)
            ind_nonan=(indnan*indnan_flat)

            #cl=LogisticRegression(C=1/reg,class_weight='balanced')
            cl=LinearSVC(C=1/reg,class_weight='balanced')
            cl.fit(pseudo_tr_flat[ind_nonan],feat_binary[:,0][ind_nonan])
            perf_all[i,ii,0]=cl.score(pseudo_te_flat[ind_nonan],feat_binary[:,0][ind_nonan])
            #cl=LogisticRegression(C=1/reg,class_weight='balanced')
            cl=LinearSVC(C=1/reg,class_weight='balanced')
            cl.fit(pseudo_tr_flat[ind_nonan],feat_binary[:,1][ind_nonan])
            perf_all[i,ii,1]=cl.score(pseudo_te_flat[ind_nonan],feat_binary[:,1][ind_nonan])
            #cl=LogisticRegression(C=1/reg,class_weight='balanced')
            xor=np.sum(feat_binary,axis=1)%2
            cl=LinearSVC(C=1/reg,class_weight='balanced')
            cl.fit(pseudo_tr_flat[ind_nonan],xor[ind_nonan])
            perf_all[i,ii,2]=cl.score(pseudo_te_flat[ind_nonan],xor[ind_nonan])
            # CCGP
            for f in range(len(bias_vec)):
                ccgp_all[i,ii,f]=abstraction_2D(pseudo_flat[ind_nonan],feat_binary[ind_nonan],bias=bias_vec[f])

    perf_all_m=np.mean(perf_all,axis=1)
    ccgp_all_m=np.mean(ccgp_all,axis=1)
    #print (ccgp_all_m)
    print (ccgp_all_m[:,15])
    print (np.max(ccgp_all_m,axis=1))
    print (perf_all_m)
    label_vec=['Choice','Context','XOR']
    for ll in range(3):
        #for lll in range(len(files_groups)): 
        sig01=significance_bootstrap(perf_all[0,:,ll],perf_all[1,:,ll],n_shuff_sig)
        sig02=significance_bootstrap(perf_all[0,:,ll],perf_all[2,:,ll],n_shuff_sig)
        sig12=significance_bootstrap(perf_all[1,:,ll],perf_all[2,:,ll],n_shuff_sig)
        print (label_vec[ll],sig01[0],sig01[2])
        print (label_vec[ll],sig02[0],sig02[2])
        print (label_vec[ll],sig12[0],sig12[2])
    

#     ###################################
#     # Plot performance Tasks and XOR
#     fig=plt.figure(figsize=(2.3,2))
#     ax=fig.add_subplot(111)
#     miscellaneous.adjust_spines(ax,['left','bottom'])
#     ax.scatter(np.arange(3),perf_all_m)
#     ax.plot(np.arange(3),0.5*np.ones(3),color='black',linestyle='--')
#     ax.set_ylim([0.4,1])
#     ax.set_ylabel('Probability Right Response')
    #ax.set_xlabel('Evidence Right Choice (%)')
    #plt.xticks(xx[indnan0],coh_plot[k][indnan0])
    #plt.xticks(xx,coh_plot[1])
    #fig.savefig('/home/ramon/Dropbox/Esteki_Kiani/figures/figure_neuro_pseudo_perf_tasks_xor_%s_all_saccade_late.pdf'%(monkeys[k]),dpi=500,bbox_inches='tight')

    ###################################
    # # Plot Shifted CCGP
    # fig=plt.figure(figsize=(2.3,2))
    # ax=fig.add_subplot(111)
    # miscellaneous.adjust_spines(ax,['left','bottom'])
    # ax.plot(bias_vec,ccgp_all_m[:,0,0],color='blue')
    # ax.plot(bias_vec,ccgp_all_m[:,0,1],color='royalblue')
    # ax.plot(bias_vec,ccgp_all_m[:,1,0],color='brown')
    # ax.plot(bias_vec,ccgp_all_m[:,1,1],color='orange')
    # ax.plot(bias_vec,0.5*np.ones(len(bias_vec)),color='black',linestyle='--')
    # ax.set_ylim([0.4,1])
    # ax.set_ylabel('Shifted-CCGP')
    # ax.set_xlabel('Shift')
    # #plt.xticks(xx[indnan0],coh_plot[k][indnan0])
    # #plt.xticks(xx,coh_plot[1])
    # fig.savefig('/home/ramon/Dropbox/Esteki_Kiani/figures/figure_neuro_pseudo_shifted_ccgp_%s_all_late.pdf'%(monkeys[k]),dpi=500,bbox_inches='tight')
 
       
    # for i in range(steps):
    #     #print (i)
    #     for ii in range(n_rand):
    #         #print (' ',ii)
    #         try:
    #             #cl=LogisticRegression(C=1/reg,class_weight='balanced')
    #             cl=LinearSVC(C=1/reg,class_weight='balanced')
    #             cl.fit(pseudo_tr[i,ii][indnan],clase_coh[indnan])
    #             perf_coh[i,ii]=cl.score(pseudo_te[i,ii][indnan],clase_coh[indnan])
    #             #cl=LogisticRegression(C=1/reg,class_weight='balanced')
    #             cl=LinearSVC(C=1/reg,class_weight='balanced')
    #             cl.fit(pseudo_tr[i,ii][indnan],context[indnan])
    #             perf_ctx[i,ii]=cl.score(pseudo_te[i,ii][indnan],context[indnan])
    #         except:
    #             None
            
    # print (np.mean(perf_coh,axis=1))
    # print (np.mean(perf_ctx,axis=1))
        
    
    # clase_coh_uq=np.unique(clase_coh)    
    # # ps_coh_ctx=nan*np.zeros((steps,n_rand))
    # # ps_ctx_coh=nan*np.zeros((steps,n_rand))
    # dim_pre=nan*np.zeros((steps,n_rand)) # Shattering dimensionality
    # dim_pre_pert=nan*np.zeros((steps,n_rand)) # Shattering dimensionality
    
    # #dim_pre_sh=nan*np.zeros((steps,n_sh,n_rand))
    # ps_score_pairs=nan*np.zeros((steps,n_rand,2*len(coh_use)-2,2*len(coh_use)-2))
    # ccgp_pairs=nan*np.zeros((steps,n_rand,2,len(coh_use)-1,len(coh_use)-1))
    # perf_pairs=nan*np.zeros((steps,n_rand,2,len(coh_use)-1)) # Performance between pairs of coherences in context 0 and 1
    # #ps_score_pairs_sh=nan*np.zeros((steps,n_sh,n_rand,2*len(coh_use)-2,2*len(coh_use)-2))
    # #perf_pairs_sh=nan*np.zeros((steps,n_sh,n_rand,2,len(coh_use)-1))
    # for i in range(steps):
    #     print (i)
    #     for ii in range(n_rand):
    #         print ('  ',ii)
    #         pert_std=0.5
    #         repr_tr,repr_te=null_model_coh(pseudo_tr[i,ii],pseudo_te[i,ii],pert_std,n_coh,nt)
    #         #dim_pre[i,ii]=dimensionality(pseudo_tr[i,ii],pseudo_te[i,ii],clase_coh,coh_use,clase_coh_uq,n_dim)
    #         #dim_pre_pert[i,ii]=dimensionality(repr_tr,repr_te,clase_coh,coh_use,clase_coh_uq,n_dim)

    #         # Parallelism Score
    #         wei_cl=nan*np.zeros((2*len(coh_use)-2,num_neu))
    #         for j in range(len(coh_use)-1):
    #             ind_cl0=np.where(((clase_coh==coh_use[j])|(clase_coh==coh_use[j+1]))&(context==0))[0]
    #             ind_cl1=np.where(((clase_coh==coh_use[j])|(clase_coh==coh_use[j+1]))&(context==1))[0]
    #             cl0=LogisticRegression(C=1/reg,class_weight='balanced')
    #             #cl0.fit(pseudo_tr[i,ii,ind_cl0],clase_coh[ind_cl0])
    #             #perf_pairs[i,ii,0,j]=cl0.score(pseudo_te[i,ii,ind_cl0],clase_coh[ind_cl0])
    #             cl0.fit(repr_tr[ind_cl0],clase_coh[ind_cl0])
    #             perf_pairs[i,ii,0,j]=cl0.score(repr_te[ind_cl0],clase_coh[ind_cl0])
    #             wei_cl[j]=cl0.coef_[0]
    #             cl1=LogisticRegression(C=1/reg,class_weight='balanced')
    #             #cl1.fit(pseudo_tr[i,ii,ind_cl1],clase_coh[ind_cl1])
    #             #perf_pairs[i,ii,1,j]=cl1.score(pseudo_te[i,ii,ind_cl1],clase_coh[ind_cl1])
    #             cl1.fit(repr_tr[ind_cl1],clase_coh[ind_cl1])
    #             perf_pairs[i,ii,1,j]=cl1.score(repr_te[ind_cl1],clase_coh[ind_cl1])
    #             wei_cl[j+len(coh_use)-1]=cl1.coef_[0]
    #         for j in range(len(wei_cl)):
    #             for jj in range(j+1,len(wei_cl)):
    #                 ps_score_pairs[i,ii,j,jj]=abs(ps_score(wei_cl[j],wei_cl[jj]))

    #         # # CCGP Context0
    #         # for j in range(len(coh_use)-1):
    #         #     for jj in range(len(coh_use)-1):
    #         #         ind_cl0=np.where(((clase_coh==coh_use[j])|(clase_coh==coh_use[j+1]))&(context==0))[0]
    #         #         ind_cl1=np.where(((clase_coh==coh_use[jj])|(clase_coh==coh_use[jj+1]))&(context==1))[0]
                    
    #         #         cl0=LogisticRegression(C=1/reg,class_weight='balanced')
    #         #         cl0.fit(pseudo_tr[i,ii,ind_cl0],clase_coh[ind_cl0])
    #         #         ccgp_pairs[i,ii,0,j,jj]=cl0.score(pseudo_te[i,ii,ind_cl1],clase_coh[ind_cl1])
                    
    #         # # CCGP Context1
    #         # for j in range(len(coh_use)-1):
    #         #     for jj in range(len(coh_use)-1):
    #         #         ind_cl0=np.where(((clase_coh==coh_use[j])|(clase_coh==coh_use[j+1]))&(context==0))[0]
    #         #         ind_cl1=np.where(((clase_coh==coh_use[jj])|(clase_coh==coh_use[jj+1]))&(context==1))[0]
                    
    #         #         cl1=LogisticRegression(C=1/reg,class_weight='balanced')
    #         #         cl1.fit(pseudo_tr[i,ii,ind_cl1],clase_coh[ind_cl1])
    #         #         ccgp_pairs[i,ii,1,j,jj]=cl1.score(pseudo_te[i,ii,ind_cl0],clase_coh[ind_cl0])
            
    #     print (np.mean(perf_pairs,axis=1))
    #     plt.imshow(np.mean(ps_score_pairs,axis=1)[i])
    #     plt.colorbar()
    #     plt.show()

    #     # plt.imshow(np.mean(ccgp_pairs,axis=1)[i,0])
    #     # plt.colorbar()
    #     # plt.show()

    #     # plt.imshow(np.mean(ccgp_pairs,axis=1)[i,1])
    #     # plt.colorbar()
    #     # plt.show()

    #     # Shuffle
    #     # for ss in range(n_sh):
    #     #     index_sh=index_shuffle(num_neu,clase_all)
    #     #     for ii in range(n_rand):
    #     #         print ('  ',ii)
    #     #         pseudo_tr_sh,pseudo_te_sh=shuffle_distr(pseudo_tr[i,ii],pseudo_te[i,ii],clase_all,index_sh)
    #     #         dim_pre_sh[i,ss,ii]=dimensionality(pseudo_tr_sh,pseudo_te_sh,clase_coh,coh_use,clase_coh_uq,n_dim)

    #     #         wei_cl=nan*np.zeros((2*len(coh_use)-2,num_neu))
    #     #         for j in range(len(coh_use)-1):
    #     #             ind_cl0=np.where(((clase_coh==coh_use[j])|(clase_coh==coh_use[j+1]))&(context==0))[0]
    #     #             ind_cl1=np.where(((clase_coh==coh_use[j])|(clase_coh==coh_use[j+1]))&(context==1))[0]
                    
    #     #             cl0=LogisticRegression(C=1/reg,class_weight='balanced')
    #     #             cl0.fit(pseudo_tr_sh[ind_cl0],clase_coh[ind_cl0])
    #     #             perf_pairs_sh[i,ss,ii,0,j]=cl0.score(pseudo_te_sh[ind_cl0],clase_coh[ind_cl0])
    #     #             wei_cl[j]=cl0.coef_[0]
                    
    #     #             cl1=LogisticRegression(C=1/reg,class_weight='balanced')
    #     #             cl1.fit(pseudo_tr_sh[ind_cl1],clase_coh[ind_cl1])
    #     #             perf_pairs_sh[i,ss,ii,1,j]=cl1.score(pseudo_te_sh[ind_cl1],clase_coh[ind_cl1])
    #     #             wei_cl[j+len(coh_use)-1]=cl1.coef_[0]

    #     #         for j in range(len(wei_cl)):
    #     #             for jj in range(j+1,len(wei_cl)):
    #     #                 ps_score_pairs_sh[i,ss,ii,j,jj]=ps_score(wei_cl[j],wei_cl[jj])

    #         # print (np.mean(perf_pairs_sh,axis=2))
    #         # print (np.mean(dim_pre,axis=1)[i],np.mean(dim_pre_sh,axis=2)[i,ss])
    #         # plt.imshow(np.mean(ps_score_pairs_sh,axis=2)[i,ss])
    #         # plt.colorbar()
    #         # plt.show()

    # # dim_vec=np.mean(dim_pre,axis=(1))
    # # dim_vec_std=np.std(dim_pre,axis=(1))
    # # plt.errorbar(xx,dim_vec,yerr=dim_vec_std)
    # # plt.plot(xx,0.5*np.ones(len(xx)),color='black',linestyle='--')
    # # plt.xlabel('Time (sec)')
    # # plt.ylabel('Shattering Dim.')
    # # plt.ylim([0.4,1.1])
    # # plt.show()

    # dim_pert=np.mean(dim_pre_pert,axis=(1))
    # dim_pert_std=np.std(dim_pre_pert,axis=(1))
    # plt.errorbar(xx,dim_pert,yerr=dim_pert_std)
    # plt.plot(xx,0.5*np.ones(len(xx)),color='black',linestyle='--')
    # plt.xlabel('Time (sec)')
    # plt.ylabel('Shattering Dim.')
    # plt.ylim([0.4,1.1])
    # plt.show()

    #         # # Coherence wrt to context
    #         # ind_c0=np.where((clase_ctx==0)&((clase_coh==1)|(clase_coh==3)))[0]
    #         # ind_c1=np.where((clase_ctx==1)&((clase_coh==1)|(clase_coh==3)))[0]
    #         # cl_c0=LogisticRegression(C=1/reg,class_weight='balanced')
    #         # cl_c0.fit(pseudo_all[i,ii,ind_c0],clase_coh[ind_c0])
    #         # wei0=cl_c0.coef_[0]
    #         # cl_c1=LogisticRegression(C=1/reg,class_weight='balanced')
    #         # cl_c1.fit(pseudo_all[i,ii,ind_c1],clase_coh[ind_c1])
    #         # wei1=cl_c1.coef_[0]
    #         # ps_coh_ctx[i,ii]=np.dot(wei0,wei1)/(np.linalg.norm(wei0)*np.linalg.norm(wei1))

    #         # # Context wrt to coherence
    #         # ind_c0=np.where(clase_coh==1)[0]
    #         # ind_c1=np.where(clase_coh==3)[0]
    #         # cl_c0=LogisticRegression(C=1/reg,class_weight='balanced')
    #         # cl_c0.fit(pseudo_all[i,ii,ind_c0],clase_ctx[ind_c0])
    #         # wei0=cl_c0.coef_[0]
    #         # cl_c1=LogisticRegression(C=1/reg,class_weight='balanced')
    #         # cl_c1.fit(pseudo_all[i,ii,ind_c1],clase_ctx[ind_c1])
    #         # wei1=cl_c1.coef_[0]
    #         # ps_ctx_coh[i,ii]=np.dot(wei0,wei1)/(np.linalg.norm(wei0)*np.linalg.norm(wei1))
            
    # # # Coherence wrt Context
    # # ps_coh_ctx_m=np.mean(ps_coh_ctx,axis=1)
    # # ps_coh_ctx_std=np.std(ps_coh_ctx,axis=1)
    # # plt.errorbar(xx,ps_coh_ctx_m,yerr=ps_coh_ctx_std)
    # # plt.plot(xx,0*np.ones(len(xx)),color='black',linestyle='--')
    # # plt.xlabel('Time (sec)')
    # # plt.ylabel('PS Coherence wrt Context')
    # # plt.ylim([-0.1,1])
    # # plt.show()
    
    # # # Context wrt Coherence
    # # ps_ctx_coh_m=np.mean(ps_ctx_coh,axis=1)
    # # ps_ctx_coh_std=np.std(ps_ctx_coh,axis=1)
    # # plt.errorbar(xx,ps_ctx_coh_m,yerr=ps_ctx_coh_std)
    # # plt.plot(xx,0*np.ones(len(xx)),color='black',linestyle='--')
    # # plt.xlabel('Time (sec)')
    # # plt.ylabel('PS Context wrt Coherence')
    # # plt.ylim([-0.1,1])
    # # plt.show()  
        
