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
        
def calculate_everything(monkey,group_coh_vec,bias_vec,abs_path,files,talig,dic_time,steps,thres,nt,n_rand,n_rot,perc_tr,tpre_sacc,group_ref,shuff):
    perf_all=nan*np.zeros((steps_all,n_rand,3))
    ccgp_all=nan*np.zeros((steps_all,n_rand,len(bias_vec),2,2))
    ccgp_rot_all=nan*np.zeros((n_rot,steps_all,n_rand,len(bias_vec),2,2))
    
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
            #pseudo_rot_tr=create_rot(pseudo_tr[ii],index_cat,index_rot[0])
            #pseudo_rot_te=create_rot(pseudo_te[ii],index_cat,index_rot[0])

            #print (' ',ii)
            sum_nan=np.sum(np.isnan(pseudo_tr[ii]),axis=1)
            indnan_flat=(sum_nan==0) # True will be used, False discarded
            ind_nonan=(indnan*indnan_flat) # Index used combination of discarded from RT and discarded from group_coh
            print (np.sum(ind_nonan))
            # sum_nan=np.sum(np.isnan(pseudo_rot_tr),axis=1)
            # indnan_flat=(sum_nan==0) # True will be used, False discarded
            # ind_nonan=(indnan*indnan_flat) # Index used combination of discarded from RT and discarded from group_coh
                
            if monkey=='Niels':
                neu_rnd=np.arange(neu_total)
            if monkey=='Galileo':
                neu_rnd=np.arange(neu_total)
                #n_max=96*4 # 96 channels, 4 files
                #neu_rnd=np.sort(np.random.choice(np.arange(neu_total),n_max,replace=False)) # Careful!!!
            
            # # Choice
            # cl=LogisticRegression(C=1/reg,class_weight='balanced')
            # #cl=LinearSVC(C=1/reg,class_weight='balanced')
            # cl.fit(pseudo_tr[ii][ind_nonan][:,neu_rnd],feat_binary[:,0][ind_nonan])
            # perf_all[kk,ii,0]=cl.score(pseudo_te[ii][ind_nonan][:,neu_rnd],feat_binary[:,0][ind_nonan])
            # # Context
            # cl=LogisticRegression(C=1/reg,class_weight='balanced')
            # #cl=LinearSVC(C=1/reg,class_weight='balanced')
            # cl.fit(pseudo_tr[ii][ind_nonan][:,neu_rnd],feat_binary[:,1][ind_nonan])
            # perf_all[kk,ii,1]=cl.score(pseudo_te[ii][ind_nonan][:,neu_rnd],feat_binary[:,1][ind_nonan])
            # # XOR
            # cl=LogisticRegression(C=1/reg,class_weight='balanced')
            # #cl=LinearSVC(C=1/reg,class_weight='balanced')
            # xor=np.sum(feat_binary,axis=1)%2
            # cl.fit(pseudo_tr[ii][ind_nonan][:,neu_rnd],xor[ind_nonan])
            # perf_all[kk,ii,2]=cl.score(pseudo_te[ii][ind_nonan][:,neu_rnd],xor[ind_nonan])

            # cl=LogisticRegression(C=1/reg,class_weight='balanced')
            # #cl=LinearSVC(C=1/reg,class_weight='balanced')
            # cl.fit(pseudo_rot_tr[ind_nonan][:,neu_rnd],feat_binary[:,0][ind_nonan])
            # perf_all[kk,ii,0]=cl.score(pseudo_rot_te[ind_nonan][:,neu_rnd],feat_binary[:,0][ind_nonan])
            # # Context
            # cl=LogisticRegression(C=1/reg,class_weight='balanced')
            # #cl=LinearSVC(C=1/reg,class_weight='balanced')
            # cl.fit(pseudo_rot_tr[ind_nonan][:,neu_rnd],feat_binary[:,1][ind_nonan])
            # perf_all[kk,ii,1]=cl.score(pseudo_rot_te[ind_nonan][:,neu_rnd],feat_binary[:,1][ind_nonan])
            # # XOR
            # cl=LogisticRegression(C=1/reg,class_weight='balanced')
            # #cl=LinearSVC(C=1/reg,class_weight='balanced')
            # xor=np.sum(feat_binary,axis=1)%2
            # cl.fit(pseudo_rot_tr[ind_nonan][:,neu_rnd],xor[ind_nonan])
            # perf_all[kk,ii,2]=cl.score(pseudo_rot_te[ind_nonan][:,neu_rnd],xor[ind_nonan])
     
            # CCGP
            # for f in range(len(bias_vec)):
            #     ccgp=abstraction_2D(pseudo_all[ii][ind_nonan][:,neu_rnd],feat_binary[ind_nonan],bias=bias_vec[f],reg=reg)
            #     ccgp_all[kk,ii,f]=ccgp[0]

            # # Distribution of ccgp after breaking geometry through rotations
            # for n in range(n_rot):
            #     #print ('rot ',n)
            #     pseudo_rot=create_rot(pseudo_all[ii],index_cat,index_rot[n])
            #     for f in range(len(bias_vec)):
            #         ccgp_rot=abstraction_2D(pseudo_rot[ind_nonan][:,neu_rnd],feat_binary[ind_nonan],bias=bias_vec[f],reg=reg)
            #         ccgp_rot_all[n,kk,ii,f]=ccgp_rot[0]

    return perf_all,ccgp_all,ccgp_rot_all

def calculate_shccgp(ccgp_all,steps,steps_all,n_rand):
    shccgp_pre=nan*np.zeros((steps_all,n_rand,2,2))
    for p in range(steps):
        for pp in range(n_rand):
            for ppp in range(2):
                shccgp_pre[p,pp,ppp,0]=np.max(ccgp_all[p,pp,:,ppp,0])
                shccgp_pre[p,pp,ppp,1]=np.max(ccgp_all[p,pp,:,ppp,1])
    return shccgp_pre


##############################################

monkeys=['Galileo']
talig='dots_on'

#bias_vec=np.linspace(-20,20,31) #Niels
bias_vec=np.linspace(-15,15,31) #Galileo

nt=100 #100 for coh signed, 200 for coh unsigned, 50 for coh signed with context
n_rand=10
n_shuff=0
perc_tr=0.8
thres=0
reg=1e2
n_coh=15
tpre_sacc=50

group_ref=np.array([-7 ,-6 ,-5 ,-4 ,-3 ,-2 ,-1 ,0  ,1  ,2  ,3  ,4  ,5  ,6  ,7  ])
group_coh_vec=np.array([0  ,0  ,0  ,0  ,0  ,0  ,0  ,nan,1  ,1  ,1  ,1  ,1  ,1  ,1  ])


for hh in range(len(monkeys)):
    monkey=monkeys[hh]
    if monkey=='Niels':
        dic_time=np.array([0,600,200,200]) # time pre, time post, bin size, step size
        xx_coh_pre=np.array([-75,-51.2,-25.6,-12.8,-6.4,-3.2,-1.6,0,1.6,3.2,6.4,12.8,25.6,51.2,75])
        xx_plot=np.array(['-75','-51.2','-25.6','-12.8','-6.4','-3.2','-1.6','0','1.6','3.2','6.4','12.8','25.6','51.2','75'])
        ind_ext=np.array([1,2,3,4,5,6,7,8,9,10,11,12,13]) # missing 0 and 14
        ind_put=np.array([0,1,2,3,5,6,7,8,9,11,12,13,14]) # missing 4 and 10
        ind_l=8
        ind_u=12
    if monkey=='Galileo':
        dic_time=np.array([0,800,200,200]) # Careful! time pre, time post, bin size, step size
        xx_coh_pre=np.array([-51.2,-25.6,-12.8,-6.4,-4.5,-3.2,-1.6,0,1.6,3.2,4.5,6.4,12.8,25.6,51.2])
        xx_plot=np.array(['-51.2','-25.6','-12.8','-6.4','-4.5','-3.2','-1.6','0','1.6','3.2','4.5','6.4','12.8','25.6','51.2'])
        ind_ext=np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14])
        ind_put=np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14])
        ind_l=20
        ind_u=30

    steps=int((dic_time[0]+dic_time[1])/dic_time[3])
    xx=np.linspace(-dic_time[0]/1000,dic_time[1]/1000,steps,endpoint=False)

    #abs_path='/home/ramon/Dropbox/Esteki_Kiani/data/sorted/late/%s/'%(monkeys[k])
    abs_path='/home/ramon/Dropbox/Proyectos_Postdoc/Esteki_Kiani/data/unsorted/%s/'%(monkey)
    files_pre=np.array(os.listdir(abs_path))
    order=miscellaneous.order_files(files_pre)
    files=np.array(files_pre[order])[ind_l:ind_u]
    print (files)

    perf_all=nan*np.zeros((len(files_groups),steps,n_rand,3))
    ccgp_all=nan*np.zeros((len(files_groups),steps,n_rand,len(bias_vec),2,2))
    
    for i,nn in enumerate(files_groups):
        print ('group ',i)    
        pseudo=miscellaneous.pseudopop_coherence_context_correct(abs_path,files[nn[0]:nn[1]],talig,dic_time,steps,thres,nt,n_rand,perc_tr,True,tpre_sacc,group_ref,shuff=False,learning=True)   
        for kk in range(steps):
            print (kk)
            # Careful! in this function I am only using correct trials so that choice and stimulus are the same    
            pseudo_all=pseudo['pseudo_all'][kk]
            pseudo_tr=pseudo['pseudo_tr'][kk]
            pseudo_te=pseudo['pseudo_te'][kk]
            context=pseudo['clase_ctx']
            clase_all=pseudo['clase_all']
            coherence=pseudo['clase_coh']

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
    
            for ii in range(n_rand):
                #print (' ',ii)
                sum_nan=np.sum(np.isnan(pseudo_tr[ii]),axis=1)
                indnan_flat=(sum_nan==0) # True will be used, False discarded
                ind_nonan=(indnan*indnan_flat) # Index used combination of discarded from RT and discarded from group_coh
                # Choice
                cl=LogisticRegression(C=1/reg,class_weight='balanced')
                #cl=LinearSVC(C=1/reg,class_weight='balanced')
                cl.fit(pseudo_tr[ii][ind_nonan],feat_binary[:,0][ind_nonan])
                perf_all[i,kk,ii,0]=cl.score(pseudo_te[ii][ind_nonan],feat_binary[:,0][ind_nonan])
                # Context
                cl=LogisticRegression(C=1/reg,class_weight='balanced')
                #cl=LinearSVC(C=1/reg,class_weight='balanced')
                cl.fit(pseudo_tr[ii][ind_nonan],feat_binary[:,1][ind_nonan])
                perf_all[i,kk,ii,1]=cl.score(pseudo_te[ii][ind_nonan],feat_binary[:,1][ind_nonan])
                # XOR
                cl=LogisticRegression(C=1/reg,class_weight='balanced')
                #cl=LinearSVC(C=1/reg,class_weight='balanced')
                xor=np.sum(feat_binary,axis=1)%2
                cl.fit(pseudo_tr[ii][ind_nonan],xor[ind_nonan])
                perf_all[i,kk,ii,2]=cl.score(pseudo_te[ii][ind_nonan],xor[ind_nonan])
                # # CCGP
                for f in range(len(bias_vec)):
                    ccgp=abstraction_2D(pseudo_all[ii][ind_nonan],feat_binary[ind_nonan],bias=bias_vec[f],reg=reg)
                    ccgp_all[i,kk,ii,f]=ccgp[0]

    perf_all_m=np.nanmean(perf_all,axis=2)
    perf_all_std=np.std(perf_all,axis=2)
    print (perf_all_m)
    
    # Plot performance Tasks and XOR vs time
    for t in range(steps):
        fig=plt.figure(figsize=(3,2.5))
        ax=fig.add_subplot(111)
        miscellaneous.adjust_spines(ax,['left','bottom'])
        ax.plot(np.arange(len(files_groups)),perf_all_m[:,t,0],color='blue',label='Direction')
        ax.fill_between(np.arange(len(files_groups)),perf_all_m[:,t,0]-perf_all_std[:,t,0],perf_all_m[:,t,0]+perf_all_std[:,t,0],color='blue',alpha=0.5)
        ax.plot(np.arange(len(files_groups)),perf_all_m[:,t,1],color='brown',label='Context')
        ax.fill_between(np.arange(len(files_groups)),perf_all_m[:,t,1]-perf_all_std[:,t,1],perf_all_m[:,t,1]+perf_all_std[:,t,1],color='brown',alpha=0.5)
        ax.plot(np.arange(len(files_groups)),perf_all_m[:,t,2],color='black',label='XOR')
        ax.fill_between(np.arange(len(files_groups)),perf_all_m[:,t,2]-perf_all_std[:,t,2],perf_all_m[:,t,2]+perf_all_std[:,t,2],color='black',alpha=0.5)
        ax.plot(np.arange(len(files_groups)),0.5*np.ones(len(files_groups)),color='black',linestyle='--')
        ax.set_ylim([0.4,1])
        ax.set_xlabel('Learning Phase')
        ax.set_ylabel('Decoding Performance')
        plt.legend(loc='best')
        fig.savefig('/home/ramon/Dropbox/Esteki_Kiani/plots/choice_ctx_xor_pseudo_tl_%s_%s_learning_time_%i.pdf'%(talig,monkeys[k],t),dpi=500,bbox_inches='tight')
        fig.savefig('/home/ramon/Dropbox/Esteki_Kiani/plots/choice_ctx_xor_pseudo_tl_%s_%s_learning_time_%i.png'%(talig,monkeys[k],t),dpi=500,bbox_inches='tight')
 
    # Plot Shifted CCGP
    ccgp_orig_m=np.mean(ccgp_all[:,:,:,15],axis=(2,4))
    ccgp_orig_std=np.std(np.mean(ccgp_all[:,:,:,15],axis=4),axis=2)
    #ccgp_orig_std=np.std(ccgp_all[:,0,:,15],axis=(1,3))

    shccgp_pre=nan*np.zeros((len(files_groups),steps,n_rand,2,2))
    for o in range(len(files_groups)):
        for p in range(steps):
            for pp in range(n_rand):
                for ppp in range(2):
                    shccgp_pre[o,p,pp,ppp,0]=np.max(ccgp_all[o,p,pp,:,ppp,0])
                    shccgp_pre[o,p,pp,ppp,1]=np.max(ccgp_all[o,p,pp,:,ppp,1])
    shccgp_m=np.mean(shccgp_pre,axis=(2,4))
    #shccgp_std=np.std(shccgp_pre,axis=(1,3))
    shccgp_std=np.std(np.mean(shccgp_pre,axis=4),axis=2)

    for t in range(steps):
        fig=plt.figure(figsize=(3,2.5))
        ax=fig.add_subplot(111)
        miscellaneous.adjust_spines(ax,['left','bottom'])
        ax.plot(np.arange(len(files_groups)),ccgp_orig_m[:,t,0],color='royalblue',label='CCGP Direction')
        ax.fill_between(np.arange(len(files_groups)),ccgp_orig_m[:,t,0]-ccgp_orig_std[:,t,0],ccgp_orig_m[:,t,0]+ccgp_orig_std[:,t,0],color='royalblue',alpha=0.5)
        ax.plot(np.arange(len(files_groups)),ccgp_orig_m[:,t,1],color='orange',label='CCGP Context')
        ax.fill_between(np.arange(len(files_groups)),ccgp_orig_m[:,t,1]-ccgp_orig_std[:,t,1],ccgp_orig_m[:,t,1]+ccgp_orig_std[:,t,1],color='orange',alpha=0.5)
        ax.plot(np.arange(len(files_groups)),shccgp_m[:,t,0],color='blue',label='Sh-CCGP Direction')
        ax.fill_between(np.arange(len(files_groups)),shccgp_m[:,t,0]-shccgp_std[:,t,0],shccgp_m[:,t,0]+shccgp_std[:,t,0],color='blue',alpha=0.5)
        ax.plot(np.arange(len(files_groups)),shccgp_m[:,t,1],color='brown',label='Sh-CCGP Context')
        ax.fill_between(np.arange(len(files_groups)),shccgp_m[:,t,1]-shccgp_std[:,t,1],shccgp_m[:,t,1]+shccgp_std[:,t,1],color='brown',alpha=0.5)
        ax.plot(np.arange(len(files_groups)),0.5*np.ones(len(files_groups)),color='black',linestyle='--')
        ax.set_ylim([0.4,1])
        ax.set_xlabel('Learning Phase')
        ax.set_ylabel('Decoding Performance')
        plt.legend(loc='best')
        fig.savefig('/home/ramon/Dropbox/Esteki_Kiani/plots/ccgp_choice_ctx_xor_pseudo_tl_%s_%s_learning_time_%i.pdf'%(talig,monkeys[k],t),dpi=500,bbox_inches='tight')
        fig.savefig('/home/ramon/Dropbox/Esteki_Kiani/plots/ccgp_choice_ctx_xor_pseudo_tl_%s_%s_learning_time_%i.png'%(talig,monkeys[k],t),dpi=500,bbox_inches='tight')


        # for t in range(steps):
        #     fig=plt.figure(figsize=(3,2.5))
        #     ax=fig.add_subplot(111)
        #     miscellaneous.adjust_spines(ax,['left','bottom'])
        #     ax.plot(bias_vec,np.mean(ccgp_all[t,:,:,0,0],axis=0),color='royalblue',label='Sh-CCGP Direction 1')
        #     ax.plot(bias_vec,np.mean(ccgp_all[t,:,:,0,1],axis=0),color='blue',label='Sh-CCGP Direction 2')
        #     ax.plot(bias_vec,np.mean(ccgp_all[t,:,:,1,0],axis=0),color='orange',label='Sh-CCGP Context 1')
        #     ax.plot(bias_vec,np.mean(ccgp_all[t,:,:,1,1],axis=0),color='brown',label='Sh-CCGP Context 2')
        #     #ax.fill_between(xx,ccgp_orig_m[:,0]-ccgp_orig_std[:,0],ccgp_orig_m[:,0]+ccgp_orig_std[:,0],color='royalblue',alpha=0.5)
        #     ax.plot(bias_vec,0.5*np.ones(len(bias_vec)),color='black',linestyle='--')
        #     ax.set_ylim([0.4,1])
        #     ax.set_xlabel('Bias')
        #     ax.set_ylabel('Decoding Performance')
        #     plt.legend(loc='best')
        #     fig.savefig('/home/ramon/Dropbox/Esteki_Kiani/plots/shifted_ccgp_tl_%s_%s_t%i_learn_%i.pdf'%(talig,monkeys[k],t,i),dpi=500,bbox_inches='tight')

    
 
       
