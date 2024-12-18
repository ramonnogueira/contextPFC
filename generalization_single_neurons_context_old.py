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
from scipy.stats import spearmanr
from numpy.random import permutation
import miscellaneous
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.svm import LinearSVC
from scipy.stats import ortho_group 
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import KFold,StratifiedKFold,StratifiedShuffleSplit
from sklearn.neural_network import MLPClassifier
from scipy import stats
from scipy.optimize import curve_fit
plt.close('all')

# In this script we evaluate generalization throughout learning.
# Behavior: proabability that choice = context after context switch
# Neural: decoding of context right after context switch. Classifier is trained on all the rest of trials (no context switch)

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

def order_files(x):
    ord_pre=[]
    for i in range(len(x)):
        ord_pre.append(x[i][1:9])
    ord_pre=np.array(ord_pre)
    order=np.argsort(ord_pre)
    return order

def calculate_ind_ch_corr(ind_ch,reward):
    n_forw=7
    n_ch=len(ind_ch)
    ind_ch_corr=np.zeros(n_ch)
    for i in range(n_ch):
        aa=(np.arange(n_forw)+ind_ch[i])
        bb_pre=aa*reward[aa]
        bb=bb_pre[bb_pre>0]
        ind_ch_corr[i]=bb[0]
    return np.array(ind_ch_corr,dtype=np.int16)

# This function returns the indices for trials where:
# - context change from 0 to 1 correct stimulus 0
# - context change from 0 to 1 correct stimulus 1
# - context change from 1 to 0 correct stimulus 0
# - context change from 1 to 0 correct stimulus 1 
def calculate_ind_ch_corr2(ind_ch01,ind_ch10,reward,stimulus):
    n_forw=7

    # Context change from 0 to 1
    ind_ch01_s0=[]
    ind_ch01_s1=[]
    n_ch01=len(ind_ch01)
    for i in range(n_ch01):
        aa=(np.arange(n_forw)+ind_ch01[i])
        bb_pre=aa*reward[aa]
        bb=bb_pre[bb_pre>0]
        if stimulus[bb[0]]==0:
            ind_ch01_s0.append(bb[0])
        if stimulus[bb[0]]==1:
            ind_ch01_s1.append(bb[0])

    # Context change from 1 to 0
    ind_ch10_s0=[]
    ind_ch10_s1=[]
    n_ch10=len(ind_ch10)
    for i in range(n_ch10):
        aa=(np.arange(n_forw)+ind_ch10[i])
        bb_pre=aa*reward[aa]
        bb=bb_pre[bb_pre>0]
        if stimulus[bb[0]]==0:
            ind_ch10_s0.append(bb[0])
        if stimulus[bb[0]]==1:
            ind_ch10_s1.append(bb[0])
    return np.array(ind_ch01_s0,dtype=np.int16),np.array(ind_ch01_s1,dtype=np.int16),np.array(ind_ch10_s0,dtype=np.int16),np.array(ind_ch10_s1,dtype=np.int16)

def create_context_subj(context_pre,ctx_ch_pre,ctx_ch):
    context_subj=context_pre.copy()
    for i in range(len(ctx_ch)):
        diff=(ctx_ch[i]-ctx_ch_pre[i])
        context_subj[ctx_ch_pre[i]:(ctx_ch_pre[i]+diff)]=context_pre[ctx_ch_pre[i]-1]
    return context_subj

# Best for behavior
# def func1(x,a,b,c):
#     y=1.0/(1+np.exp(-a*x))
#     return b*y+c

def func1(x,a,b,c,d):
    y=1.0/(1+np.exp(-a*x+d))
    return b*y+c

# def func1(x,a,b,c):
#     y0=(0.5*b+c)
#     y1=1.0/(1+np.exp(-a*x))
#     return np.heaviside(-x,1)*y0+np.heaviside(x,0)*(b*y1+c)

def fit_plot(xx,yy,t_back,t_forw,maxfev,method,bounds,p0):
    #popt,pcov=curve_fit(func1,xx[(t_back+1):],yy[(t_back+1):],nan_policy='omit',maxfev=maxfev,bounds=bounds,p0=p0,method=method)
    #fit_func=func1(xx[(t_back+1):],popt[0],popt[1],popt[2])#,popt[3])
    popt,pcov=curve_fit(func1,xx,yy,nan_policy='omit',maxfev=maxfev,bounds=bounds,p0=p0,method=method)
    fit_func=func1(xx,popt[0],popt[1],popt[2],popt[3])
    print ('Fit ',popt)
    print (pcov)
    #plt.scatter(xx,yy,color='blue',s=5)
    # #plt.plot(xx[(t_back+1):],fit_func,color='black')
    #plt.plot(xx,fit_func,color='black')
    #plt.axvline(0,color='black',linestyle='--')
    #plt.show()
    return fit_func,popt

#################################################

# 
# logistic function with lateral shift with the following parameters work for both (filter out neurons with slope parameter above 5):
# thres = 0, t_back=50, t_forw=100, bounds=([0,0,-5,2],[10,1,5,7]), p0=(0.1,0.5,-0.3,4)

monkeys=['Niels','Galileo']
t_back=50
t_forw=100

talig='dots_on' #'response_edf' #dots_on
thres=0
reg=1e-3
maxfev=100000
method='dogbox'
#bounds=([0,0,-5],[1e5,1,5])
#p0=(0.2,0.5,-0.3)
bounds=([0,0,-5,2],[10,1,5,7])
p0=(0.1,0.5,-0.3,4)

xx=np.arange(t_back+t_forw)-t_back
group_ref=np.array([-7 ,-6 ,-5 ,-4 ,-3 ,-2 ,-1 ,0  ,1  ,2  ,3  ,4  ,5  ,6  ,7  ])

thres_all=nan*np.zeros((len(monkeys),15))
thres_neu_all=nan*np.zeros((len(monkeys),15,96))
fr_ch_all=nan*np.zeros((len(monkeys),15,t_back+t_forw))

for k in range(len(monkeys)):     
    if monkeys[k]=='Niels':
        dic_time=np.array([0,200,200,200])# time pre, time post, bin size, step size (time pre always positive)
        files_groups=[[0,1],[1,2],[2,3],[3,4],[4,5],[5,6],[6,7],[7,8],[8,9],[9,10],[10,11],[11,12]]
    if monkeys[k]=='Galileo':
        dic_time=np.array([0,300,300,300])# time pre, time post, bin size, step size (time pre always positive)
        files_groups=[[0,2],[2,4],[4,6],[6,8],[8,10],[10,12],[12,14],[14,16],[16,18],[18,20],[20,22],[22,24],[24,26],[26,28],[28,30]]

    abs_path='/home/ramon/Dropbox/Proyectos_Postdoc/Esteki_Kiani/data/unsorted/%s/'%(monkeys[k]) 
    files_pre=np.array(os.listdir(abs_path))
    order=order_files(files_pre)
    files_all=np.array(files_pre[order])
    print (files_all)

    thres_vec=nan*np.zeros(len(files_groups))
    thres_neu_vec=nan*np.zeros((len(files_groups),96))
    fr_ch_vec=nan*np.zeros((len(files_groups),t_back+t_forw))
    
    for hh in range(len(files_groups)):
        files=files_all[files_groups[hh][0]:files_groups[hh][1]]
        diff_fr_gr=nan*np.zeros((len(files),t_back+t_forw))
        diff_fr_gr_neu=nan*np.zeros((len(files),96,t_back+t_forw))
        for kk in range(len(files)):
            print (files[kk])
            #Load data
            data=scipy.io.loadmat(abs_path+'%s'%(files[kk]),struct_as_record=False,simplify_cells=True)
            beha=miscellaneous.behavior(data,group_ref)
            index_nonan=beha['index_nonan']
            # We discard first trial of session because we are interested in context changes
            stimulus=beha['stimulus'][1:]
            choice=beha['choice'][1:]
            coherence=beha['coherence_signed'][1:]
            coh_uq=np.unique(coherence)
            reward=beha['reward'][1:]
            rt=beha['reaction_time'][1:]
            context_prepre=beha['context']
            ctx_ch=(context_prepre[1:]-context_prepre[0:-1])
            context_pre=context_prepre[1:] 
            ind_ch_pre=np.where(abs(ctx_ch)==1)[0] # ind_ch_pre index where there is a context change
            #ind_ch=np.where(abs(ctx_ch)==1)[0] # ind_ch_pre index where there is a context change
            indch_ct01_pre=np.where(ctx_ch==1)[0]
            indch_ct10_pre=np.where(ctx_ch==-1)[0]
            ind_ch=calculate_ind_ch_corr(ind_ch_pre,reward) # ind_ch first correct trial after context change (otherwise animal doesn't know there was a change)
            context=create_context_subj(context_pre,ind_ch_pre,ind_ch) # Careful! this is subjective context
            ind_ch01_s0,ind_ch01_s1,ind_ch10_s0,ind_ch10_s1=calculate_ind_ch_corr2(indch_ct01_pre,indch_ct10_pre,reward,stimulus)
            ind_ch01=np.concatenate((ind_ch01_s0,ind_ch01_s1))
            ind_ch10=np.concatenate((ind_ch10_s0,ind_ch10_s1))
            ind_ch_vec=[ind_ch01,ind_ch10]
            
            firing_rate_pre=miscellaneous.getRasters_unsorted(data,talig,dic_time,index_nonan,threshold=thres)
            firing_rate=miscellaneous.normalize_fr(firing_rate_pre)[1:,:,0]
            
            # Get the sign of firing rate change after context change for each neuron
            diff_ctx=nan*np.zeros((2,96))
            for i in range(2):
                ind_ch_u=ind_ch_vec[i]
                diff_ctx_pre=np.zeros((len(ind_ch_u),96))
                for j in range(len(ind_ch_u)):
                    for ii in range(96): 
                        fi_pre=np.mean(firing_rate[(ind_ch_u[j]-t_back):(ind_ch_u[j]-1),ii])
                        fi_post=np.mean(firing_rate[(ind_ch_u[j]):(ind_ch_u[j]+t_forw),ii])
                        diff_ctx_pre[j,ii]=(fi_post-fi_pre)
                diff_ctx[i]=np.nanmean(diff_ctx_pre,axis=0)
            # firing rate change for 0 to 1 and sign
            sign_01=np.ones(96)
            sign_01[diff_ctx[0]<0]=-1
            # firing rate change for 1 to 0 and sign
            sign_10=np.ones(96)
            sign_10[diff_ctx[1]<0]=-1

            #########################################
            # Mean across context changes and neurons. 
            # diff_ctx=nan*np.zeros((2,t_back+t_forw))
            # for i in range(2):
            #     ind_ch_u=ind_ch_vec[i]
            #     if i==0:
            #         s_mult=sign_01
            #     if i==1:
            #         s_mult=sign_10
                        
            #     diff_ctx_pre=nan*np.zeros((len(ind_ch_u),t_back+t_forw))
            #     for j in range(len(ind_ch_u)):
            #         for ii in range(t_back+t_forw):
            #             try:
            #                 outlier=2
            #                 act_neu=s_mult*firing_rate[ind_ch_u[j]-t_back+ii]
            #                 diff_ctx_pre[j,ii]=np.nanmean(act_neu[abs(act_neu)<outlier])
            #             except:
            #                 None

            #     diff_ctx[i]=np.nanmean(diff_ctx_pre,axis=0)
            # diff_fr_gr[kk]=np.nanmean(diff_ctx,axis=0)
            
            #########################################
            # Mean across context changes for each individual neuron
            diff_ctx_neu=nan*np.zeros((2,96,t_back+t_forw))
            for i in range(2):
                ind_ch_u=ind_ch_vec[i]
                if i==0:
                    s_mult=sign_01
                if i==1:
                    s_mult=sign_10
                        
                diff_ctx_neu_pre=nan*np.zeros((len(ind_ch_u),96,t_back+t_forw))
                for j in range(len(ind_ch_u)):
                    for ii in range(t_back+t_forw):
                        try:
                            diff_ctx_neu_pre[j,:,ii]=s_mult*firing_rate[ind_ch_u[j]-t_back+ii]#act_neu[abs(act_neu)<outlier]
                        except:
                            None

                diff_ctx_neu[i]=np.nanmean(diff_ctx_neu_pre,axis=0)
            diff_fr_gr_neu[kk]=np.nanmean(diff_ctx_neu,axis=0)

        diff_fr_gr_neu_m=np.nanmean(diff_fr_gr_neu,axis=0)
        for oo in range(96):
            print (oo)
            try:
                popt=fit_plot(xx,diff_fr_gr_neu_m[oo],t_back,t_forw,maxfev,method,bounds,p0)[1]
            except:
                print ('error fit')
            thres_neu_vec[hh,oo]=popt[0]
            thres_neu_all[k,hh,oo]=popt[0]
            
#         diff_fr_gr_m=np.nanmean(diff_fr_gr,axis=0)
#         fr_ch_vec[hh]=diff_fr_gr_m
#         fr_ch_all[k,hh]=diff_fr_gr_m

#         thres_vec[hh]=popt[0]
#         thres_all[k,hh]=popt[0]

#     # Plot all thresholds throughout learning
#     fig=plt.figure(figsize=(2.3,2))
#     ax=fig.add_subplot(111)
#     miscellaneous.adjust_spines(ax,['left','bottom'])
#     ax.plot(np.arange(len(files_groups)),thres_vec,color='green')
#     #ax.fill_between(np.arange(len(stage_vec)),beha_def_m-beha_def_sem,beha_def_m+beha_def_sem,color='green',alpha=0.5)
#     #ax.plot(np.arange(len(stage_vec)),np.zeros(len(stage_vec)),color='black',linestyle='--')
#     ax.set_ylabel('Slope Fit $\Delta$FR after change')
#     ax.set_xlabel('Sessions')
#     #ax.set_ylim([-150,250])
#     fig.savefig('/home/ramon/Dropbox/Proyectos_Postdoc/Esteki_Kiani/plots/slope_fr_change_all_sessions_learning_%s.pdf'%monkeys[k],dpi=500,bbox_inches='tight')

#     # Slope of fit throughout learning (3 points)
#     slope_epoch=np.zeros((3,2))
#     if monkeys[k]=='Niels':
#         slope_epoch[0,0]=np.mean(thres_vec[0:4])
#         slope_epoch[1,0]=np.mean(thres_vec[4:8])
#         slope_epoch[2,0]=np.mean(thres_vec[8:12])
#         slope_epoch[0,1]=sem(thres_vec[0:4])
#         slope_epoch[1,1]=sem(thres_vec[4:8])
#         slope_epoch[2,1]=sem(thres_vec[8:12])
#     if monkeys[k]=='Galileo':
#         slope_epoch[0,0]=np.mean(thres_vec[0:5])
#         slope_epoch[1,0]=np.mean(thres_vec[5:10])
#         slope_epoch[2,0]=np.mean(thres_vec[10:15])
#         slope_epoch[0,1]=sem(thres_vec[0:5])
#         slope_epoch[1,1]=sem(thres_vec[5:10])
#         slope_epoch[2,1]=sem(thres_vec[10:15])
#     fig=plt.figure(figsize=(2.3,2))
#     ax=fig.add_subplot(111)
#     miscellaneous.adjust_spines(ax,['left','bottom'])
#     ax.plot(np.arange(3),slope_epoch[:,0],color='green')
#     ax.fill_between(np.arange(3),slope_epoch[:,0]-slope_epoch[:,1],slope_epoch[:,0]+slope_epoch[:,1],color='green',alpha=0.5)
#     ax.set_ylabel('Slope Fit $\Delta$FR after change')
#     ax.set_xlabel('Sessions')
#     fig.savefig('/home/ramon/Dropbox/Proyectos_Postdoc/Esteki_Kiani/plots/slope_fr_change_three_epochs_%s.pdf'%monkeys[k],dpi=500,bbox_inches='tight')

#     # Mean FR for each epoch and fit the curve (3 plots)
#     epoch=['early','mid','late']
#     fr_epoch=np.zeros((3,t_back+t_forw))
#     if monkeys[k]=='Niels':
#         fr_epoch[0]=np.mean(fr_ch_vec[0:4],axis=0)
#         fr_epoch[1]=np.mean(fr_ch_vec[4:8],axis=0)
#         fr_epoch[2]=np.mean(fr_ch_vec[8:12],axis=0)
#     if monkeys[k]=='Galileo':
#         fr_epoch[0]=np.mean(fr_ch_vec[0:5],axis=0)
#         fr_epoch[1]=np.mean(fr_ch_vec[5:10],axis=0)
#         fr_epoch[2]=np.mean(fr_ch_vec[10:15],axis=0)

#     for i in range(3):
#         ff,popt=fit_plot(xx,fr_epoch[i],t_back,t_forw,maxfev,method,bounds,p0)
#         fig=plt.figure(figsize=(2.3,2))
#         ax=fig.add_subplot(111)
#         miscellaneous.adjust_spines(ax,['left','bottom'])
#         ax.scatter(xx,fr_epoch[i],color='black',s=5)
#         ax.plot(xx,ff,color='green')
#         #ax.plot(xx[(t_back+1):],ff,color='green')
#         ax.axvline(0,color='black',linestyle='--')
#         ax.set_ylabel('Mean firing rate (z-score)')
#         ax.set_xlabel('Sessions')
#         fig.savefig('/home/ramon/Dropbox/Proyectos_Postdoc/Esteki_Kiani/plots/fr_ctx_change_%s_%s.pdf'%(monkeys[k],epoch[i]),dpi=500,bbox_inches='tight')

# # Both
# # Slope of fit throughout learning (3 points)
# slope_epoch=np.zeros((3,2))
# slope_epoch[0,0]=np.nanmean(np.concatenate((thres_all[0,0:4],thres_all[1,0:5])))
# slope_epoch[1,0]=np.nanmean(np.concatenate((thres_all[0,4:8],thres_all[1,5:10])))
# slope_epoch[2,0]=np.nanmean(np.concatenate((thres_all[0,8:12],thres_all[1,10:15])))
# fig=plt.figure(figsize=(2.3,2))
# ax=fig.add_subplot(111)
# miscellaneous.adjust_spines(ax,['left','bottom'])
# ax.plot(np.arange(3),slope_epoch[:,0],color='green')
# ax.fill_between(np.arange(3),slope_epoch[:,0]-slope_epoch[:,1],slope_epoch[:,0]+slope_epoch[:,1],color='green',alpha=0.5)
# ax.set_ylabel('Slope Fit $\Delta$FR after change')
# ax.set_xlabel('Sessions')
# fig.savefig('/home/ramon/Dropbox/Proyectos_Postdoc/Esteki_Kiani/plots/slope_fr_change_three_epochs_both.pdf',dpi=500,bbox_inches='tight')

# # Mean FR for each epoch and fit the curve (3 plots)
# fr_epoch=np.zeros((3,t_back+t_forw))
# fr_epoch[0]=np.nanmean(np.concatenate((fr_ch_all[0,0:4],fr_ch_all[1,0:5]),axis=0),axis=0)
# fr_epoch[1]=np.nanmean(np.concatenate((fr_ch_all[0,4:8],fr_ch_all[1,5:10]),axis=0),axis=0)
# fr_epoch[2]=np.nanmean(np.concatenate((fr_ch_all[0,8:12],fr_ch_all[1,10:15]),axis=0),axis=0)
# epoch=['early','mid','late']
# for i in range(3):
#     ff,popt=fit_plot(xx,fr_epoch[i],t_back,t_forw,maxfev,method,bounds,p0)
#     fig=plt.figure(figsize=(2.3,2))
#     ax=fig.add_subplot(111)
#     miscellaneous.adjust_spines(ax,['left','bottom'])
#     ax.scatter(xx,fr_epoch[i],color='black',s=5)
#     ax.plot(xx,ff,color='green')
#     #ax.plot(xx[(t_back+1):],ff,color='green')
#     ax.axvline(0,color='black',linestyle='--')
#     ax.set_ylabel('Mean firing rate (z-score)')
#     ax.set_xlabel('Sessions')
#     fig.savefig('/home/ramon/Dropbox/Proyectos_Postdoc/Esteki_Kiani/plots/fr_ctx_change_both_%s.pdf'%epoch[i],dpi=500,bbox_inches='tight')
