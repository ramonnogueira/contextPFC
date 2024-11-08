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

# Best for behavior
def func1(x,a,b,c):
    y=1.0/(1+np.exp(-a*x))
    return b*y+c

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

def func_eval(index,t_back,t_forw,stimulus,choice,new_ctx):
    pp=nan*np.zeros((2,t_forw+t_back))
    for j in range(t_back):
        indj=(index-t_back+j)
        try:
            if new_ctx=='right':
                pp[int(stimulus[indj]),j]=choice[indj]
            if new_ctx=='left':
                pp[int(stimulus[indj]),j]=int(1-choice[indj])
        except:
            None
    for j in range(t_forw):
        indj=(index+j)
        try:
            if new_ctx=='right':
                pp[int(stimulus[indj]),j+t_back]=choice[indj]
            if new_ctx=='left':
                pp[int(stimulus[indj]),j+t_back]=int(1-choice[indj])
        except:
            None
    return pp

# Extract indices for training classifier (remove the one for testing from the entire dataset) and fit the classifier
def ret_ind_train(coherence,ind_ch,t_back,t_forw):      
    ind_train=np.arange(len(coherence))
    for p in range(len(ind_ch)):
        ind_t=np.arange(t_back+t_forw)-t_back+ind_ch[p]
        ind_del=[]
        for pp in range(len(ind_t)):
            try:
                ind_del.append(np.where(ind_train==ind_t[pp])[0][0])
            except:
                None
                #print ('error aqui')
        ind_del=np.array(ind_del)
        ind_train=np.delete(ind_train,ind_del)
    return ind_train

def fit_plot(xx,yy,t_back,t_forw,maxfev,method,bounds,p0):
    popt,pcov=curve_fit(func1,xx[(t_back+1):],yy[(t_back+1):],nan_policy='omit',maxfev=maxfev,bounds=bounds,p0=p0,method=method)
    fit_func=func1(xx[(t_back+1):],popt[0],popt[1],popt[2])#,popt[3])
    #print ('Fit ',popt)
    #print (pcov)
    #plt.scatter(xx,yy,color='blue',s=5)
    #plt.plot(xx[(t_back+1):],fit_func,color='black')
    #plt.axvline(0,color='black',linestyle='--')
    #plt.plot(xx,0.5*np.ones(len(xx)),color='black',linestyle='--')
    #plt.ylim([-0.1,1.1])
    #plt.show()
    return fit_func,popt

def create_context_subj(context_pre,ctx_ch_pre,ctx_ch):
    context_subj=context_pre.copy()
    for i in range(len(ctx_ch)):
        diff=(ctx_ch[i]-ctx_ch_pre[i])
        context_subj[ctx_ch_pre[i]:(ctx_ch_pre[i]+diff)]=context_pre[ctx_ch_pre[i]-1]
    return context_subj

#################################################

# Function 2 for both. Bounds and p0 are important. 
# Niels: t_back 20, t_forw 80, time window 200ms. No kernel. Groups of 1 session
# Galileo: t_back 20, t_forw 80, time window 300ms. No kernel. Groups of 3 sessions

monkeys=['Galileo']
stage_vec=['late']
t_back=50
t_forw=50
delta_type='fit'

talig='dots_on' #'response_edf' #dots_on
thres=0
reg=1e-3
maxfev=100000
method='dogbox'
#bounds=([0,0,-0.5],[10,1,0.5])
bounds=([-np.inf,-np.inf,-np.inf],[np.inf,np.inf,np.inf])
p0_low=(0.05,0.5,-0.3)
p0_high=(0.05,0.5,0.3)
p0=(0.05,0.5,0.01)

xx=np.arange(t_back+t_forw)-t_back

group_ref=np.array([-7 ,-6 ,-5 ,-4 ,-3 ,-2 ,-1 ,0  ,1  ,2  ,3  ,4  ,5  ,6  ,7  ])

# Careful with length of files groups
delta_beha_all=nan*np.zeros((len(stage_vec),2,2,5,len(monkeys)))
delta_neuro_all=nan*np.zeros((len(stage_vec),2,2,5,len(monkeys)))

for k in range(len(monkeys)):
    for ss in range(len(stage_vec)):
        stage=stage_vec[ss]
        
        if monkeys[k]=='Niels':
            dic_time=np.array([0,200,200,200])# time pre, time post, bin size, step size (time pre always positive)
            if stage=='early':
                files_groups=[[0,1],[1,2],[2,3],[3,4]]
            if stage=='mid':
                files_groups=[[4,5],[5,6],[6,7],[7,8]]
            if stage=='late':
                files_groups=[[8,9],[9,10],[10,11],[11,12]]
        if monkeys[k]=='Galileo':
            dic_time=np.array([0,300,300,300])# time pre, time post, bin size, step size (time pre always positive)
            if stage=='early':
                files_groups=[[0,2],[2,4],[4,6],[6,8],[8,10]]
            if stage=='mid':
                files_groups=[[10,12],[12,14],[14,16],[16,18],[18,20]]
            if stage=='late':
                files_groups=[[20,22],[22,24],[24,26],[26,28],[28,30]]
        
        abs_path='/home/ramon/Dropbox/Proyectos_Postdoc/Esteki_Kiani/data/unsorted/%s/'%(monkeys[k]) 
        files_pre=np.array(os.listdir(abs_path))
        order=order_files(files_pre)
        files_all=np.array(files_pre[order])
        print (files_all)
            
        for hh in range(len(files_groups)):
            files=files_all[files_groups[hh][0]:files_groups[hh][1]]
            diff_fr_gr=nan*np.zeros((len(files),t_back+t_forw))
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
                print (ind_ch01_s0,ind_ch01_s1,ind_ch10_s0,ind_ch10_s1)
                ind_ch_vec=[ind_ch01_s0,ind_ch01_s1,ind_ch10_s0,ind_ch10_s1]
                
                firing_rate_pre=miscellaneous.getRasters_unsorted(data,talig,dic_time,index_nonan,threshold=thres)
                firing_rate=miscellaneous.normalize_fr(firing_rate_pre)[1:,:,0]

                # Get the sign of firing rate change after context change for each neuron
                diff_ctx=nan*np.zeros((4,96))
                for i in range(4):
                    ind_ch_u=ind_ch_vec[i]
                    diff_ctx_pre=np.zeros((len(ind_ch_u),96))
                    for j in range(len(ind_ch_u)):
                        for ii in range(96): 
                            fi_pre=np.mean(firing_rate[(ind_ch_u[j]-t_back):(ind_ch_u[j]-1),ii])
                            fi_post=np.mean(firing_rate[(ind_ch_u[j]):(ind_ch_u[j]+t_forw),ii])
                            diff_ctx_pre[j,ii]=(fi_post-fi_pre)
                    diff_ctx[i]=np.nanmean(diff_ctx_pre,axis=0)
                # firing rate change for 0 to 1 and sign
                diff_01=np.nanmean(diff_ctx[0:2],axis=0)
                sign_01=np.ones(96)
                sign_01[diff_01<0]=-1
                # firing rate change for 1 to 0 and sign
                diff_10=np.nanmean(diff_ctx[2:],axis=0)
                sign_10=np.ones(96)
                sign_10[diff_10<0]=-1

                # Lets get rid of splitting between 01_0 and 01_1!
                # Calculation
                diff_ctx=nan*np.zeros((4,t_back+t_forw))
                for i in range(4):
                    ind_ch_u=ind_ch_vec[i]
                    if i==0 or i==1:
                        s_mult=sign_01
                    if i==2 or i==3:
                        s_mult=sign_10
                        
                    diff_ctx_pre=nan*np.zeros((len(ind_ch_u),t_back+t_forw))
                    for j in range(len(ind_ch_u)):
                        #print (j)
                        for ii in range(t_back+t_forw):
                            try:
                                outlier=2
                                act_neu=s_mult*firing_rate[ind_ch_u[j]-t_back+ii]
                                diff_ctx_pre[j,ii]=np.nanmean(act_neu[abs(act_neu)<outlier])
                            except:
                                None

                    #diff_ctx_pre[diff_ctx_pre]
                    diff_ctx[i]=np.nanmean(diff_ctx_pre,axis=0)
                diff_fr_gr[kk]=np.nanmean(diff_ctx,axis=0)
                
            diff_fr_gr_m=np.nanmean(diff_fr_gr,axis=0)
            popt=fit_plot(xx,diff_fr_gr_m,t_back,t_forw,maxfev,method,bounds,p0)[1]
            print ('Parameters ')
            print (popt)
                
#                 ##################################################
#                 # Neuro
#                 # Probability of Choice of classifier = Context for all possibilities: 01 0, 01 1, 10 0, 10 1
#                 ind_train=ret_ind_train(coherence,ind_ch,t_back,t_forw)
#                 cl=LogisticRegression(C=1/reg,class_weight='balanced')
#                 cl.fit(firing_rate[ind_train],context[ind_train])
#                 choice_cl=cl.predict(firing_rate)

#                 # Numero 1 y 2 top
#                 for h in range(len(ind_ch01_s0)):
#                     cc_01_0=func_eval(ind_ch01_s0[h],t_back,t_forw,stimulus,choice_cl,new_ctx='right')
#                     neuro_tested_rlow.append(cc_01_0[0]) #1
#                     neuro_untested_rhigh.append(cc_01_0[1]) #2
#                 # Numero 3 y 4 top
#                 for h in range(len(ind_ch01_s1)):
#                     cc_01_1=func_eval(ind_ch01_s1[h],t_back,t_forw,stimulus,choice_cl,new_ctx='right')
#                     neuro_untested_rlow.append(cc_01_1[0]) #3
#                     neuro_tested_rhigh.append(cc_01_1[1]) #4
#                 # Numero 3 y 4 bottom           
#                 for h in range(len(ind_ch10_s0)):
#                     cc_10_0=func_eval(ind_ch10_s0[h],t_back,t_forw,stimulus,choice_cl,new_ctx='left')
#                     neuro_untested_rlow.append(cc_10_0[1]) #3
#                     neuro_tested_rhigh.append(cc_10_0[0]) #4
#                 # Numero 1 y 2 bottom           
#                 for h in range(len(ind_ch10_s1)):
#                     cc_10_1=func_eval(ind_ch10_s1[h],t_back,t_forw,stimulus,choice_cl,new_ctx='left')
#                     neuro_tested_rlow.append(cc_10_1[1]) #1
#                     neuro_untested_rhigh.append(cc_10_1[0]) #2
#             # Neuro
#             neuro_te_unte[0,0,hh]=np.nanmean(neuro_tested_rlow,axis=0)
#             neuro_te_unte[0,1,hh]=np.nanmean(neuro_tested_rhigh,axis=0)
#             neuro_te_unte[1,0,hh]=np.nanmean(neuro_untested_rlow,axis=0)
#             neuro_te_unte[1,1,hh]=np.nanmean(neuro_untested_rhigh,axis=0)
#             try:
#                 aa00=fit_plot(xx,neuro_te_unte[0,0,hh],t_back,t_forw,maxfev,method=method,p0=p0,bounds=bounds)
#                 aa01=fit_plot(xx,neuro_te_unte[0,1,hh],t_back,t_forw,maxfev,method=method,p0=p0,bounds=bounds)
#                 aa10=fit_plot(xx,neuro_te_unte[1,0,hh],t_back,t_forw,maxfev,method=method,p0=p0,bounds=bounds)
#                 aa11=fit_plot(xx,neuro_te_unte[1,1,hh],t_back,t_forw,maxfev,method=method,p0=p0,bounds=bounds)
#                 fit_neuro[0,0,hh,(t_back+1):]=aa00
#                 fit_neuro[0,0,hh,0:t_back]=np.nanmean(neuro_te_unte[0,0,hh,0:t_back])
#                 fit_neuro[0,1,hh,(t_back+1):]=aa01
#                 fit_neuro[0,1,hh,0:t_back]=np.nanmean(neuro_te_unte[0,1,hh,0:t_back])
#                 fit_neuro[1,0,hh,(t_back+1):]=aa10
#                 fit_neuro[1,0,hh,0:t_back]=np.nanmean(neuro_te_unte[1,0,hh,0:t_back])
#                 fit_neuro[1,1,hh,(t_back+1):]=aa11
#                 fit_neuro[1,1,hh,0:t_back]=np.nanmean(neuro_te_unte[1,1,hh,0:t_back])
#                 y0_neuro[0,0,hh]=aa00[0]
#                 y0_neuro[0,1,hh]=aa01[0]
#                 y0_neuro[1,0,hh]=aa10[0]
#                 y0_neuro[1,1,hh]=aa11[0]
#             except:
#                 print ('aqui neuro')
        
#         ####################################################

#         if delta_type=='raw':
#             delta_neuro=(neuro_te_unte[:,:,:,t_back+1]-np.nanmean(neuro_te_unte[:,:,:,0:t_back],axis=3))
#         if delta_type=='fit':
#             delta_neuro=(y0_neuro-np.nanmean(neuro_te_unte[:,:,:,0:t_back],axis=3))
#         delta_neuro_all[ss,:,:,:,k]=delta_neuro

        
# #########################################
# # All monkeys    

# delta_neurof=np.reshape(delta_neuro_all,(len(stage_vec),-1))

# delta_neurof_m=np.nanmean(delta_neurof,axis=1)
# delta_neurof_sem=sem(delta_neurof,axis=1,nan_policy='omit')


