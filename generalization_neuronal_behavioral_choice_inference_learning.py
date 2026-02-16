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

# Best for behavior
def func1(x,a,b,c):
    y=1.0/(1+np.exp(-a*x))
    return b*y+c

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
    # plt.scatter(xx,yy,color='blue',s=5)
    # plt.plot(xx[(t_back+1):],fit_func,color='black')
    # plt.axvline(0,color='black',linestyle='--')
    # plt.plot(xx,0.5*np.ones(len(xx)),color='black',linestyle='--')
    # plt.ylim([-0.1,1.1])
    # plt.show()
    return fit_func

#################################################

# Function 2 for both. Bounds and p0 are important. 
# Niels: t_back 20, t_forw 80, time window 200ms. No kernel. Groups of 1 session
# Galileo: t_back 20, t_forw 80, time window 300ms. No kernel. Groups of 3 sessions

monkeys=['Niels','Galileo']
stage_vec=['early','mid','late']
t_back=20
t_forw=80
delta_type='fit'

talig='dots_on' #'response_edf' #dots_on
thres=0
reg=1e-3
maxfev=1000000000
method='dogbox'
bounds=([0,0,-0.5],[10,0.8,0.5])
#bounds=([-np.inf,-np.inf,-np.inf],[np.inf,np.inf,np.inf])
p0=(0.05,0.5,-0.2)

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
        order=miscellaneous.order_files(files_pre)
        files_all=np.array(files_pre[order])
        #print (files_all)
        
        fit_beha=nan*np.zeros((2,2,5,t_back+t_forw))
        y0_beha=nan*np.zeros((2,2,5))
        beha_te_unte=nan*np.zeros((2,2,5,t_back+t_forw))
        fit_neuro=nan*np.zeros((2,2,5,t_back+t_forw))
        y0_neuro=nan*np.zeros((2,2,5))
        neuro_te_unte=nan*np.zeros((2,2,5,t_back+t_forw))
     
        for hh in range(len(files_groups)):
            beha_tested_rlow=[]
            beha_tested_rhigh=[]
            beha_untested_rlow=[]
            beha_untested_rhigh=[]
            neuro_tested_rlow=[]
            neuro_tested_rhigh=[]
            neuro_untested_rlow=[]
            neuro_untested_rhigh=[]
            files=files_all[files_groups[hh][0]:files_groups[hh][1]]
            #print (files)
        
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
                ind_ch=miscellaneous.calculate_ind_ch_corr(ind_ch_pre,reward) # ind_ch first correct trial after context change
                context=miscellaneous.create_context_subj(context_pre,ind_ch_pre,ind_ch) # Careful! this is subjective context
                ind_ch01_s0,ind_ch01_s1,ind_ch10_s0,ind_ch10_s1=miscellaneous.calculate_ind_ch_corr_stim(indch_ct01_pre,indch_ct10_pre,reward,stimulus)
                
                firing_rate_pre=miscellaneous.getRasters_unsorted(data,talig,dic_time,index_nonan,threshold=thres)
                firing_rate=miscellaneous.normalize_fr(firing_rate_pre)[1:,:,0]
                
                #print (ind_ch01_s0,ind_ch01_s1,ind_ch10_s0,ind_ch10_s1)
                
                ##################################################
                # Behavior
                # Probability of Choice = Context for all possibilities: 01 0, 01 1, 10 0, 10 1

                # Numero 1 y 2 top
                for h in range(len(ind_ch01_s0)):
                    cc_01_0=func_eval(ind_ch01_s0[h],t_back,t_forw,stimulus,choice,new_ctx='right')
                    beha_tested_rlow.append(cc_01_0[0]) #1
                    beha_untested_rhigh.append(cc_01_0[1]) #2
                # Numero 3 y 4 top
                for h in range(len(ind_ch01_s1)):
                    cc_01_1=func_eval(ind_ch01_s1[h],t_back,t_forw,stimulus,choice,new_ctx='right')
                    beha_untested_rlow.append(cc_01_1[0]) #3
                    beha_tested_rhigh.append(cc_01_1[1]) #4
                # Numero 3 y 4 bottom           
                for h in range(len(ind_ch10_s0)):
                    cc_10_0=func_eval(ind_ch10_s0[h],t_back,t_forw,stimulus,choice,new_ctx='left')
                    beha_untested_rlow.append(cc_10_0[1]) #3 
                    beha_tested_rhigh.append(cc_10_0[0]) #4 
                # Numero 1 y 2 bottom           
                for h in range(len(ind_ch10_s1)):
                    cc_10_1=func_eval(ind_ch10_s1[h],t_back,t_forw,stimulus,choice,new_ctx='left')
                    beha_tested_rlow.append(cc_10_1[1]) #1
                    beha_untested_rhigh.append(cc_10_1[0]) #2

                ##################################################
                # Neuro
                # Probability of Choice of classifier = Context for all possibilities: 01 0, 01 1, 10 0, 10 1
                ind_train=ret_ind_train(coherence,ind_ch,t_back,t_forw)
                cl=LogisticRegression(C=1/reg,class_weight='balanced')
                cl.fit(firing_rate[ind_train],context[ind_train])
                choice_cl=cl.predict(firing_rate)

                # Numero 1 y 2 top
                for h in range(len(ind_ch01_s0)):
                    cc_01_0=func_eval(ind_ch01_s0[h],t_back,t_forw,stimulus,choice_cl,new_ctx='right')
                    neuro_tested_rlow.append(cc_01_0[0]) #1
                    neuro_untested_rhigh.append(cc_01_0[1]) #2
                # Numero 3 y 4 top
                for h in range(len(ind_ch01_s1)):
                    cc_01_1=func_eval(ind_ch01_s1[h],t_back,t_forw,stimulus,choice_cl,new_ctx='right')
                    neuro_untested_rlow.append(cc_01_1[0]) #3
                    neuro_tested_rhigh.append(cc_01_1[1]) #4
                # Numero 3 y 4 bottom           
                for h in range(len(ind_ch10_s0)):
                    cc_10_0=func_eval(ind_ch10_s0[h],t_back,t_forw,stimulus,choice_cl,new_ctx='left')
                    neuro_untested_rlow.append(cc_10_0[1]) #3
                    neuro_tested_rhigh.append(cc_10_0[0]) #4
                # Numero 1 y 2 bottom           
                for h in range(len(ind_ch10_s1)):
                    cc_10_1=func_eval(ind_ch10_s1[h],t_back,t_forw,stimulus,choice_cl,new_ctx='left')
                    neuro_tested_rlow.append(cc_10_1[1]) #1
                    neuro_untested_rhigh.append(cc_10_1[0]) #2
                ############################################

            # Behavior
            # print (beha_tested_rlow)
            # print (beha_tested_rhigh)
            # print (beha_untested_rlow)
            # print (beha_untested_rhigh)
            beha_te_unte[0,0,hh]=np.nanmean(beha_tested_rlow,axis=0)
            beha_te_unte[0,1,hh]=np.nanmean(beha_tested_rhigh,axis=0)
            beha_te_unte[1,0,hh]=np.nanmean(beha_untested_rlow,axis=0)
            beha_te_unte[1,1,hh]=np.nanmean(beha_untested_rhigh,axis=0)
            try:
                aa00=fit_plot(xx,beha_te_unte[0,0,hh],t_back,t_forw,maxfev,method=method,p0=p0,bounds=bounds)
                aa01=fit_plot(xx,beha_te_unte[0,1,hh],t_back,t_forw,maxfev,method=method,p0=p0,bounds=bounds)
                aa10=fit_plot(xx,beha_te_unte[1,0,hh],t_back,t_forw,maxfev,method=method,p0=p0,bounds=bounds)
                aa11=fit_plot(xx,beha_te_unte[1,1,hh],t_back,t_forw,maxfev,method=method,p0=p0,bounds=bounds)
                fit_beha[0,0,hh,(t_back+1):]=aa00
                fit_beha[0,0,hh,0:t_back]=np.nanmean(beha_te_unte[0,0,hh,0:t_back])
                fit_beha[0,1,hh,(t_back+1):]=aa01
                fit_beha[0,1,hh,0:t_back]=np.nanmean(beha_te_unte[0,1,hh,0:t_back])
                fit_beha[1,0,hh,(t_back+1):]=aa10
                fit_beha[1,0,hh,0:t_back]=np.nanmean(beha_te_unte[1,0,hh,0:t_back])
                fit_beha[1,1,hh,(t_back+1):]=aa11
                fit_beha[1,1,hh,0:t_back]=np.nanmean(beha_te_unte[1,1,hh,0:t_back])
                y0_beha[0,0,hh]=aa00[0]
                y0_beha[0,1,hh]=aa01[0]
                y0_beha[1,0,hh]=aa10[0]
                y0_beha[1,1,hh]=aa11[0]
            except:
                print ('aqui beha')

            # Neuro
            neuro_te_unte[0,0,hh]=np.nanmean(neuro_tested_rlow,axis=0)
            neuro_te_unte[0,1,hh]=np.nanmean(neuro_tested_rhigh,axis=0)
            neuro_te_unte[1,0,hh]=np.nanmean(neuro_untested_rlow,axis=0)
            neuro_te_unte[1,1,hh]=np.nanmean(neuro_untested_rhigh,axis=0)
            try:
                aa00=fit_plot(xx,neuro_te_unte[0,0,hh],t_back,t_forw,maxfev,method=method,p0=p0,bounds=bounds)
                aa01=fit_plot(xx,neuro_te_unte[0,1,hh],t_back,t_forw,maxfev,method=method,p0=p0,bounds=bounds)
                aa10=fit_plot(xx,neuro_te_unte[1,0,hh],t_back,t_forw,maxfev,method=method,p0=p0,bounds=bounds)
                aa11=fit_plot(xx,neuro_te_unte[1,1,hh],t_back,t_forw,maxfev,method=method,p0=p0,bounds=bounds)
                fit_neuro[0,0,hh,(t_back+1):]=aa00
                fit_neuro[0,0,hh,0:t_back]=np.nanmean(neuro_te_unte[0,0,hh,0:t_back])
                fit_neuro[0,1,hh,(t_back+1):]=aa01
                fit_neuro[0,1,hh,0:t_back]=np.nanmean(neuro_te_unte[0,1,hh,0:t_back])
                fit_neuro[1,0,hh,(t_back+1):]=aa10
                fit_neuro[1,0,hh,0:t_back]=np.nanmean(neuro_te_unte[1,0,hh,0:t_back])
                fit_neuro[1,1,hh,(t_back+1):]=aa11
                fit_neuro[1,1,hh,0:t_back]=np.nanmean(neuro_te_unte[1,1,hh,0:t_back])
                y0_neuro[0,0,hh]=aa00[0]
                y0_neuro[0,1,hh]=aa01[0]
                y0_neuro[1,0,hh]=aa10[0]
                y0_neuro[1,1,hh]=aa11[0]
            except:
                print ('aqui neuro')
        
        ####################################################

        if delta_type=='raw':
            delta_beha=(beha_te_unte[:,:,:,t_back+1]-np.nanmean(beha_te_unte[:,:,:,0:t_back],axis=3))
            delta_neuro=(neuro_te_unte[:,:,:,t_back+1]-np.nanmean(neuro_te_unte[:,:,:,0:t_back],axis=3))
        if delta_type=='fit':
            delta_beha=(y0_beha-np.nanmean(beha_te_unte[:,:,:,0:t_back],axis=3))
            delta_neuro=(y0_neuro-np.nanmean(neuro_te_unte[:,:,:,0:t_back],axis=3))
        delta_beha_all[ss,:,:,:,k]=delta_beha
        delta_neuro_all[ss,:,:,:,k]=delta_neuro

        
#########################################
# All monkeys    
delta_behaf=np.reshape(delta_beha_all,(len(stage_vec),-1))
delta_neurof=np.reshape(delta_neuro_all,(len(stage_vec),-1))

delta_behaf_m=np.nanmean(delta_behaf,axis=1)
delta_behaf_sem=sem(delta_behaf,axis=1,nan_policy='omit')
delta_neurof_m=np.nanmean(delta_neurof,axis=1)
delta_neurof_sem=sem(delta_neurof,axis=1,nan_policy='omit')

print ('##### Behavior ######')
for i in range(len(stage_vec)):
    print ('Stage ',stage_vec[i])
    #print (scipy.stats.wilcoxon(delta_behaf[i],nan_policy='omit'))
    print (scipy.stats.ttest_rel(delta_behaf[i],np.zeros(len(delta_behaf[i])),nan_policy='omit'))

print ('Early vs Late ',scipy.stats.ttest_rel(delta_behaf[0],delta_behaf[2],nan_policy='omit'))

print ('##### Neurons ######')
for i in range(len(stage_vec)):
    print ('Stage ',stage_vec[i])
    #print (scipy.stats.wilcoxon(delta_neurof[i],nan_policy='omit'))
    print (scipy.stats.ttest_rel(delta_neurof[i],np.zeros(len(delta_neurof[i])),nan_policy='omit'))

print ('Early vs Late ',scipy.stats.ttest_rel(delta_neurof[0],delta_neurof[2],nan_policy='omit'))
    
fig=plt.figure(figsize=(2.3,2))
ax=fig.add_subplot(111)
miscellaneous.adjust_spines(ax,['left','bottom'])
ax.plot(np.arange(len(stage_vec)),delta_behaf_m,color='green',label='Behavior')
ax.fill_between(np.arange(len(stage_vec)),delta_behaf_m-delta_behaf_sem,delta_behaf_m+delta_behaf_sem,color='green',alpha=0.5)
ax.plot(np.arange(len(stage_vec)),delta_neurof_m,color='blue',label='Neurons')
ax.fill_between(np.arange(len(stage_vec)),delta_neurof_m-delta_neurof_sem,delta_neurof_m+delta_neurof_sem,color='blue',alpha=0.5)
ax.plot(np.arange(len(stage_vec)),np.zeros(len(stage_vec)),color='black',linestyle='--')
ax.set_ylabel('$\Delta$ Prob. (Choice = New Ctx)')
ax.set_ylim([-0.15,0.15])
fig.savefig('/home/ramon/Dropbox/Proyectos_Postdoc/Esteki_Kiani/plots/inference_choice_both_learning.pdf',dpi=500,bbox_inches='tight')
