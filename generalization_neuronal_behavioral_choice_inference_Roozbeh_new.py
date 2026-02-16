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
nan=float('nan')
minf=float('-inf')
pinf=float('inf')
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn


def log_curve(x,a,c):
    num=1+np.exp(-a*(x+c))
    return 1/num

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

def fit_log_behav_ctx_ch(xx,yy,nback,nfor,reg):
    aa=nan*np.zeros((nforw+1,3))
    cl_bs=LogisticRegression(C=1/reg,class_weight='balanced')
    cl_bs.fit(np.reshape(xx[:,0:nback],(-1,1)),np.reshape(yy[:,0:nback],-1))
    aa[0]=(cl_bs.intercept_[0],cl_bs.coef_[0][0],cl_bs.intercept_[0]/cl_bs.coef_[0][0])
    for i in range(nforw):
        try:
            cl=LogisticRegression(C=1/reg,class_weight='balanced')
            cl.fit(xx[:,nback+i],yy[:,nback+i])
            aa[i+1]=(cl.intercept_[0],cl.coef_[0][0],cl.intercept_[0]/cl.coef_[0][0])
        except:
            None
    return aa


#################################################

monkeys=['Niels','Galileo']
stage='early'
nback=20
nforw=80

talig='dots_on' #'response_edf' #dots_on
thres=0
reg=1e0
maxfev=100000
method='dogbox'

group_ref=np.array([-7 ,-6 ,-5 ,-4 ,-3 ,-2 ,-1 ,0  ,1  ,2  ,3  ,4  ,5  ,6  ,7  ])

xx01s0_all_pre=[]
xx01s1_all_pre=[]
xx10s0_all_pre=[]
xx10s1_all_pre=[]
yy01s0_all_pre=[]
yy01s1_all_pre=[]
yy10s0_all_pre=[]
yy10s1_all_pre=[]
for k in range(len(monkeys)):
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
    print (files_all)

    beha_te_unte=nan*np.zeros((2,2,len(files_groups)))

    xx01s0_pre=[]
    xx01s1_pre=[]
    xx10s0_pre=[]
    xx10s1_pre=[]
    yy01s0_pre=[]
    yy01s1_pre=[]
    yy10s0_pre=[]
    yy10s1_pre=[]
    
    for hh in range(len(files_groups)):
        files=files_all[files_groups[hh][0]:files_groups[hh][1]]
        print (files)
        
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
            ind_ch=miscellaneous.calculate_ind_ch_corr(ind_ch_pre,reward) # ind_ch first correct trial after context change (otherwise animal doesn't know there was a change)
            context=miscellaneous.create_context_subj(context_pre,ind_ch_pre,ind_ch) # Careful! this is subjective context
            ind_ch01_s0,ind_ch01_s1,ind_ch10_s0,ind_ch10_s1=miscellaneous.calculate_ind_ch_corr_stim(indch_ct01_pre,indch_ct10_pre,reward,stimulus)

            firing_rate_pre=miscellaneous.getRasters_unsorted(data,talig,dic_time,index_nonan,threshold=thres)
            firing_rate=miscellaneous.normalize_fr(firing_rate_pre)[1:,:,0]

            print (ind_ch01_s0,ind_ch01_s1,ind_ch10_s0,ind_ch10_s1)
            xx=np.array([100*coherence]).T
        
            ##################################################
            # Behavior
            # Probability of Choice = Context for all possibilities: 01 0, 01 1, 10 0, 10 1
            vec_xx=nan*np.zeros((nback+nforw,1))
            vec_yy=nan*np.zeros(nback+nforw)
            for h in range(len(ind_ch01_s0)):
                vec_xx[0:nback]=xx[(ind_ch01_s0[h]-nback):(ind_ch01_s0[h])]
                vec_yy[0:nback]=choice[(ind_ch01_s0[h]-nback):(ind_ch01_s0[h])]
                for j in range(nforw):
                    try:
                        vec_xx[nback+j]=xx[ind_ch01_s0[h]+j]
                        vec_yy[nback+j]=choice[ind_ch01_s0[h]+j]
                    except:
                        None
                xx01s0_pre.append(vec_xx)
                yy01s0_pre.append(vec_yy)
                xx01s0_all_pre.append(vec_xx)
                yy01s0_all_pre.append(vec_yy)
            vec_xx=nan*np.zeros((nback+nforw,1))
            vec_yy=nan*np.zeros(nback+nforw)
            for h in range(len(ind_ch01_s1)):
                vec_xx[0:nback]=xx[(ind_ch01_s1[h]-nback):(ind_ch01_s1[h])]
                vec_yy[0:nback]=choice[(ind_ch01_s1[h]-nback):(ind_ch01_s1[h])]
                for j in range(nforw):
                    try:
                        vec_xx[nback+j]=xx[ind_ch01_s1[h]+j]
                        vec_yy[nback+j]=choice[ind_ch01_s1[h]+j]
                    except:
                        None
                xx01s1_pre.append(vec_xx)
                yy01s1_pre.append(vec_yy)
                xx01s1_all_pre.append(vec_xx)
                yy01s1_all_pre.append(vec_yy)
            vec_xx=nan*np.zeros((nback+nforw,1))
            vec_yy=nan*np.zeros(nback+nforw)
            for h in range(len(ind_ch10_s0)):
                vec_xx[0:nback]=xx[(ind_ch10_s0[h]-nback):(ind_ch10_s0[h])]
                vec_yy[0:nback]=choice[(ind_ch10_s0[h]-nback):(ind_ch10_s0[h])]
                for j in range(nforw):
                    try:
                        vec_xx[nback+j]=xx[ind_ch10_s0[h]+j]
                        vec_yy[nback+j]=choice[ind_ch10_s0[h]+j]
                    except:
                        None
                xx10s0_pre.append(vec_xx)
                yy10s0_pre.append(vec_yy)
                xx10s0_all_pre.append(vec_xx)
                yy10s0_all_pre.append(vec_yy)
            vec_xx=nan*np.zeros((nback+nforw,1))
            vec_yy=nan*np.zeros(nback+nforw)
            for h in range(len(ind_ch10_s1)):
                vec_xx[0:nback]=xx[(ind_ch10_s1[h]-nback):(ind_ch10_s1[h])]
                vec_yy[0:nback]=choice[(ind_ch10_s1[h]-nback):(ind_ch10_s1[h])]
                for j in range(nforw):
                    try:
                        vec_xx[nback+j]=xx[ind_ch10_s1[h]+j]
                        vec_yy[nback+j]=choice[ind_ch10_s1[h]+j]
                    except:
                        None
                xx10s1_pre.append(vec_xx)
                yy10s1_pre.append(vec_yy)
                xx10s1_all_pre.append(vec_xx)
                yy10s1_all_pre.append(vec_yy)
            
    xx01s0=np.array(xx01s0_pre)
    xx01s1=np.array(xx01s1_pre)
    xx10s0=np.array(xx10s0_pre)
    xx10s1=np.array(xx10s1_pre)
    yy01s0=np.array(yy01s0_pre,dtype=np.int16)
    yy01s1=np.array(yy01s1_pre,dtype=np.int16)
    yy10s0=np.array(yy10s0_pre,dtype=np.int16)
    yy10s1=np.array(yy10s1_pre,dtype=np.int16)

    aa01s0=fit_log_behav_ctx_ch(xx01s0,yy01s0,nback,nforw,reg)
    aa01s1=fit_log_behav_ctx_ch(xx01s1,yy01s1,nback,nforw,reg)
    xx01=np.vstack((xx01s0,xx01s1))
    yy01=np.vstack((yy01s0,yy01s1))
    aa01=fit_log_behav_ctx_ch(xx01,yy01,nback,nforw,reg)

    aa10s0=fit_log_behav_ctx_ch(xx10s0,yy10s0,nback,nforw,reg)
    aa10s1=fit_log_behav_ctx_ch(xx10s1,yy10s1,nback,nforw,reg)
    xx10=np.vstack((xx10s0,xx10s1))
    yy10=np.vstack((yy10s0,yy10s1))
    aa10=fit_log_behav_ctx_ch(xx10,yy10,nback,nforw,reg)

xx01s0_all=np.array(xx01s0_all_pre)
xx01s1_all=np.array(xx01s1_all_pre)
xx10s0_all=np.array(xx10s0_all_pre)
xx10s1_all=np.array(xx10s1_all_pre)
yy01s0_all=np.array(yy01s0_all_pre,dtype=np.int16)
yy01s1_all=np.array(yy01s1_all_pre,dtype=np.int16)
yy10s0_all=np.array(yy10s0_all_pre,dtype=np.int16)
yy10s1_all=np.array(yy10s1_all_pre,dtype=np.int16)

aa01s0_all=fit_log_behav_ctx_ch(xx01s0_all,yy01s0_all,nback,nforw,reg)
aa01s1_all=fit_log_behav_ctx_ch(xx01s1_all,yy01s1_all,nback,nforw,reg)
xx01_all=np.vstack((xx01s0_all,xx01s1_all))
yy01_all=np.vstack((yy01s0_all,yy01s1_all))
aa01_all=fit_log_behav_ctx_ch(xx01_all,yy01_all,nback,nforw,reg)

aa10s0_all=fit_log_behav_ctx_ch(xx10s0_all,yy10s0_all,nback,nforw,reg)
aa10s1_all=fit_log_behav_ctx_ch(xx10s1_all,yy10s1_all,nback,nforw,reg)
xx10_all=np.vstack((xx10s0_all,xx10s1_all))
yy10_all=np.vstack((yy10s0_all,yy10s1_all))
aa10_all=fit_log_behav_ctx_ch(xx10_all,yy10_all,nback,nforw,reg)

# aa_m0=0.5*(aa01_all[:,0]-aa10_all[:,0])
# plt.plot(np.arange(nback+nforw-1),aa_m0)
# #plt.ylim([-10,10])
# plt.()
# plt.show()

aa_m2=0.5*(aa01_all[:,2]-aa10_all[:,2])
plt.plot(np.arange(nforw+1)-1,aa_m2)
plt.plot(np.zeros(nforw+1),color='black',linestyle='--')
plt.axvline(0,color='black',linestyle='--')
plt.xlabel('Trials After Context Switch')
plt.ylabel('Bias (w0/w1)')
plt.ylim([-10,10])
plt.show()

#             ##################################################
#             # Neuro
#             # Probability of Choice of classifier = Context for all possibilities: 01 0, 01 1, 10 0, 10 1
#             ind_train=ret_ind_train(coherence,ind_ch,t_back,t_forw)
#             cl=LogisticRegression(C=1/reg,class_weight='balanced')
#             cl.fit(firing_rate[ind_train],context[ind_train])
#             choice_cl=cl.predict(firing_rate)
