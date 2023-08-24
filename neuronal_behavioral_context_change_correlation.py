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

def gauss(x,mu,sig):
    return 1/(sig*np.sqrt(2*np.pi))*np.exp(-0.5*((x-mu)**2)/(sig**2))

def func(x,a,b):
    return 1.0/(1+np.exp(-a*x+b))

def func0(x,a,b):
    y=1.0/(1+np.exp(-a*x))
    return b*y

# Best for behavior
def func1(x,a,b,c):
    y=1.0/(1+np.exp(-a*x))
    return b*y+c

def func2(x,a,b,c):
    y=1.0/(1+np.exp(-a*x+c))
    return b*y

def func3(x,a,b,c,d):
    y=1.0/(1+np.exp(-a*x+d))
    return b*y+c

def func4(x,a,b):
    return a*x+b

def intercept(a,b):
    return b/a

def intercept0(a,b):
    return np.log(b/0.5-1)/(-a)

def intercept1(a,b,c):
    return np.log(b/(0.5-c)-1)/(-a)

def intercept2(a,b,c):
    return (np.log(b/0.5-1)-c)/(-a)

def intercept3(a,b,c,d):
    return (np.log(b/(0.5-c)-1)-d)/(-a)

def intercept4(a,b):
    return (0.5-b)/a

def fit_plot(xx,yy,t_back,t_forw,sig_kernel,maxfev,method,bounds,p0):
    kernel=gauss(xx,int((t_back+t_forw)/2.0)-t_back,sig_kernel)
    convo=np.convolve(yy,kernel,mode='same')
    
    popt,pcov=curve_fit(func2,xx[t_back:],yy[t_back:],nan_policy='omit',maxfev=maxfev,bounds=bounds,p0=p0,method=method)
    #popt,pcov=curve_fit(func1,xx[t_back:],convo[t_back:],nan_policy='omit',maxfev=maxfev,bounds=bounds,p0=p0,method=method)
    fit_func=func2(xx[t_back:],popt[0],popt[1],popt[2])#,popt[3])
    inter=intercept2(popt[0],popt[1],popt[2])#,popt[3])
    print ('Fit ',popt)
    print (pcov)
    #print (inter)
    # plt.scatter(xx,yy,color='blue',s=1)
    # plt.scatter(xx,convo,color='green',s=1)
    # plt.plot(xx[t_back:],fit_func,color='black')
    # plt.axvline(0,color='black',linestyle='--')
    # plt.plot(xx,0.5*np.ones(len(xx)),color='black',linestyle='--')
    # plt.ylim([-0.1,1.1])
    # plt.show()
    return fit_func,inter
  
#################################################

# Function 2 for both. Bounds and p0 are important. 
# Niels: t_back 20, t_forw 80, time window 200ms. No kernel. Groups of 1 session
# Galileo: t_back 20, t_forw 80, time window 400ms. No kernel. Groups of 3 sessions

monkey='Galileo'
t_back=20
t_forw=80
sig_kernel=1 # not smaller than 1

talig='dots_on' #'response_edf' #dots_on
dic_time=np.array([0,400,400,400])# time pre, time post, bin size, step size (time pre always positive) #For Galileo use timepost 800 or 1000. For Niels use 

thres=0
reg=1e0
maxfev=100000
method='dogbox'
#bounds=([0,0,-1,-100],[10,1,1,100])
#p0=(0.05,0.5,0.2,1)
bounds=([0,0,0],[1,1,10])
p0=(0.05,0.5,1)

xx=np.arange(t_back+t_forw)-t_back

group_ref=np.array([-7 ,-6 ,-5 ,-4 ,-3 ,-2 ,-1 ,0  ,1  ,2  ,3  ,4  ,5  ,6  ,7  ])
if monkey=='Niels':
    #files_groups=[[0,4],[4,8],[8,12]]
    #files_groups=[[0,3],[3,6],[6,9],[9,12]]
    #files_groups=[[0,2],[2,4],[4,6],[6,8],[8,10],[10,12]]
    files_groups=[[0,1],[1,2],[2,3],[3,4],[4,5],[5,6],[6,7],[7,8],[8,9],[9,10],[10,11],[11,12]]

if monkey=='Galileo':
    #files_groups=[[0,10],[10,20],[20,30]]
    #files_groups=[[0,5],[5,10],[10,15],[15,20],[20,25],[25,30]]
    files_groups=[[0,3],[3,6],[6,9],[9,12],[12,15],[15,18],[18,21],[21,24],[24,27],[27,30]]
    #files_groups=[[0,2],[2,4],[4,6],[6,8],[8,10],[10,12],[12,14],[14,16],[16,18],[18,20],[20,22],[22,24],[24,26],[26,28],[28,30]]
    #files_groups=[[0,1],[1,2],[2,3],[3,4],[4,5],[5,6],[6,7],[7,8],[8,9],[9,10],[10,11],[11,12],[12,13],[13,14],[14,15],[15,16],[16,17],[17,18],[18,19],[19,20],[20,21],[21,22],[22,23],[23,24],[24,25],[25,26],[26,27],[27,28],[28,29],[29,30]]

abs_path='/home/ramon/Dropbox/Esteki_Kiani/data/unsorted/%s/'%(monkey) 
files_pre=np.array(os.listdir(abs_path))
order=order_files(files_pre)
files_all=np.array(files_pre[order])
print (files_all)

beha_ctx_ch=nan*np.zeros((len(files_groups),t_back+t_forw))
fit_beha=nan*np.zeros((len(files_groups),t_back+t_forw))
inter_beha=nan*np.zeros((len(files_groups)))
neu_ctx_ch=nan*np.zeros((len(files_groups),t_back+t_forw))
fit_neu=nan*np.zeros((len(files_groups),t_back+t_forw))
inter_neu=nan*np.zeros((len(files_groups)))

for hh in range(len(files_groups)):
    xx_forw_pre=nan*np.zeros((100,(t_back+t_forw)))
    beha_pre=nan*np.zeros((100,(t_back+t_forw)))
    neu_ctx_pre=nan*np.zeros((100,(t_back+t_forw)))
    gg=-1
    oo=-1
    files=files_all[files_groups[hh][0]:files_groups[hh][1]]
    print (files)
    for kk in range(len(files)):
        print (files[kk])
        #Load data
        data=scipy.io.loadmat(abs_path+'%s'%(files[kk]),struct_as_record=False,simplify_cells=True)
        beha=miscellaneous.behavior(data)
        index_nonan=beha['index_nonan']
        # We discard first trial of session because we are interested in context changes
        stimulus=beha['stimulus'][1:]
        choice=beha['choice'][1:]
        coherence=beha['coherence_signed'][1:]
        coh_uq=np.unique(coherence)
        reward=beha['reward'][1:]
        rt=beha['reaction_time'][1:]
        context_pre=beha['context']
        ctx_ch=(context_pre[1:]-context_pre[0:-1])
        context=context_pre[1:]
        ind_ch=np.where(abs(ctx_ch)==1)[0]
        indch_ct10=np.where(ctx_ch==-1)[0]
        indch_ct01=np.where(ctx_ch==1)[0]
        #print (ind_ch,len(choice))

        firing_rate_pre=miscellaneous.getRasters_unsorted(data,talig,dic_time,index_nonan,threshold=thres)
        firing_rate=miscellaneous.normalize_fr(firing_rate_pre)[1:,:,0]
        
        ##################################################
        # Behavior
        # Probability of Choice

        for h in range(len(ind_ch)):
            gg+=1
            #print (gg)
            for j in range(t_back):
                try:
                    beha_pre[gg,j]=(choice[ind_ch[h]-t_back+j]==context[ind_ch[h]-t_back+j])
                except:
                    None
                    #print ('Error Behavior Back ',h,j)
            for j in range(t_forw):
                try:
                    beha_pre[gg,t_back+j]=(choice[ind_ch[h]+j]==context[ind_ch[h]+j])
                except:
                    None
                    #print ('Error Behavior Forward ',h,j)
            xx_forw_pre[gg]=(np.arange(t_forw+t_back)-t_back)

        ####################################################3
        # Neuronal
        # Extract indices for training classifier (remove the one for testing from the entire dataset)
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

        print (len(context),len(ind_train),np.mean(context[ind_train]))
        
        # Fit classifier
        cl=LogisticRegression(C=1/reg,class_weight='balanced')
        cl.fit(firing_rate[ind_train],context[ind_train])
        #print ('norm ',np.linalg.norm(cl.coef_[0]))

        for o in range(len(ind_ch)):
            oo+=1
            for j in range(t_back):
                try:
                    neu_ctx_pre[oo,j]=(cl.predict(firing_rate[(ind_ch[o]-t_back+j):(ind_ch[o]-t_back+j+1)])==context[ind_ch[o]-t_back+j])
                    #neu_ctx_pre[oo,j]=cl.score(firing_rate[(ind_ch[o]-t_back+j)].reshape(1,-1),context[ind_ch[o]-t_back+j].reshape(1,-1))
                except:
                    None
                    #print ('Error Neuro Back ',o,j)
            for j in range(t_forw):
                try:
                    neu_ctx_pre[oo,t_back+j]=(cl.predict(firing_rate[(ind_ch[o]+j):(ind_ch[o]+j+1)])==context[ind_ch[o]+j])
                    #neu_ctx_pre[oo,t_back+j]=cl.score(firing_rate[(ind_ch[o]+j)].reshape(1,-1),context[ind_ch[o]+j].reshape(1,-1))
                except:
                    None
                    #print ('Error Neuro Forward ',o,j) 
                    
    beha_ctx_ch[hh]=np.nanmean(beha_pre,axis=0)
    neu_ctx_ch[hh]=np.nanmean(neu_ctx_pre,axis=0)

    popt,pcov=curve_fit(func2,xx_forw_pre[:,t_back:].ravel(),beha_pre[:,t_back:].ravel(),nan_policy='omit',maxfev=maxfev,p0=p0,method=method,bounds=bounds)
    fit_func=func2(xx[t_back:],popt[0],popt[1],popt[2])
    print ('Beha2 ',intercept2(popt[0],popt[1],popt[2]))
           
    popt,pcov=curve_fit(func2,xx_forw_pre[:,t_back:].ravel(),neu_ctx_pre[:,t_back:].ravel(),nan_policy='omit',maxfev=maxfev,p0=p0,method=method,bounds=bounds)
    fit_func=func2(xx[t_back:],popt[0],popt[1],popt[2])
    print ('Neu2 ',intercept2(popt[0],popt[1],popt[2]))
    
    aa=fit_plot(xx,beha_ctx_ch[hh],t_back,t_forw,sig_kernel,maxfev,method=method,p0=p0,bounds=bounds)
    fit_beha[hh,t_back:]=aa[0]
    fit_beha[hh,0:t_back]=np.mean(beha_ctx_ch[hh,0:t_back])
    inter_beha[hh]=aa[1]

    aa=fit_plot(xx,neu_ctx_ch[hh],t_back,t_forw,sig_kernel,maxfev,method=method,p0=p0,bounds=bounds)
    fit_neu[hh,t_back:]=aa[0]
    fit_neu[hh,0:t_back]=np.mean(neu_ctx_ch[hh,0:t_back])
    inter_neu[hh]=aa[1]

    print ('Beha ',inter_beha[hh])
    print ('Neu ',inter_neu[hh])

##################################
beha_ch_m=np.mean(beha_ctx_ch,axis=0)
beha_ch_sem=sem(beha_ctx_ch,axis=0)
neu_ch_m=np.mean(neu_ctx_ch,axis=0)
neu_ch_sem=sem(neu_ctx_ch,axis=0)

fit_beha_m=np.nanmean(fit_beha,axis=0)
fit_beha_sem=sem(fit_beha,axis=0,nan_policy='omit')
fit_neu_m=np.nanmean(fit_neu,axis=0)
fit_neu_sem=sem(fit_neu,axis=0,nan_policy='omit')

# Plot Behavior
fig=plt.figure(figsize=(2.3,2))
ax=fig.add_subplot(111)
miscellaneous.adjust_spines(ax,['left','bottom'])
ax.axvline(0,color='black',linestyle='--')
ax.plot(xx,0.5*np.ones(len(xx)),color='black',linestyle='--')
ax.scatter(xx,beha_ch_m,color='green',s=1)
ax.plot(xx,fit_beha_m,color='green')
ax.fill_between(xx,fit_beha_m-fit_beha_sem,fit_beha_m+fit_beha_sem,color='green',alpha=0.5)
ax.set_ylim([0,1])
ax.set_xlabel('Trials after context change')
ax.set_ylabel('Prob. (Choice == Context)')
plt.legend(loc='best')
fig.savefig('/home/ramon/Dropbox/Esteki_Kiani/plots/prob_choice_context_beha_%s.pdf'%(monkey),dpi=500,bbox_inches='tight')

##########################

# Plot Neuro
fig=plt.figure(figsize=(2.3,2))
ax=fig.add_subplot(111)
miscellaneous.adjust_spines(ax,['left','bottom'])
ax.axvline(0,color='black',linestyle='--')
ax.plot(xx,0.5*np.ones(len(xx)),color='black',linestyle='--')
ax.scatter(xx,neu_ch_m,color='blue',s=1)
ax.plot(xx,fit_neu_m,color='blue')
ax.fill_between(xx,fit_neu_m-fit_neu_sem,fit_neu_m+fit_neu_sem,color='blue',alpha=0.5)
ax.set_ylim([0,1])
ax.set_xlabel('Trials after context change')
ax.set_ylabel('Prob. (Choice == Context)')
plt.legend(loc='best')
fig.savefig('/home/ramon/Dropbox/Esteki_Kiani/plots/prob_choice_context_neu_%s.pdf'%(monkey),dpi=500,bbox_inches='tight')

# Plot decrease Behavior and Neuronal theresholds
fig=plt.figure(figsize=(2.3,2))
ax=fig.add_subplot(111)
miscellaneous.adjust_spines(ax,['left','bottom'])
ax.plot(np.arange(len(inter_beha[inter_beha>0])),inter_beha[inter_beha>0],color='green',label='Behavioral')
ax.plot(np.arange(len(inter_beha[inter_beha>0])),inter_neu[inter_beha>0],color='blue',label='Neuronal')
ax.set_xlabel('Sessions')
ax.set_ylabel('Threshold')
plt.legend(loc='best')
plt.xticks([0,5,10])
fig.savefig('/home/ramon/Dropbox/Esteki_Kiani/plots/thresholds_vs_learning_%s.pdf'%(monkey),dpi=500,bbox_inches='tight')

# Plot correlation
fig=plt.figure(figsize=(2.3,2))
ax=fig.add_subplot(111)
miscellaneous.adjust_spines(ax,['left','bottom'])
for i in range(len(inter_beha)):
    if (inter_beha[i]>0) and (inter_neu[i]>0):
        ax.scatter(inter_beha[i],inter_neu[i],color='black',alpha=(i+1)/len(inter_beha),s=10)
ax.set_xlabel('Threshold Behavioral')
ax.set_ylabel('Threshold Neuronal')
fig.savefig('/home/ramon/Dropbox/Esteki_Kiani/plots/correlation_prob_choice_context_neu_%s.pdf'%(monkey),dpi=500,bbox_inches='tight')



    
