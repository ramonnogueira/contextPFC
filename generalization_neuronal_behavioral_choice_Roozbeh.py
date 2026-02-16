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

# In this script we evaluate generalization through learning.
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

def gauss(x,mu,sig):
    return 1/(sig*np.sqrt(2*np.pi))*np.exp(-0.5*((x-mu)**2)/(sig**2))

def func2(x,a,b,c):
    y=1.0/(1+np.exp(-a*x+c))
    return b*y

def intercept2(a,b,c):
    return (np.log(b/0.5-1)-c)/(-a)

def fit_plot(xx,yy,t_back,t_forw,sig_kernel,maxfev,method,bounds,p0):
    kernel=gauss(xx,int((t_back+t_forw)/2.0)-t_back,sig_kernel)
    convo=np.convolve(yy,kernel,mode='same')
    
    popt,pcov=curve_fit(func2,xx[t_back:],yy[t_back:],nan_policy='omit',maxfev=maxfev,bounds=bounds,p0=p0,method=method)
    #popt,pcov=curve_fit(func2,xx[t_back:],convo[t_back:],nan_policy='omit',maxfev=maxfev,bounds=bounds,p0=p0,method=method)
    fit_func=func2(xx[t_back:],popt[0],popt[1],popt[2])#,popt[3])
    inter=intercept2(popt[0],popt[1],popt[2])#,popt[3])
    print ('Fit ',popt)
    print (pcov)
    print (inter)
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
# Galileo: t_back 20, t_forw 80, time window 300ms. No kernel. Groups of 3 sessions

monkeys=['Niels']#,'Galileo']

talig='dots_on' #'response_edf' #dots_on
thres=0
reg=1e0
maxfev=100000
method='dogbox'
#bounds=([0,0,-1,-100],[10,1,1,100])
#p0=(0.05,0.5,0.2,1)
bounds=([0,0,0],[1,1,10])
p0=(0.05,0.5,1)

t_back=100
t_forw=40
sig_kernel=1 # not smaller than 1
xx=np.arange(t_back+t_forw)-t_back

group_ref=np.array([-7 ,-6 ,-5 ,-4 ,-3 ,-2 ,-1 ,0  ,1  ,2  ,3  ,4  ,5  ,6  ,7  ])

beha_ctx_ch_all=nan*np.zeros((27,t_back+t_forw))
fit_beha_all=nan*np.zeros((27,t_back+t_forw))
inter_beha_all=nan*np.zeros((27))
neu_ctx_ch_all=nan*np.zeros((27,t_back+t_forw))
fit_neu_all=nan*np.zeros((27,t_back+t_forw))
inter_neu_all=nan*np.zeros((27))

uu=-1
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

    beha_ctx_ch=nan*np.zeros((len(files_groups),t_back+t_forw))
    fit_beha=nan*np.zeros((len(files_groups),t_back+t_forw))
    inter_beha=nan*np.zeros((len(files_groups)))
    neu_ctx_ch=nan*np.zeros((len(files_groups),t_back+t_forw))
    fit_neu=nan*np.zeros((len(files_groups),t_back+t_forw))
    inter_neu=nan*np.zeros((len(files_groups)))

    params_vec=[]
    
    for hh in range(len(files_groups)):
        uu+=1
        xx_forw_pre=nan*np.zeros((100,(t_back+t_forw)))
        beha_pre=nan*np.zeros((100,(t_back+t_forw)))
        neu_ctx_pre=nan*np.zeros((100,(t_back+t_forw)))
        gg=-1
        oo=-1
        files=files_all[files_groups[hh][0]:files_groups[hh][1]]
        print (files)

        y_matrix01_pre=[]
        x_matrix01_pre=[]
        y_matrix10_pre=[]
        x_matrix10_pre=[]
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
            ctx_ch_pre=(context_prepre[1:]-context_prepre[0:-1])
            context_pre=context_prepre[1:]
            ind_ch_pre=np.where(abs(ctx_ch_pre)==1)[0] #Careful!
            ind_ch=miscellaneous.calculate_ind_ch_corr(ind_ch_pre,reward) # ind_ch first correct trial after context change (otherwise animal doesn't know there was a change)
            context=miscellaneous.create_context_subj(context_pre,ind_ch_pre,ind_ch) # Careful! this is subjective context. 
            
            ctx_ch=nan*np.zeros(len(reward))
            ctx_ch[1:]=(context[1:]-context[0:-1])
            ctx_ch[0]=0
            indch_ct10=np.where(ctx_ch==-1)[0]
            indch_ct01=np.where(ctx_ch==1)[0]
            
            firing_rate_pre=miscellaneous.getRasters_unsorted(data,talig,dic_time,index_nonan,threshold=thres)
            firing_rate=miscellaneous.normalize_fr(firing_rate_pre)[1:,:,0]
        
            ##################################################
            # Behavior
            # Probability of Choice
           
            for h in range(len(indch_ct01)): #from left to right
                #gg+=1
                for j in range(t_back):
                    try:
                        y_matrix01_pre.append(choice[indch_ct01[h]-t_back+j])
                        x_matrix01_pre.append([coherence[indch_ct01[h]-t_back+j],0])
                    except:
                        None
                for j in range(t_forw):
                    try:
                        y_matrix01_pre.append(choice[indch_ct01[h]+j])
                        x_matrix01_pre.append([coherence[indch_ct01[h]+j],j+1])
                    except:
                        None

            for h in range(len(indch_ct10)): #from left to right
                #gg+=1
                for j in range(t_back):
                    try:
                        y_matrix10_pre.append(choice[indch_ct10[h]-t_back+j])
                        x_matrix10_pre.append([coherence[indch_ct10[h]-t_back+j],0])
                    except:
                        None
                for j in range(t_forw):
                    try:
                        y_matrix10_pre.append(choice[indch_ct10[h]+j])
                        x_matrix10_pre.append([coherence[indch_ct10[h]+j],j+1])
                    except:
                        None


        y_matrix01=np.array(y_matrix01_pre,dtype=np.int16)
        x_matrix01=np.array(x_matrix01_pre)
        b01=LogisticRegression(C=1,class_weight='balanced')
        b01.fit(x_matrix01,y_matrix01)
        print ('From left to right ',len(y_matrix01),b01.intercept_[0],b01.coef_[0])

        y_matrix10=np.array(y_matrix10_pre,dtype=np.int16)
        x_matrix10=np.array(x_matrix10_pre)
        b10=LogisticRegression(C=1,class_weight='balanced')
        b10.fit(x_matrix10,y_matrix10)
        print ('From right to left ',len(y_matrix10),b10.intercept_[0],b10.coef_[0])

        parpre=[[b01.intercept_[0],b01.coef_[0][0],b01.coef_[0][1]],[b10.intercept_[0],b10.coef_[0][0],b10.coef_[0][1]]]
        params_vec.append(np.array(parpre))

    params_vec=np.array(params_vec)

    plt.plot(params_vec[:,0,0],color='blue',label='Left to Right')
    plt.plot(params_vec[:,1,0],color='orange',label='Right to Left')
    plt.xlabel('Sessions')
    plt.legend(loc='best')
    plt.savefig('/home/ramon/Desktop/Figure_20to80_beta0.png',dpi=500,bbox_inches='tight')
    plt.close()
    plt.plot(params_vec[:,1,0]-params_vec[:,0,0],color='black',label='Difference Beta0')
    plt.xlabel('Sessions')
    plt.legend(loc='best')
    plt.savefig('/home/ramon/Desktop/Figure_20to80_beta0_diff.png',dpi=500,bbox_inches='tight')
    plt.close()

    plt.plot(params_vec[:,0,2],color='blue',label='Left to Right')
    plt.plot(params_vec[:,1,2],color='orange',label='Right to Left')
    plt.xlabel('Sessions')
    plt.legend(loc='best')
    plt.savefig('/home/ramon/Desktop/Figure_20to80_beta2.png',dpi=500,bbox_inches='tight')
    plt.close()
    plt.plot(params_vec[:,1,2]-params_vec[:,0,2],color='black',label='Difference Beta0')
    plt.xlabel('Sessions')
    plt.legend(loc='best')
    plt.savefig('/home/ramon/Desktop/Figure_20to80_beta2_diff.png',dpi=500,bbox_inches='tight')
    plt.close()

    plt.plot(params_vec[:,0,2]/params_vec[:,0,0],color='blue',label='Left to Right')
    plt.plot(params_vec[:,1,2]/params_vec[:,1,0],color='orange',label='Right to Left')
    plt.xlabel('Sessions')
    plt.legend(loc='best')
    plt.savefig('/home/ramon/Desktop/Figure_20to80_beta2beta0.png',dpi=500,bbox_inches='tight')
    plt.close()
    plt.plot(params_vec[:,1,2]/params_vec[:,1,0]-params_vec[:,0,2]/params_vec[:,0,0],color='black',label='Difference beta2/beta0')
    plt.xlabel('Sessions')
    plt.legend(loc='best')
    plt.savefig('/home/ramon/Desktop/Figure_20to80_beta2beta0_diff.png',dpi=500,bbox_inches='tight')
    plt.close()

    
