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
from scipy.optimize import curve_fit
#from numba import jit
import miscellaneous

nan=float('nan')
minf=float('-inf')
pinf=float('inf')
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

###################################################3

monkeys=['Galileo']#'Niels']#,]

talig_vec=np.array(['dots_on'])#'targ_on','dots_on'
dic_time={} # same number of steps for all time locks
dic_time['dots_on']=np.array([0,800,200,200]) # time pre, time post, bin size, step size
steps_dic={}
xx_dic={}
for i in range(len(talig_vec)):
    steps_dic[talig_vec[i]]=int((dic_time[talig_vec[i]][0]+dic_time[talig_vec[i]][1])/dic_time[talig_vec[i]][3])
    xx_dic[talig_vec[i]]=np.linspace(-dic_time[talig_vec[i]][0]/1000,dic_time[talig_vec[i]][1]/1000,steps_dic[talig_vec[i]],endpoint=False)

nt=50 #100 for coh signed, 200 for coh unsigned, 50 for coh signed with context
n_rand=100
perc_tr=0.8
thres=0
reg=1e-3
n_coh=15

coh_plot=np.array([['-75','-51.2','-25.6','-12.8','-6.4','-3.2','-1.6','0','1.6','3.2','6.4','12.8','25.6','51.2','75'],
                   ['-51.2','-25.6','-12.8','-6.4','-4.5','-3.2','-1.6','0','1.6','3.2','4.5','6.4','12.8','25.6','51.2']])

tpre_sacc=50
#group_ref=np.array([-7 ,-6 ,-5 ,-4 ,-3 ,-2 ,-1 ,0  ,1  ,2  ,3  ,4  ,5  ,6  ,7])
group_coh=np.array([-7 ,-6 ,-5 ,-4 ,-3 ,-2 ,-1 ,0  ,1  ,2  ,3  ,4  ,5  ,6  ,7])

def log_curve(x,a,c):
    num=1+np.exp(-a*x+c)
    return 1/num
    
def log_curve_abs(x,a,c):
    num=1+np.exp(-a*abs(x)+c)
    return 1/num
    
for k in range(len(monkeys)):
    print (monkeys[k])
    abs_path='/home/ramon/Dropbox/Esteki_Kiani/data/sorted/late/%s/'%(monkeys[k])
    #abs_path='/home/ramon/Dropbox/Esteki_Kiani/data/unsorted/%s/'%(monkeys[k])
    files=os.listdir(abs_path)
        
    steps=steps_dic[talig_vec[0]]
    xx=xx_dic[talig_vec[0]]
     
    #perf_pre=nan*np.zeros((steps,n_rand,n_coh,n_coh,3))
    #ccgp_pre=nan*np.zeros((steps,n_rand,n_coh,n_coh,2))
    chrono_neuro_pre=nan*np.zeros((steps,n_rand,n_coh,3))
    psycho_neuro_pre=nan*np.zeros((steps,n_rand,n_coh,3))
    chrono_neuro_flat_pre=nan*np.zeros((n_rand,n_coh,3))
    psycho_neuro_flat_pre=nan*np.zeros((n_rand,n_coh,3))
    for ii in range(n_rand):
        print (ii)
        # Careful! in this function I am only using correct trials so that choice and stimulus are the same
        pseudo=miscellaneous.pseudopop_coherence_context_correct(abs_path,files,talig_vec[0],dic_time[talig_vec[0]],steps_dic[talig_vec[0]],thres,nt,1,perc_tr,True,tpre_sacc,group_coh)
        pseudo_tr_pre=pseudo['pseudo_tr']
        pseudo_te_pre=pseudo['pseudo_te']
        neu_total=pseudo_tr_pre.shape[-1]
        clase_all_pre=pseudo['clase_all']
        clase_coh_pre=pseudo['clase_coh']
        indc0=np.where(clase_coh_pre==7)[0]
        clase_ctx_pre=pseudo['clase_ctx']
        stim_pre=nan*np.zeros(len(clase_coh_pre))
        stim_pre[clase_coh_pre>7]=1
        stim_pre[clase_coh_pre<7]=0
        stim_pre[indc0]=np.array(np.random.normal(0,1,len(indc0))>0,dtype=np.int16)

        ##########################################
        # Psychometric for all time steps together
        pseudo_tr_flat_pre=nan*np.zeros((pseudo_tr_pre.shape[2],steps*neu_total))
        pseudo_te_flat_pre=nan*np.zeros((pseudo_te_pre.shape[2],steps*neu_total))
        for pp in range(steps):
            pseudo_tr_flat_pre[:,pp*neu_total:(pp+1)*neu_total]=pseudo_tr_pre[pp,0]
            pseudo_te_flat_pre[:,pp*neu_total:(pp+1)*neu_total]=pseudo_te_pre[pp,0]

        sum_nan=np.sum(np.isnan(pseudo_tr_flat_pre),axis=1)
        index_nonan=np.where(sum_nan==0)[0]
        pseudo_tr_flat=pseudo_tr_flat_pre[index_nonan]
        pseudo_te_flat=pseudo_te_flat_pre[index_nonan]
        clase_all=clase_all_pre[index_nonan]
        clase_coh=clase_coh_pre[index_nonan]
        clase_ctx=clase_ctx_pre[index_nonan]
        stim=stim_pre[index_nonan]
                
        cl=LogisticRegression(C=1/reg,class_weight='balanced')
        cl.fit(pseudo_tr_flat,stim) # Cuidado con decode Stim y no Choice!
                    
        for j in range(n_coh):
            #print ('  ',j)
            try:
                ind_coh=np.where((clase_coh==j))[0]
                ind_coh0=np.where((clase_coh==j)&(clase_ctx==0))[0] # Left more rewarded
                ind_coh1=np.where((clase_coh==j)&(clase_ctx==1))[0] # Right more rewarded
            
                chrono_neuro_flat_pre[ii,j,0]=cl.score(pseudo_te_flat[ind_coh],stim[ind_coh])
                chrono_neuro_flat_pre[ii,j,1]=cl.score(pseudo_te_flat[ind_coh0],stim[ind_coh0])
                chrono_neuro_flat_pre[ii,j,2]=cl.score(pseudo_te_flat[ind_coh1],stim[ind_coh1])
                psycho_neuro_flat_pre[ii,j,0]=np.mean(cl.predict(pseudo_te_flat[ind_coh]))
                psycho_neuro_flat_pre[ii,j,1]=np.mean(cl.predict(pseudo_te_flat[ind_coh0]))
                psycho_neuro_flat_pre[ii,j,2]=np.mean(cl.predict(pseudo_te_flat[ind_coh1]))
            except:
            #    None
                print ('rand ',ii,'coh ',j)
            
    psycho_neuro_flat_m=np.mean(psycho_neuro_flat_pre,axis=(0))
    psycho_neuro_flat_std=np.std(psycho_neuro_flat_pre,axis=(0))
    chrono_neuro_flat_m=np.mean(chrono_neuro_flat_pre,axis=(0))
    chrono_neuro_flat_std=np.std(chrono_neuro_flat_pre,axis=(0))
    #print (psycho_neuro_m)
    #print (chrono_neuro_m)

    ##################################
    # Figure Psychometric all time steps
    fig=plt.figure(figsize=(2.3,2))
    ax=fig.add_subplot(111)
    miscellaneous.adjust_spines(ax,['left','bottom'])
    xx=np.arange(n_coh)-int(n_coh/2)
    
    # Curve fit
    indnan0=~np.isnan(psycho_neuro_flat_m[:,0])
    popt0,pcov0=curve_fit(log_curve,xx[indnan0],psycho_neuro_flat_m[:,0][indnan0])
    yy0=log_curve(xx[indnan0],popt0[0],popt0[1])
    ax.scatter(xx[indnan0],psycho_neuro_flat_m[:,0][indnan0],color='black',s=3)
    ax.plot(xx[indnan0],yy0,color='black')

    indnan=~np.isnan(psycho_neuro_flat_m[:,1])
    popt1,pcov1=curve_fit(log_curve,xx[indnan],psycho_neuro_flat_m[:,1][indnan])
    yy1=log_curve(xx[indnan],popt1[0],popt1[1])
    ax.scatter(xx[indnan],psycho_neuro_flat_m[:,1][indnan],color='green',s=3)
    ax.plot(xx[indnan],yy1,color='green')

    indnan=~np.isnan(psycho_neuro_flat_m[:,2])
    popt2,pcov2=curve_fit(log_curve,xx[indnan],psycho_neuro_flat_m[:,2][indnan])
    yy2=log_curve(xx[indnan],popt2[0],popt2[1])
    ax.scatter(xx[indnan],psycho_neuro_flat_m[:,2][indnan],color='blue',s=3)
    ax.plot(xx[indnan],yy2,color='blue')       
        
    ax.plot(np.arange(n_coh)-n_coh/2,0.5*np.ones(15),color='black',linestyle='--')
    ax.axvline(0,color='black',linestyle='--')
    ax.set_ylim([0,1])
    ax.set_ylabel('Probability Right Response')
    ax.set_xlabel('Evidence Right Choice (%)')
    #plt.xticks(xx[indnan0],coh_plot[k][indnan0])
    plt.xticks(xx,coh_plot[1])
    fig.savefig('/home/ramon/Dropbox/Esteki_Kiani/figures/figure_neuro_pseudo_psychometric_%s_all.pdf'%(monkeys[k]),dpi=500,bbox_inches='tight')
 
    # ##################################
    # # Figure Psychometric
    # for t_plot in range(steps):
    #     fig=plt.figure(figsize=(2.3,2))
    #     ax=fig.add_subplot(111)
    #     miscellaneous.adjust_spines(ax,['left','bottom'])
    #     xx=np.arange(n_coh)-int(n_coh/2)
    #     # Curve fit
    #     indnan0=~np.isnan(psycho_neuro_m[t_plot,:,0])
    #     popt0,pcov0=curve_fit(log_curve,xx[indnan0],psycho_neuro_m[t_plot,:,0][indnan0])
    #     yy0=log_curve(xx[indnan0],popt0[0],popt0[1])
    #     ax.scatter(xx[indnan0],psycho_neuro_m[t_plot,:,0][indnan0],color='black',s=3)
    #     ax.plot(xx[indnan0],yy0,color='black')

    #     indnan=~np.isnan(psycho_neuro_m[t_plot,:,1])
    #     popt1,pcov1=curve_fit(log_curve,xx[indnan],psycho_neuro_m[t_plot,:,1][indnan])
    #     yy1=log_curve(xx[indnan],popt1[0],popt1[1])
    #     ax.scatter(xx[indnan],psycho_neuro_m[t_plot,:,1][indnan],color='green',s=3)
    #     ax.plot(xx[indnan],yy1,color='green')

    #     indnan=~np.isnan(psycho_neuro_m[t_plot,:,2])
    #     popt2,pcov2=curve_fit(log_curve,xx[indnan],psycho_neuro_m[t_plot,:,2][indnan])
    #     yy2=log_curve(xx[indnan],popt2[0],popt2[1])
    #     ax.scatter(xx[indnan],psycho_neuro_m[t_plot,:,2][indnan],color='blue',s=3)
    #     ax.plot(xx[indnan],yy2,color='blue')       
        
    #     ax.plot(np.arange(n_coh)-n_coh/2,0.5*np.ones(15),color='black',linestyle='--')
    #     ax.axvline(0,color='black',linestyle='--')
    #     ax.set_ylim([0,1])
    #     ax.set_ylabel('Probability Right Response')
    #     ax.set_xlabel('Evidence Right Choice (%)')
    #     plt.xticks(xx[indnan0],coh_plot[k][indnan0])
    #     fig.savefig('/home/ramon/Dropbox/Esteki_Kiani/figures/figure_neuro_pseudo_psychometric_%s_t_%i.pdf'%(monkeys[k],t_plot),dpi=500,bbox_inches='tight')

    # # # Figure Chronometric
    # # for t_plot in range(steps):
    # #     fig=plt.figure(figsize=(2.3,2))
    # #     ax=fig.add_subplot(111)
    # #     miscellaneous.adjust_spines(ax,['left','bottom'])
    # #     xx=np.arange(n_coh)-int(n_coh/2)
    # #     # Curve fit
    # #     indnan0=~np.isnan(chrono_neuro_m[t_plot,:,0])
    # #     popt0,pcov0=curve_fit(log_curve_abs,xx[indnan0],chrono_neuro_m[t_plot,:,0][indnan0])
    # #     yy0=log_curve_abs(xx[indnan0],popt0[0],popt0[1])
    # #     ax.scatter(xx[indnan0],chrono_neuro_m[t_plot,:,0][indnan0],color='black',s=3)
    # #     ax.plot(xx[indnan0],yy0,color='black')

    # #     indnan=~np.isnan(chrono_neuro_m[t_plot,:,1])
    # #     popt1,pcov1=curve_fit(log_curve_abs,xx[indnan],chrono_neuro_m[t_plot,:,1][indnan])
    # #     yy1=log_curve_abs(xx[indnan],popt1[0],popt1[1])
    # #     ax.scatter(xx[indnan],chrono_neuro_m[t_plot,:,1][indnan],color='green',s=3)
    # #     ax.plot(xx[indnan],yy1,color='green')

    # #     indnan=~np.isnan(chrono_neuro_m[t_plot,:,2])
    # #     popt2,pcov2=curve_fit(log_curve_abs,xx[indnan],chrono_neuro_m[t_plot,:,2][indnan])
    # #     yy2=log_curve_abs(xx[indnan],popt2[0],popt2[1])
    # #     ax.scatter(xx[indnan],chrono_neuro_m[t_plot,:,2][indnan],color='blue',s=3)
    # #     ax.plot(xx[indnan],yy2,color='blue')
         
    # #     ax.plot(np.arange(n_coh)-n_coh/2,0.5*np.ones(15),color='black',linestyle='--')
    # #     ax.axvline(0,color='black',linestyle='--')
    # #     ax.set_ylim([0,1])
    # #     ax.set_ylabel('Probability Right Response')
    # #     ax.set_xlabel('Evidence Right Choice (%)')
    # #     plt.xticks(xx[indnan0],coh_plot[k][indnan0])
    #     fig.savefig('/home/ramon/Dropbox/Esteki_Kiani/figures/figure_neuro_pseudo_chronometric_%s_t_%i.pdf'%(monkeys[k],t_plot),dpi=500,bbox_inches='tight')

        
#ax.plot(np.arange(15),psycho_neuro_m[t_plot,:,1],color='green')        
#ax.plot(np.arange(15),psycho_neuro_m[t_plot,:,2],color='blue')
#ax.fill_between(np.arange(15),psycho_neuro_m[t_plot,:,0]-psycho_neuro_std[t_plot,:,0],psycho_neuro_m[t_plot,:,0]+psycho_neuro_std[t_plot,:,0],color='black',alpha=0.6)
#ax.fill_between(np.arange(15),psycho_neuro_m[t_plot,:,1]-psycho_neuro_std[t_plot,:,1],psycho_neuro_m[t_plot,:,1]+psycho_neuro_std[t_plot,:,1],color='green',alpha=0.6)
#ax.fill_between(np.arange(15),psycho_neuro_m[t_plot,:,2]-psycho_neuro_std[t_plot,:,2],psycho_neuro_m[t_plot,:,2]+psycho_neuro_std[t_plot,:,2],color='blue',alpha=0.6)
            
 #     # Psychometric for the different time steps
    #     for i in range(steps):
    #         #print (' ',i)
    #         #try:
    #         sum_nan=np.sum(np.isnan(pseudo_tr_pre[i,0]),axis=1)
    #         index_nonan=np.where(sum_nan==0)[0]
    #         pseudo_tr=pseudo_tr_pre[i,0][index_nonan]
    #         pseudo_te=pseudo_te_pre[i,0][index_nonan]
    #         clase_all=clase_all_pre[index_nonan]
    #         clase_coh=clase_coh_pre[index_nonan]
    #         clase_ctx=clase_ctx_pre[index_nonan]
    #         stim=stim_pre[index_nonan]
                
    #         cl=LogisticRegression(C=1/reg,class_weight='balanced')
    #         cl.fit(pseudo_tr,stim) # Cuidado con decode Stim y no Choice!
                    
    #         for j in range(n_coh):
    #             #print ('  ',j)
    #             try:
    #                 ind_coh=np.where((clase_coh==j))[0]
    #                 ind_coh0=np.where((clase_coh==j)&(clase_ctx==0))[0] # Left more rewarded
    #                 ind_coh1=np.where((clase_coh==j)&(clase_ctx==1))[0] # Right more rewarded
                    
    #                 chrono_neuro_pre[i,ii,j,0]=cl.score(pseudo_te[ind_coh],stim[ind_coh])
    #                 chrono_neuro_pre[i,ii,j,1]=cl.score(pseudo_te[ind_coh0],stim[ind_coh0])
    #                 chrono_neuro_pre[i,ii,j,2]=cl.score(pseudo_te[ind_coh1],stim[ind_coh1])
    #                 psycho_neuro_pre[i,ii,j,0]=np.mean(cl.predict(pseudo_te[ind_coh]))
    #                 psycho_neuro_pre[i,ii,j,1]=np.mean(cl.predict(pseudo_te[ind_coh0]))
    #                 psycho_neuro_pre[i,ii,j,2]=np.mean(cl.predict(pseudo_te[ind_coh1]))
    #             except:
    #                 None
    #                 #print ('Error t ',i,'rand ',ii,'coh ',j)

    # psycho_neuro_m=np.mean(psycho_neuro_pre,axis=(1))
    # psycho_neuro_std=np.std(psycho_neuro_pre,axis=(1))
    # chrono_neuro_m=np.mean(chrono_neuro_pre,axis=(1))
    # chrono_neuro_std=np.std(chrono_neuro_pre,axis=(1))
    #print (psycho_neuro_m)
    #print (chrono_neuro_m)
