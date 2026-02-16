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
import statsmodels.api as sm

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

def index_neu_lr(coherence,ind_ch,ind_ch01,ind_ch10,t_forw,t_back):
    ind_train=np.arange(len(coherence))
    #ind_ch_all=[]
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
        #ind_ch_all.append(ind_del)

    #ind_ch_all=np.reshape(np.array(ind_ch_all),-1)
    #print (len(context),len(ind_train),np.mean(context[ind_train]))
    return ind_train#,ind_ch_all


def calculate_weights_lr(files,reg,talig,dic_time,t_back,t_forw,thres):
    dic={}
    y_matrix01_neu_pre=[]
    y_matrix01_pre=[]
    x_matrix01_pre=[]
    y_matrix10_neu_pre=[]
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
        coherence=100*beha['coherence_signed'][1:]
        coh_uq=np.unique(coherence)
        reward=beha['reward'][1:]
        rt=beha['reaction_time'][1:]
        context_prepre=beha['context']
        ctx_ch_pre=(context_prepre[1:]-context_prepre[0:-1])
        context_pre=context_prepre[1:]
        #ind_ch=np.where(abs(ctx_ch)==1)[0]
        ind_ch_pre=np.where(abs(ctx_ch_pre)==1)[0] #Careful!
        ind_ch=miscellaneous.calculate_ind_ch_corr(ind_ch_pre,reward) # ind_ch first correct trial after context change 
        context=miscellaneous.create_context_subj(context_pre,ind_ch_pre,ind_ch) # Careful! this is subjective context. 
        
        ctx_ch=nan*np.zeros(len(reward))
        ctx_ch[1:]=(context[1:]-context[0:-1])
        ctx_ch[0]=0
        indch_ct10=np.where(ctx_ch==-1)[0]
        indch_ct01=np.where(ctx_ch==1)[0]

        firing_rate_pre=miscellaneous.getRasters_unsorted(data,talig,dic_time,index_nonan,threshold=thres)
        firing_rate=miscellaneous.normalize_fr(firing_rate_pre)[1:,:,0]

        ind_train=index_neu_lr(coherence,ind_ch,indch_ct01,indch_ct10,t_back,t_forw)
        
        #cl=LogisticRegression(C=1/reg,class_weight='balanced')
        #cl.fit(firing_rate[ind_train],choice[ind_train])
        #choice_neu=cl.predict(firing_rate)
        fr_sm=sm.add_constant(firing_rate)
        cl = sm.Logit(choice[ind_train],fr_sm[ind_train])
        model=cl.fit(disp=False)
        choice_neu = np.array(cl.predict(params=model.params,exog=fr_sm)>0.5,dtype=np.int16)
            
        # j+1 because the index 0 is also "before knowing there was a change"
        for h in range(len(indch_ct01)): #from left to right            
            for j in range(t_back):
                try:
                    y_matrix01_neu_pre.append(choice_neu[indch_ct01[h]-t_back+j+1])
                    y_matrix01_pre.append(choice[indch_ct01[h]-t_back+j+1])
                    x_matrix01_pre.append([coherence[indch_ct01[h]-t_back+j+1],0])
                except:
                    None
            for j in range(t_forw):
                try:
                    y_matrix01_neu_pre.append(choice_neu[indch_ct01[h]+j+1])
                    y_matrix01_pre.append(choice[indch_ct01[h]+j+1])
                    x_matrix01_pre.append([coherence[indch_ct01[h]+j+1],j+1])
                except:
                    None

        for h in range(len(indch_ct10)): #from left to right
            for j in range(t_back):
                try:
                    y_matrix10_neu_pre.append(choice_neu[indch_ct10[h]-t_back+j+1])
                    y_matrix10_pre.append(choice[indch_ct10[h]-t_back+j+1])
                    x_matrix10_pre.append([coherence[indch_ct10[h]-t_back+j+1],0])
                except:
                    None
            for j in range(t_forw):
                try:
                    y_matrix10_neu_pre.append(choice_neu[indch_ct10[h]+j+1])
                    y_matrix10_pre.append(choice[indch_ct10[h]+j+1])
                    x_matrix10_pre.append([coherence[indch_ct10[h]+j+1],j+1])
                except:
                    None

    y_matrix01_neu=np.array(y_matrix01_neu_pre,dtype=np.int16)
    y_matrix01=np.array(y_matrix01_pre,dtype=np.int16)
    x_matrix01=np.array(x_matrix01_pre)
    x01_sm=sm.add_constant(x_matrix01)
    # Behavioral
    #b01=LogisticRegression(C=reg,class_weight='balanced')
    #b01.fit(x_matrix01,y_matrix01)
    #aa01=np.zeros(3)
    #aa01[0]=b01.intercept_[0]
    #aa01[1:]=b01.coef_[0]
    cl01=sm.Logit(y_matrix01,x01_sm).fit(disp=False)
    # Neuronal
    #b01_neu=LogisticRegression(C=reg,class_weight='balanced')
    #b01_neu.fit(x_matrix01,y_matrix01_neu)
    #aa01_neu=np.zeros(3)
    #aa01_neu[0]=b01_neu.intercept_[0]
    #aa01_neu[1:]=b01_neu.coef_[0]
    cl01_neu=sm.Logit(y_matrix01_neu,x01_sm).fit(disp=False)
    dic['Left to Right']=cl01.params
    dic['Left to Right bse']=cl01.bse
    dic['Left to Right neu']=cl01_neu.params
    dic['Left to Right neu bse']=cl01_neu.bse

    y_matrix10_neu=np.array(y_matrix10_neu_pre,dtype=np.int16)
    y_matrix10=np.array(y_matrix10_pre,dtype=np.int16)
    x_matrix10=np.array(x_matrix10_pre)
    x10_sm=sm.add_constant(x_matrix10)
    # Behavioral
    #b10=LogisticRegression(C=reg,class_weight='balanced')
    #b10.fit(x_matrix10,y_matrix10)
    #aa10=np.zeros(3)
    #aa10[0]=b10.intercept_[0]
    #aa10[1:]=b10.coef_[0]
    cl10=sm.Logit(y_matrix10,x10_sm).fit(disp=False)
    # Neuronal
    #b10_neu=LogisticRegression(C=reg,class_weight='balanced')
    #b10_neu.fit(x_matrix10,y_matrix10_neu)
    #aa10_neu=np.zeros(3)
    #aa10_neu[0]=b10_neu.intercept_[0]
    #aa10_neu[1:]=b10_neu.coef_[0]
    cl10_neu=sm.Logit(y_matrix10_neu,x10_sm).fit(disp=False)
    dic['Right to Left']=cl10.params
    dic['Right to Left bse']=cl10.bse
    dic['Right to Left neu']=cl10_neu.params
    dic['Right to Left neu bse']=cl10_neu.bse

    # # Permutation different regressors
    # aa01_shx=np.zeros((n_shx,3))
    # aa10_shx=np.zeros((n_shx,3))
    # for nn in range(n_shx):
    #     y_matrix01_sh=np.random.permutation(np.array(y_matrix01_pre,dtype=np.int16))
    #     b01_shx=LogisticRegression(C=reg,class_weight='balanced')
    #     b01_shx.fit(x_matrix01,y_matrix01_sh)
    #     aa01_shx[nn,0]=b01_shx.intercept_[0]
    #     aa01_shx[nn,1:]=b01_shx.coef_[0]
    #     y_matrix10_sh=np.random.permutation(np.array(y_matrix10_pre,dtype=np.int16))
    #     b10_shx=LogisticRegression(C=reg,class_weight='balanced')
    #     b10_shx.fit(x_matrix10,y_matrix10_sh)
    #     aa10_shx[nn,0]=b10_shx.intercept_[0]
    #     aa10_shx[nn,1:]=b10_shx.coef_[0]
    # dic['Left to Right sh x']=aa01_shx
    # dic['Right to Left sh x']=aa10_shx

    return dic

  
#################################################

monkeys=['Niels','Galileo']
stage_vec=['early','mid','late']

talig='dots_on' #'response_edf' #dots_on
thres=0
reg=1e0
n_shx=0
n_sh=0

t_back=20
t_forw=30#1
xx=np.arange(t_back+t_forw)-t_back

group_ref=np.array([-7 ,-6 ,-5 ,-4 ,-3 ,-2 ,-1 ,0  ,1  ,2  ,3  ,4  ,5  ,6  ,7  ])

# Original
wei=nan*np.zeros((len(monkeys),len(stage_vec),2,3))
wei_std=nan*np.zeros((len(monkeys),len(stage_vec),2,3))
wei_neu=nan*np.zeros((len(monkeys),len(stage_vec),2,3))
wei_neu_std=nan*np.zeros((len(monkeys),len(stage_vec),2,3))
#wei_shx=nan*np.zeros((len(monkeys),len(stage_vec),n_shx,2,3))
for k in range(len(monkeys)):
    abs_path='/home/ramon/Dropbox/Proyectos_Postdoc/Esteki_Kiani/data/unsorted/%s/'%(monkeys[k]) 
    files_pre=np.array(os.listdir(abs_path))
    order=miscellaneous.order_files(files_pre)
    files_all=np.array(files_pre[order])
    print (files_all)
    for ss in range(len(stage_vec)):
        stage=stage_vec[ss]
        print (stage)
        if monkeys[k]=='Niels':
            dic_time=np.array([0,400,400,400])# time pre, time post, bin size, step size (time pre always positive)
            if stage=='early':
                fd=0
                fu=4
            if stage=='mid':
                fd=4
                fu=8
            if stage=='late':
                fd=8
                fu=12
        if monkeys[k]=='Galileo':
            dic_time=np.array([0,600,600,600])# time pre, time post, bin size, step size (time pre always positive)
            if stage=='early':
                fd=0
                fu=10
            if stage=='mid':
                fd=10
                fu=20
            if stage=='late':
                fd=20
                fu=30
  
        beha_lr=calculate_weights_lr(files_all[fd:fu],reg,talig,dic_time,t_back,t_forw,thres)
        # Behavioral
        print (beha_lr['Left to Right'],np.shape(beha_lr['Left to Right']))
        wei[k,ss,0]=beha_lr['Left to Right']
        wei[k,ss,1]=beha_lr['Right to Left']
        wei_std[k,ss,0]=beha_lr['Left to Right bse']
        wei_std[k,ss,1]=beha_lr['Right to Left bse']
        # Neuronal
        wei_neu[k,ss,0]=beha_lr['Left to Right neu']
        wei_neu[k,ss,1]=beha_lr['Right to Left neu']
        wei_neu_std[k,ss,0]=beha_lr['Left to Right neu bse']
        wei_neu_std[k,ss,1]=beha_lr['Right to Left neu bse']
        
        #wei_shx[k,ss,:,0]=beha_lr['Left to Right sh x']
        #wei_shx[k,ss,:,1]=beha_lr['Right to Left sh x']

# change_vec=['01','10']
# reg_vec=[0,1,2,3]
# for i in range(2):
#     print ('Monkeys %s'%monkeys[i])
#     for ii in range(3):
#         print ('stage %s'%stage_vec[ii])
#         for iii in range(2):
#             for iiii in range(3):
#                 plt.hist(wei_shx[i,ii,:,iii,iiii])
#                 plt.axvline(np.mean(wei_shx[i,ii,:,iii,iiii]),color='blue')
#                 plt.axvline(wei[i,ii,iii,iiii],color='black')
#                 plt.xlabel('Regressor weight')
#                 plt.ylabel('Frequency')
#                 plt.savefig('/home/ramon/Desktop/figs_roozbeh/lr_%s_%s_%s_omega%i'%(monkeys[i],stage_vec[ii],change_vec[iii],reg_vec[iiii]))
#                 plt.close()

# Behavioral
wei_m=np.mean(abs(wei),axis=2)
wei_mb_pre=np.mean(wei,axis=0)
wei_mb0=1/2*(wei_mb_pre[:,1,0]-wei_mb_pre[:,0,0])
wei_mb1=1/2*(wei_mb_pre[:,0,1]+wei_mb_pre[:,1,1])
wei_mb2=1/2*(wei_mb_pre[:,0,2]-wei_mb_pre[:,1,2])
w2c=(wei_mb2/wei_mb1) # size 3
std_mb=1/4*np.sqrt(wei_std[0,:,0]**2+wei_std[0,:,1]**2+wei_std[1,:,0]**2+wei_std[1,:,1]**2) # propagation of errors. Size: stage x params
w2c_std=(1/wei_mb1)*np.sqrt((std_mb[:,2]**2)+(w2c**2)*(std_mb[:,1]**2))

fig=plt.figure(figsize=(2.3,2))
ax=fig.add_subplot(111)
miscellaneous.adjust_spines(ax,['left','bottom'])
ax.plot(np.arange(len(stage_vec)),w2c,color='green')
ax.fill_between(np.arange(len(stage_vec)),w2c-w2c_std,w2c+w2c_std,color='green',alpha=0.5)
#ax.plot(np.arange(len(stage_vec)),np.zeros(len(stage_vec)),color='black',linestyle='--')
#ax.set_ylabel('Bias ($\beta_{2} \beta_{1}$)')
#ax.set_ylim([-150,250])
fig.savefig('/home/ramon/Dropbox/Proyectos_Postdoc/Esteki_Kiani/plots/bias_learning_roozbeh.pdf',dpi=500,bbox_inches='tight')

# Wald Test on the difference between betas (one sided)
w2c_diff=(w2c[2]-w2c[0])
se_diff=np.sqrt(w2c_std[0]**2+w2c_std[2]**2)
z_diff=w2c_diff/se_diff
p_val=(1-scipy.stats.norm.cdf(z_diff))

# Neuronal
wei_neu_m=np.mean(abs(wei_neu),axis=2)
wei_neu_mb_pre=np.mean(wei_neu,axis=0)
wei_neu_mb0=1/2*(wei_neu_mb_pre[:,1,0]-wei_neu_mb_pre[:,0,0])
wei_neu_mb1=1/2*(wei_neu_mb_pre[:,0,1]+wei_neu_mb_pre[:,1,1])
wei_neu_mb2=1/2*(wei_neu_mb_pre[:,0,2]-wei_neu_mb_pre[:,1,2])
w2c_neu=(wei_neu_mb2/wei_neu_mb1) # size 3
std_neu_mb=1/4*np.sqrt(wei_neu_std[0,:,0]**2+wei_neu_std[0,:,1]**2+wei_neu_std[1,:,0]**2+wei_neu_std[1,:,1]**2) # propagation of errors. Size: stage x params
w2c_neu_std=(1/wei_neu_mb1)*np.sqrt((std_neu_mb[:,2]**2)+(w2c_neu**2)*(std_neu_mb[:,1]**2))

fig=plt.figure(figsize=(2.3,2))
ax=fig.add_subplot(111)
miscellaneous.adjust_spines(ax,['left','bottom'])
ax.plot(np.arange(len(stage_vec)),w2c_neu,color='blue')
ax.fill_between(np.arange(len(stage_vec)),w2c_neu-w2c_neu_std,w2c_neu+w2c_neu_std,color='blue',alpha=0.5)
#ax.plot(np.arange(len(stage_vec)),np.zeros(len(stage_vec)),color='black',linestyle='--')
#ax.set_ylabel('Bias ($\beta_{2} \beta_{1}$)')
#ax.set_ylim([-150,250])
fig.savefig('/home/ramon/Dropbox/Proyectos_Postdoc/Esteki_Kiani/plots/bias_neu_learning_roozbeh.pdf',dpi=500,bbox_inches='tight')

# Wald Test on the difference between betas (one sided)
w2c_diff_neu=(w2c_neu[2]-w2c_neu[0])
se_diff_neu=np.sqrt(w2c_neu_std[0]**2+w2c_neu_std[2]**2)
z_diff_neu=w2c_diff_neu/se_diff_neu
p_val_neu=(1-scipy.stats.norm.cdf(z_diff_neu))

print ('P-Value Behavior ',p_val)
print ('P-Value Neuronal ',p_val_neu)

# # Shuffled
# wei_sh=nan*np.zeros((len(monkeys),n_sh,len(stage_vec),2,3))
# for k in range(len(monkeys)):
#     abs_path='/home/ramon/Dropbox/Proyectos_Postdoc/Esteki_Kiani/data/unsorted/%s/'%(monkeys[k]) 
#     files_pre=np.array(os.listdir(abs_path))
#     order=miscellaneous.order_files(files_pre)
#     files_all=np.array(files_pre[order])
    
#     for nn in range(n_sh):
#         files_sh=np.random.permutation(files_all)
#         print (files_sh)
        
#         for ss in range(len(stage_vec)):
#             stage=stage_vec[ss]
#             if monkeys[k]=='Niels':
#                 if stage=='early':
#                     files_groups=[[0,1],[1,2],[2,3],[3,4]]
#                 if stage=='mid':
#                     files_groups=[[4,5],[5,6],[6,7],[7,8]]
#                 if stage=='late':
#                     files_groups=[[8,9],[9,10],[10,11],[11,12]]
#             if monkeys[k]=='Galileo':
#                 if stage=='early':
#                     files_groups=[[0,2],[2,4],[4,6],[6,8],[8,10]]
#                 if stage=='mid':
#                     files_groups=[[10,12],[12,14],[14,16],[16,18],[18,20]]
#                 if stage=='late':
#                     files_groups=[[20,22],[22,24],[24,26],[26,28],[28,30]]

#             beha_lr=calculate_weights_lr(files_sh,files_groups,reg,0)
#             wei_sh[k,nn,ss,0]=beha_lr['Left to Right']
#             wei_sh[k,nn,ss,1]=beha_lr['Right to Left']
            
# wei_sh_m=np.mean(abs(wei_sh),axis=3)
# wei_sh_mb=np.mean(abs(wei_sh),axis=(0,3))

# Individual monkeys
# for i in range(2):
#     for j in range(3):
#         plt.plot(wei_m[i,:,j],color='green')
#         am=np.mean(wei_sh_m[i,:,:,j],axis=0)
#         astd=np.std(wei_sh_m[i,:,:,j],axis=0)
#         plt.plot(am,color='black',alpha=0.5)
#         plt.plot(am-astd,color='black',alpha=0.5,linestyle='--')
#         plt.plot(am+astd,color='black',alpha=0.5,linestyle='--')
#         plt.ylabel('Absolute Weight')
#         plt.xlabel('Stage')
#         plt.xticks([0,1,2],['Early','Middle','Late'])
#         plt.show()

# plt.plot(wei_m[0,:,1],color='green')
# plt.ylabel('Absolute Weight')
# plt.xlabel('Stage')
# plt.xticks([0,1,2],['Early','Middle','Late'])
# plt.show()

        
# # Combine both monkeys
# for j in range(3):
#     plt.plot(wei_mb[:,j],color='green')
#     am=np.mean(wei_sh_mb[:,:,j],axis=0)
#     astd=np.std(wei_sh_mb[:,:,j],axis=0)
#     plt.plot(am,color='black',alpha=0.5)
#     plt.plot(am-astd,color='black',alpha=0.5,linestyle='--')
#     plt.plot(am+astd,color='black',alpha=0.5,linestyle='--')
#     plt.ylabel('Absolute Weight')
#     plt.xlabel('Stage')
#     plt.xticks([0,1,2],['Early','Middle','Late'])
#     plt.show()
    

        
