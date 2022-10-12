import torch
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import scipy
#import matplotlib.pylab as plt
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
nan=float('nan')

dtype = torch.FloatTensor
#dtype = torch.cuda.FloatTensor # Uncomment this to run on GPU

# Input shape has to be: Batch x Steps x Input dim
# Target shape has to be: Batch x Steps (x Target dim)
# The output of the model concatenates all time steps from all trials

class nn_recurrent():
    def __init__(self,reg,lr,output_size,hidden_dim):
        self.regularization=reg
        self.learning_rate=lr
        self.loss=torch.nn.CrossEntropyLoss(reduction='none')
        self.model=recurrent_noisy(output_size,hidden_dim)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.regularization)

    def fit(self,input_seq,target_seq,batch_size,sigma_noise,wei_ctx): #,wei_task
        thres_fit=1e-3
        self.model.train()
        input_seq_np=np.array(input_seq,dtype=np.float32)
        context_np=np.array(input_seq[:,0,1],dtype=np.float32)
        ctx_uq=np.unique(context_np)
        target_seq_np=np.array(target_seq,dtype=np.int16)
        input_seq_torch=Variable(torch.from_numpy(input_seq_np),requires_grad=False)
        context_torch=Variable(torch.from_numpy(context_np),requires_grad=False)
        target_seq_torch=Variable(torch.from_numpy(target_seq_np),requires_grad=False)
        train_loader=DataLoader(torch.utils.data.TensorDataset(input_seq_torch,context_torch,target_seq_torch),batch_size=batch_size,shuffle=True)
        
        t_total=100
        for t in range(t_total):
            output, hidden, net_units, read_out_units = self.model(input_seq_torch,sigma_noise)
            l0c0=torch.mean(self.loss(output[(target_seq_torch==0)&(context_torch==ctx_uq[0])][:,[0,1]],target_seq_torch[(target_seq_torch==0)&(context_torch==ctx_uq[0])].view(-1).long()))
            l0c1=torch.mean(self.loss(output[(target_seq_torch==0)&(context_torch==ctx_uq[1])][:,[0,1]],target_seq_torch[(target_seq_torch==0)&(context_torch==ctx_uq[1])].view(-1).long()))
            l1c0=torch.mean(self.loss(output[(target_seq_torch==1)&(context_torch==ctx_uq[0])][:,[0,1]],target_seq_torch[(target_seq_torch==1)&(context_torch==ctx_uq[0])].view(-1).long()))
            l1c1=torch.mean(self.loss(output[(target_seq_torch==1)&(context_torch==ctx_uq[1])][:,[0,1]],target_seq_torch[(target_seq_torch==1)&(context_torch==ctx_uq[1])].view(-1).long()))
            l_total=(l0c0+l1c0+l0c1+l1c1)
            #if t==0 or t==(t_total-1):
            print (t,l_total.detach().numpy())
            #if (l0c0+l1c0+l0c1+l1c1)<thres_fit:
            #    break
            for batch_idx, (data, contxt, targets) in enumerate(train_loader):
                self.optimizer.zero_grad()
                output, hidden, net_units, read_out_units = self.model(data,sigma_noise)
                loss0_ct0=torch.mean(self.loss(output[(targets==0)&(contxt==ctx_uq[0])][:,[0,1]],targets[(targets==0)&(contxt==ctx_uq[0])].view(-1).long()))
                loss0_ct1=torch.mean(self.loss(output[(targets==0)&(contxt==ctx_uq[1])][:,[0,1]],targets[(targets==0)&(contxt==ctx_uq[1])].view(-1).long()))
                loss1_ct0=torch.mean(self.loss(output[(targets==1)&(contxt==ctx_uq[0])][:,[0,1]],targets[(targets==1)&(contxt==ctx_uq[0])].view(-1).long()))
                loss1_ct1=torch.mean(self.loss(output[(targets==1)&(contxt==ctx_uq[1])][:,[0,1]],targets[(targets==1)&(contxt==ctx_uq[1])].view(-1).long()))
                loss=(wei_ctx[0]*(loss0_ct0+loss1_ct1)+wei_ctx[1]*(loss1_ct0+loss0_ct1))
                #loss_ctx=(torch.mean(self.loss(output[:,[2,3]],contxt.view(-1).long())))
                #loss=(wei_task[0]*loss_rdm+wei_task[1]*loss_ctx)
                loss.backward()
                self.optimizer.step()
        return self.model.state_dict()

    # def score(self,input_seq,target_seq,sigma_noise):                                              
    #     self.model.eval()
    #     input_seq_np=np.array(input_seq,dtype=np.float32)
    #     target_seq_np=np.array(target_seq,dtype=np.int16)
    #     input_seq_torch=Variable(torch.from_numpy(input_seq_np),requires_grad=False)
    #     target_seq_torch=Variable(torch.from_numpy(target_seq_np),requires_grad=False)
    #     test_loader=DataLoader(torch.utils.data.TensorDataset(input_seq_torch,target_seq_torch),batch_size=len(target_seq),shuffle=False)
        
    #     for batch_idx, (data,targets) in enumerate(test_loader):
    #         output, hidden, net_units, read_out_units = self.model(data,sigma_noise)
    #         y_pred=np.argmax(output.detach().numpy(),axis=1)
    #         target_np=targets.detach().numpy()
    #         error=np.mean(abs(y_pred-target_np))
    #     return 1.0-error

    
class recurrent_noisy(torch.nn.Module): # We always send the input with size batch x time steps x input dim
    def __init__(self, output_size, hidden_dim):
        super(recurrent_noisy, self).__init__()
        self.hidden_dim=hidden_dim
        self.output_size=output_size
        self.input_weights = torch.nn.Linear(2, hidden_dim)
        self.hidden_weights = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc = torch.nn.Linear(self.hidden_dim, self.output_size)

    def forward(self, input, sigma_noise, hidden=None):
        if hidden is None:
            #hidden = torch.randn(input.size(0),self.hidden_dim).to(input.device)
            hidden = torch.zeros(input.size(0),self.hidden_dim).to(input.device)
        
        def recurrence(input, hidden):
            h_new = torch.relu(self.input_weights(input) + self.hidden_weights(hidden) + sigma_noise*torch.randn(input.size(0),self.hidden_dim))
            return h_new        
        # def recurrence_lin(input, hidden):
        #     h_new = (self.input_weights(input) + self.hidden_weights(hidden) + sigma_noise*torch.randn(input.size(0),self.hidden_dim))
        #     return h_new

        net_units = torch.zeros(input.size(0),input.size(1),self.hidden_dim)
        steps = range(input.size(1))
        for i in steps:
            hidden = recurrence(input[:,i], hidden)
            #hidden = recurrence_lin(input[:,i], hidden)
            net_units[:,i]=hidden
            
        hidden = hidden.detach()
        out = net_units[:,-1].contiguous().view(-1, self.hidden_dim)
        out = self.fc(out)
        read_out_units = self.fc(net_units)
        return out, hidden, net_units, read_out_units
            









