from __future__ import division
from __future__ import print_function

import os
import glob
import time
import random

import pickle

import random as rd

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from models import GAT



def load_data():
	print ('loading data')
	
	with open('128/y_128_full.pkl') as f:
		yl = pickle.load(f)

	with open('128/x_128_full.pkl') as f:
		xl = pickle.load(f)

	with open('128/adj_128_2D_renorm.pkl') as f:
		adjl = pickle.load(f)

	return yl,xl,adjl

torch.device('cuda:0'if torch.cuda.is_available() else 'cpu')

print ('loading data')
x,y,adj = load_data()



y_train = y[:60000]
x_train = x[:60000]
y_test = y[60000:]
x_test = x[60000:]
adj = torch.Tensor(adj)

print ('loading on cuda')


model = GAT(nfeat=1, 
                nhid=5, 
                nclass=1, 
                dropout=0.8, 
                nheads=2, 
                alpha=0.2)

optimizer = optim.Adam(model.parameters(),
                       lr=0.005	, weight_decay=5e-4)

model.cuda()

adj = adj.cuda()



def train(epoch):
#	LR = [1.00e-02, 8.89e-03, 7.78e-03, 6.67e-03, 5.56e-03, 4.45e-03, 3.34e-03, 2.23e-03, 1.12e-03, 1.00e-05]
	print ('Epoch: '+str(epoch))

	n_iter = 500

	batch_size = int(60000/n_iter)

	t = time.time()
	model.train()

	Loss = nn.SmoothL1Loss()

#	for g in optimizer.param_groups:
#   		g['lr'] = LR[epoch]
	
	for itn in range(n_iter):

		optimizer.zero_grad()

		loss_train = 0
		
		batch = np.array(rd.sample([i for i in range(60000)],batch_size))
		
                k = 0

		for _ in batch:
                        k+=1
                        
#			x_s = torch.stack([x[_][None,:].permute(1,0) for i in range(300)])
		       
			x_s = torch.Tensor(x_train[_]).float()[None, :].permute(1,0).cuda()
			y_s = torch.Tensor(y_train[_]).float().cuda()
			
		
			#print(x_s.size())
			output = model(x_s, adj, 1)
			
#			print (output.size(), y_s.size())
	
			
			loss_train = Loss(output, y_s)
                        loss_train.backward(retain_graph = True)
                        
		        optimizer.step()
                        print ('[ITER %d] [LOSS %f]' %(k, loss_train.item()))

		
		#loss_train/=batch_size


	print('Epoch: {:04d}'.format(epoch+1),
          'time: {:.4f}s'.format(time.time() - t))

def test():
	print ('testing')
	model.eval()
	O=[]
	for _ in range(len(x_test)):
		x_s = torch.Tensor(x_train[_]).float()[None, :].permute(1,0).cuda()
		output = model(x_s, adj)

		O.append(output)

	with open('31_365/pred_1_5_1_30_SL1.pkl','w') as f:
		pickle.dump(O,f)

		
	

t_total = time.time()
for epoch in range(10):
    train(epoch)
print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))


test()

print ('saving')
# save model name  - GCN_nfeat_nhid1..._nclass_batchsize
torch.save(model.state_dict(), '31_365/models/GCN_1_5_1_30_SL1.pt')
