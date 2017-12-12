import numpy as np
import torch
from torch.autograd import Variable
import IPDmodeling as ipdm

dtype = torch.cuda.FloatTensor

y1 = Variable(torch.zeros(5,1).type(dtype),requires_grad = True)
y2 = Variable(torch.zeros(5,1).type(dtype),requires_grad = True)

r1 = Variable(torch.Tensor([-1,-3,0,-2]).type(dtype))
r2 = Variable(torch.Tensor([-1,0,-3,-2]).type(dtype))
I = Variable(torch.eye(4).type(dtype))

gamma = Variable(torch.Tensor([0.96]).type(dtype))
delta = Variable(torch.Tensor([0.1]).type(dtype))

for epoch in range(1000):
	x1 = torch.sigmoid(y1)
	x2 = torch.sigmoid(y2)

	pm1Y,pm2Y = ipdm.av_return(x1,x2,r1,r2)
	pm1Y = Variable(torch.from_numpy(pm1Y).float().cuda(),requires_grad=True)
	pm2Y = Variable(torch.from_numpy(pm2Y).float().cuda(),requires_grad=True)
	pm1 = torch.sigmoid(pm1Y)
	pm2 = torch.sigmoid(pm2Y)

	P1 = torch.cat((x1*pm2,x1*(1-pm2),(1-x1)*pm2,(1-x1)*(1-pm2)),1) # Agent 1 knows own policy, and models agent 2's policy
	P2 = torch.cat((pm1*x2,pm1*(1-x2),(1-pm1)*x2,(1-pm1)*(1-x2)),1) # Agent 2 knows its own policy, and models agent 1's policy

	Zinv1 = torch.inverse(I-gamma*P1[1:,:])
	Zinv2 = torch.inverse(I-gamma*P2[1:,:])

	V1 = torch.matmul(torch.matmul(P1[0,:],Zinv1),r1)
	V2 = torch.matmul(torch.matmul(P2[0,:],Zinv2),r2)

	V1.backward(retain_graph=True)
	y1.data += delta.data*y1.grad.data
	#print("x1.grad.data ",x1.grad.data)
	#y2.grad.data.zero_()
	V2.backward()
	y2.data += delta.data*y2.grad.data
	#print("x2.grad.data ",x2.grad.data)

	y1.grad.data.zero_()
	y2.grad.data.zero_()

	#print("Epoch: {}".format(epoch))

# Have to ensure that parameters represent probabilities - stay between 0 and 1