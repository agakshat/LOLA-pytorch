import numpy as np
import torch
from torch.autograd import Variable

y1 = Variable(torch.zeros(5,1),requires_grad = True)
y2 = Variable(torch.zeros(5,1),requires_grad = True)

r1 = Variable(torch.Tensor([-1,-3,0,-2]))
r2 = Variable(torch.Tensor([-1,0,-3,-2]))
I = Variable(torch.eye(4))

gamma = 0.96
delta = 0.1

for epoch in range(5000):
	x1 = torch.sigmoid(y1)
	x2 = torch.sigmoid(y2)

	P = torch.cat((x1*x2,x1*(1-x2),(1-x1)*x2,(1-x1)*(1-x2)),1)
	Zinv = torch.inverse(I-gamma*P[1:,:])
	V1 = torch.matmul(torch.matmul(P[0,:],Zinv),r1)
	V2 = torch.matmul(torch.matmul(P[0,:],Zinv),r2)
	V1.backward(retain_graph=True)
	y1.data += delta*y1.grad.data
	#print("x1.grad.data ",x1.grad.data)
	y2.grad.data.zero_()
	V2.backward()
	y2.data += delta*y2.grad.data
	#print("x2.grad.data ",x2.grad.data)

	y1.grad.data.zero_()
	y2.grad.data.zero_()

	#print("Epoch: {}".format(epoch))

# Have to ensure that parameters represent probabilities - stay between 0 and 1