import numpy as np
import torch
from torch.autograd import Variable

dtype = torch.cuda.FloatTensor

y1 = Variable(torch.zeros(5,1).type(dtype),requires_grad = True)
y2 = Variable(torch.zeros(5,1).type(dtype),requires_grad = True)

r1 = Variable(torch.Tensor([-1,-3,0,-2]).type(dtype))
r2 = Variable(torch.Tensor([-1,0,-3,-2]).type(dtype))
I = Variable(torch.eye(4).type(dtype))

gamma = Variable(torch.Tensor([0.96]).type(dtype))
delta = Variable(torch.Tensor([0.1]).type(dtype))
eta = Variable(torch.Tensor([3]).type(dtype))

V1arr = []
V2arr = []

for epoch in range(5000):
	x1 = torch.sigmoid(y1)
	x2 = torch.sigmoid(y2)

	P = torch.cat((x1*x2,x1*(1-x2),(1-x1)*x2,(1-x1)*(1-x2)),1)
	Zinv = torch.inverse(I-gamma*P[1:,:])
	V1 = torch.matmul(torch.matmul(P[0,:],Zinv),r1)
	V2 = torch.matmul(torch.matmul(P[0,:],Zinv),r2)
	V1arr.append(V1)
	V2arr.append(V2)

	dV1 = torch.autograd.grad(V1,(y1,y2),create_graph = True)
	dV2 = torch.autograd.grad(V2,(y1,y2),create_graph = True)
	d2V2 = [torch.autograd.grad(dV2[1][i], y1, create_graph = True)[0] for i in range(y1.size(0))]
	d2V1 = [torch.autograd.grad(dV1[0][i], y2, create_graph = True)[0] for i in range(y1.size(0))]

	d2V2Tensor = torch.cat([d2V2[i] for i in range(y1.size(0))],1)
	d2V1Tensor = torch.cat([d2V1[i] for i in range(y1.size(0))],1)

	#y1.data += (delta*dV1[0] + delta*eta*torch.matmul(dV1[1].t(),d2V2Tensor.t()).t()).data
	#y2.data += (delta*dV2[1] + delta*eta*torch.matmul(dV2[0].t(),d2V1Tensor.t()).t()).data

	y1.data += (delta*dV1[0] + delta*eta*torch.matmul(d2V2Tensor,dV1[1])).data
	y2.data += (delta*dV2[1] + delta*eta*torch.matmul(d2V1Tensor,dV2[0])).data