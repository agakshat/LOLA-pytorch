import numpy as np
import torch
from torch.autograd import Variable
import IPDmodeling as ipdm
import avReturnIPD as r

dtype = torch.cuda.FloatTensor

y1 = Variable(torch.zeros(5,1).type(dtype),requires_grad = True)
y2 = Variable(torch.zeros(5,1).type(dtype),requires_grad = True)

r1 = Variable(torch.Tensor([-1,-3,0,-2]).type(dtype))
r2 = Variable(torch.Tensor([-1,0,-3,-2]).type(dtype))
I = Variable(torch.eye(4).type(dtype))

gamma = Variable(torch.Tensor([0.96]).type(dtype))
delta = Variable(torch.Tensor([0.1]).type(dtype))
eta = Variable(torch.Tensor([3]).type(dtype))

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
	V2_from1 = torch.matmul(torch.matmul(P1[0,:],Zinv1),r2) # V2 from agent 1's perspective
	V1_from2 = torch.matmul(torch.matmul(P2[0,:],Zinv2),r1)	# V1 from agent 2's perspective
#	V1arr.append(V1)
#	V2arr.append(V2)

	dV1 = torch.autograd.grad(V1,(y1,pm2Y),create_graph = True)
	dV2 = torch.autograd.grad(V2,(pm1Y,y2),create_graph = True)
	dV21 = torch.autograd.grad(V2_from1,pm2Y,create_graph = True)
	dV10 = torch.autograd.grad(V1_from2,pm1Y,create_graph = True)
	d2V2 = [torch.autograd.grad(dV21[0][i], y1, create_graph = True)[0] for i in range(y1.size(0))]
	d2V1 = [torch.autograd.grad(dV10[0][i], y2, create_graph = True)[0] for i in range(y1.size(0))]

	d2V2Tensor = torch.cat([d2V2[i] for i in range(y1.size(0))],1)
	d2V1Tensor = torch.cat([d2V1[i] for i in range(y1.size(0))],1)

	#y1.data += (delta*dV1[0] + delta*eta*torch.matmul(dV1[1].t(),d2V2Tensor.t()).t()).data
	#y2.data += (delta*dV2[1] + delta*eta*torch.matmul(dV2[0].t(),d2V1Tensor.t()).t()).data

	#y1.data += (delta*dV1[0] + delta*eta*torch.matmul(d2V2Tensor,dV1[1])).data
	y1.data += (delta*dV1[0]).data
	y2.data += (delta*dV2[1] + delta*eta*torch.matmul(d2V1Tensor,dV2[0])).data

	print("Epoch {}".format(epoch))
	if epoch%20==0:
		print("Rewards: {}".format(r.av_return(x1,x2)))
