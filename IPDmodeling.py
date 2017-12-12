import numpy as np

def av_return(policy1,policy2,r1arr = [-1,-3,0,-2],r2arr = [-1,0,-3,-2],rollout_length=1000,num_rollout=100,gamma=0.96):
	policy1 = policy1.data.cpu().numpy().tolist()
	policy2 = policy2.data.cpu().numpy().tolist()
	reward = []
	p1C = [0,0,0,0,0]
	p1Total = [0,0,0,0,0]
	p2C = [0,0,0,0,0]
	p2Total = [0,0,0,0,0]
	for _ in range(num_rollout):
		s = [0,0]
		s[0] = np.random.choice([0,1],p = [policy1[0][0],1-policy1[0][0]]) # 0 means Cooperate, 1 means Defect
		s[1] = np.random.choice([0,1],p = [policy2[0][0],1-policy2[0][0]])
		p1Total[0]+=1
		p2Total[0]+=1
		if s[0]==0:
			p1C[0] += 1
		if s[1]==0:
			p2C[0] += 1
		r1 = 0
		r2 = 0
		for i in range(rollout_length):
			if s[0]==0 and s[1]==0:
				a = 1
			elif s[0]==0 and s[1]==1:
				a = 2
			elif s[0]==1 and s[1]==0:
				a = 3
			else:
				a = 4
			#r1 = r1 + r1arr[a-1]
			#r2 = r2 + r2arr[a-1]
			s[0] = np.random.choice([0,1],p = [policy1[a][0],1-policy1[a][0]])
			s[1] = np.random.choice([0,1],p = [policy2[a][0],1-policy2[a][0]])
			#print(s)
			p1Total[a]+=1
			p2Total[a]+=1
			if s[0]==0:
				p1C[a]+=1
			if s[1]==0:
				p2C[a]+=1
		#r1 = r1/rollout_length
		#r2 = r2/rollout_length
		#reward.append([r1,r2])
	#reward = np.asarray(reward)
	#r1 = np.mean(reward[:,0])
	#r2 = np.mean(reward[:,1])
	pm1 = np.asarray(p1C)/np.asarray(p1Total)
	pm2 = np.asarray(p2C)/np.asarray(p2Total)
	pm1_y  = np.log(np.divide(pm1,1-pm1))
	pm2_y = np.log(np.divide(pm2,1-pm2))
	return pm1_y.reshape((5,1)),pm2_y.reshape((5,1))