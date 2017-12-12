import numpy as np

def av_return(policy1,policy2,r1arr = [-1,-3,0,-2],r2arr = [-1,0,-3,-2],rollout_length=10000,num_rollout=100,verbose=False):
	policy1 = policy1.data.cpu().numpy().tolist()
	policy2 = policy2.data.cpu().numpy().tolist()
	reward = []
	for _ in range(num_rollout):
		s = [0,0]
		s[0] = np.random.choice([0,1],p = [policy1[0][0],1-policy1[0][0]])
		s[1] = np.random.choice([0,1],p = [policy2[0][0],1-policy2[0][0]])
		r1 = 0
		r2 = 0
		if verbose:
			print("Initial states are {}".format(s))
		for i in range(rollout_length):
			if s[0]==0 and s[1]==0:
				a = 1
			elif s[0]==0 and s[1]==1:
				a = 2
				if verbose:
					print("Coop/Def")
			elif s[0]==1 and s[1]==0:
				a = 3
				if verbose:
					print("Def/Coop")
			else:
				a = 4
				if verbose:
					print("Both Defected!")
			r1 = r1 + r1arr[a-1]
			r2 = r2 + r2arr[a-1]
			s[0] = np.random.choice([0,1],p = [policy1[a][0],1-policy1[a][0]])
			s[1] = np.random.choice([0,1],p = [policy2[a][0],1-policy2[a][0]])
			#print(s)
		r1 = r1/rollout_length
		r2 = r2/rollout_length
		reward.append([r1,r2])
	reward = np.asarray(reward)
	r1 = np.mean(reward[:,0])
	r2 = np.mean(reward[:,1])
	return r1,r2