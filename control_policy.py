import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class customDataset(torch.utils.data.Dataset):

    def __init__(self,path_action,path_obs) -> None:

        self.action = np.load(path_action[0])
        if len(path_action) > 1:
            for k in range(1,len(path_action)):
                self.action = np.concatenate((self.action,np.load(path_action[k])),axis=0)

        self.obs = np.load(path_obs[0])[1:]
        if len(path_obs) > 1:
            for k in range(1,len(path_obs)):
                self.obs = np.concatenate((self.obs,np.load(path_obs[k])[1:]),axis=0)


        self.zero = np.load(path_obs[0])[0]
        self.transform = None

    def __len__(self):
        return len(self.zero)

    def __getitem__(self,idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        o = (self.obs[idx]/1028).flatten()
        a = self.action[idx]

        sample = [o,a]#{'observation': o, 'action': a}


        return sample
        

class policyNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        #for k in range(len(layers)):
        self.lin1 = nn.Linear(18*9,32)
        self.lin2 = nn.Linear(32,16)
        self.lin3 = nn.Linear(16,2)
        
    def forward(self,x):
        
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x = F.tanh(self.lin3(x))

        return x

def predict(obs,policy):

    obs = torch.tensor(obs).float()
    action = policy(obs).detach().numpy()
    action = [action[0]*10,action[1]*10]

    return predict