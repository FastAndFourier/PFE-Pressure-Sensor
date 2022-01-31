import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.utils.data import DataLoader

class customDataset(torch.utils.data.Dataset):

    def __init__(self,path_action,path_obs,augmentData=False) -> None:

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

        if augmentData:
            self.dataAugmentation()

    def __len__(self):
        return len(self.obs)

    def __getitem__(self,idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        o = (self.obs[idx]/1028).flatten()
        a = self.action[idx]

        sample = [o,a]#{'observation': o, 'action': a}


        return sample

    def dataAugmentation(self):

        for k in range(len(self.obs)):
            self.obs = np.concatenate((self.obs,np.fliplr(self.obs[k])[np.newaxis]),axis=0)
            self.action = np.concatenate((self.action,self.action[k][::-1][np.newaxis]),axis=0)


class policyNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        #for k in range(len(layers)):
        self.lin1 = nn.Linear(18*9,32)
        self.lin2 = nn.Linear(32,16)
        self.lin3 = nn.Linear(16,2)
        
    def forward(self,x):
        
        x = torch.relu(self.lin1(x))
        x = torch.relu(self.lin2(x))
        x = torch.tanh(self.lin3(x))

        return x

def predict(obs,policy):

        obs = torch.flatten(torch.tensor(obs).float())
        print(policy)
        action = policy(obs).detach().numpy()
        action = [action[0]*10,action[1]*10]

        return action

def imitation(dataset,epochs,learning_rate):

    policy = policyNet()
    
    loss_func = nn.MSELoss()
    
    optimizer = torch.optim.SGD(policy.parameters(), lr=learning_rate, momentum=0.9)

    loss_array = []

    for epoch in range(epochs):
        
        running_loss = 0.0
        for i, data in enumerate(dataset, 0):

            inputs, labels = data

            optimizer.zero_grad()
            outputs = policy(inputs.float())
            loss = loss_func(outputs,labels.float())
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        loss_array.append(running_loss/dataset.__len__())


        if (epoch+1)%10==True:
            print(f'[{epoch}/{epochs}] loss: {running_loss/dataset.__len__():.3f}')
        


    print(f'[{epoch + 1}/{epochs}] loss: {running_loss/dataset.__len__():.3f}')

    return policy, loss_array

if __name__ == "__main__":


    epochs=1000
    batch_size=16
    augmentData = True

    dataset = customDataset(["./action.npy","./action1.npy","./action2.npy","./action3.npy"],
                            ["./observation.npy","./observation1.npy","./observation2.npy","./observation3.npy"],
                            augmentData=augmentData)
    

    len_trainDataset = int(.7*len(dataset))

    trainDataset, testDataset = torch.utils.data.random_split(dataset,[len_trainDataset,len(dataset)-len_trainDataset])

    
    print("Number of samples in the training set: ",trainDataset.__len__())

    trainLoader = DataLoader(trainDataset,batch_size=batch_size,shuffle=True)
    testLoader = DataLoader(testDataset,batch_size=batch_size,shuffle=True)

    policy, loss_array = imitation(trainLoader,epochs=epochs,learning_rate=0.001)
    torch.save(policy,"./expert_policy.pt")
    policy = torch.load('./expert_policy.pt')
    
    test_loss = 0.0
    loss_func = nn.MSELoss()
    
    with torch.no_grad():
        for sample in testLoader:
            obs, labels = sample
            outputs = policy(obs.float())
            temp =  loss_func(outputs,labels.float())
            test_loss += temp.item()

    print(f'Loss on test set {test_loss/testLoader.__len__():.3f}')

    plt.figure()
    plt.plot([round(loss_,2) for loss_ in loss_array])
    plt.show()

