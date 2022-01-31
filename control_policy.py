import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.utils.data import DataLoader

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


# class upperCrop(object):

#     def __call__(self, sample):
#         return sample[7:]

class normalize(object):

    def __call__(self,sample):
        
        mu = np.mean(sample)
        std = np.std(sample)

        return (sample-mu)/std


class customDataset(torch.utils.data.Dataset):

    def __init__(self,path_action,path_obs,transform=[None],augmentData=False) -> None:

        self.action = np.load(path_action[0])
        if len(path_action) > 1:
            for k in range(1,len(path_action)):
                self.action = np.concatenate((self.action,np.load(path_action[k])),axis=0)

        self.obs = np.load(path_obs[0])[1:]
        if len(path_obs) > 1:
            for k in range(1,len(path_obs)):
                self.obs = np.concatenate((self.obs,np.load(path_obs[k])[1:]),axis=0)


        self.zero = np.load(path_obs[0])[0]
        
        self.transform = transform
        
        if augmentData:
            self.dataAugmentation()
        
        if self.transform:
            
            for transform in self.transform:
                self.zero = transform(self.zero)
            
            temp_obs = []
            for k in range(len(self.obs)):
                sample = self.obs[k]
                for transform in self.transform:
                    sample = transform(sample)
                temp_obs.append(sample)
            self.obs = temp_obs

    def __len__(self):
        return len(self.obs)

    def __getitem__(self,idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        o = (self.obs[idx]).flatten()
        a = self.action[idx]

        sample = [o,a]

        return sample

    def dataAugmentation(self):

        shape_data = self.obs[0].shape
        print("Before augmentation: ",self.obs.shape)
        for k in range(len(self.obs)):
            self.obs = np.append(self.obs,np.fliplr(self.obs[k])[np.newaxis],axis=0)
            self.action = np.append(self.action,np.array([self.action[k][0],-self.action[k][1]])[np.newaxis],axis=0)
            #self.obs = np.concatenate((self.obs,np.fliplr(self.obs[k])[np.newaxis]),axis=0)
            #self.action = np.concatenate((self.action,np.array([self.action[k][0],-self.action[k][1]])[np.newaxis]),axis=0)

        for k in range(len(self.obs)):
            noise = np.random.normal(loc=0,scale=1.5,size=tuple(shape_data))
            self.obs = np.append(self.obs,np.array(self.obs[k]+noise)[np.newaxis],axis=0)
            self.action = np.append(self.action,self.action[k][np.newaxis],axis=0)
            
        print("After augmentation: ",self.obs.shape)
        

class policyNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
        self.lin1 = nn.Linear(11*9,32)
        self.lin2 = nn.Linear(32,16)
        self.do2 = nn.Dropout(p=0.2)
        self.lin3 = nn.Linear(16,2)
        
    def forward(self,x):
        #print(x.shape)
        x = torch.relu(self.lin1(x))
        x = torch.relu(self.lin2(x))
        x = self.do2(x)
        x = torch.tanh(self.lin3(x))

        return x

def predict(obs,policy):

        n_obs = obs.shape[0]
        #if type(obs==np.ndarray) and obs.shape[1]!=99:
        if obs.shape[1]==99:
            obs = torch.tensor(obs).float()
        else:
            obs = torch.tensor(obs.reshape(n_obs,9*11)).float()
        #print(len(obs))
        action = policy(obs).detach().numpy()
        #print(len(action))
        action = [[round(a[0]*10),round(a[1]*10)] for a in action]

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


    epochs=800
    batch_size=16
    augmentData = True

    dataset = customDataset(["./action.npy","./action1.npy","./action2.npy","./action3.npy"],
                            ["./observation_cropped.npy","./observation1_cropped.npy",
                            "./observation2_cropped.npy","./observation3_cropped.npy"],
                            augmentData=augmentData,transform=[normalize()])
    

    len_trainDataset = int(.8*len(dataset))

    trainDataset, testDataset = torch.utils.data.random_split(dataset,[len_trainDataset,len(dataset)-len_trainDataset])

    
    print("Number of samples in the training set: ",trainDataset.__len__())

    trainLoader = DataLoader(trainDataset,batch_size=batch_size,shuffle=True)
    testLoader = DataLoader(testDataset,batch_size=batch_size,shuffle=True)

    #policy, loss_array = imitation(trainLoader,epochs=epochs,learning_rate=0.001)
    #torch.save(policy,"./expert_policy_truncated.pt")
    policy = torch.load('./expert_policy_truncated.pt')
    
    test_loss = 0.0
    loss_func = nn.MSELoss()
    
    policy.eval()

    with torch.no_grad():
        for sample in testLoader:
            obs, labels = sample
            #print(obs.detach().numpy().shape)
            outputs = policy(obs.float())
            temp =  loss_func(outputs,labels.float())
            test_loss += temp.item()

    print(f'Loss on test set {test_loss/testLoader.__len__():.3f}')

    # plt.figure()
    # plt.plot([round(loss_,2) for loss_ in loss_array])
    # plt.show()

