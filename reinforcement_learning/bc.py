import torch
import torch.optim as optim
import numpy as np
from utils import rollout
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def simulate_policy_bc(env, policy, expert_data, num_epochs=500, episode_length=100, 
                       batch_size=32):
    
    # Hint: Just flatten your expert dataset and use standard pytorch supervised learning code to train the policy. 
    optimizer = optim.Adam(list(policy.parameters()))
    idxs = np.array(range(len(expert_data)))
    num_batches = len(idxs)*episode_length // batch_size
    #all_losses = []
    # loss_val = []
    data_s = torch.flatten(torch.tensor(np.array([expert_data[t]['observations'] for t in range(len(expert_data))]), dtype=torch.float32), start_dim=0, end_dim=1)
    data_a = torch.flatten(torch.tensor(np.array([expert_data[t]['actions'] for t in range(len(expert_data))]), dtype=torch.float32), start_dim=0, end_dim=1)
    policy.train()
    criterion = torch.nn.MSELoss()
    #for el in episode_length:
    #    num_batches = len(idxs)*el // batch_size
    losses = []
    for epoch in range(num_epochs): 
            ## TODO Students
        np.random.shuffle(idxs)
        running_loss = 0.0
        for i in range(num_batches):
            optimizer.zero_grad()
            # TODO start: Fill in your behavior cloning implementation here
            s_batch = data_s[i*batch_size:(i+1)*batch_size].to(device)
            a_batch = data_a[i*batch_size:(i+1)*batch_size].to(device)
            a_hat = policy(s_batch)
            loss = criterion(a_hat, a_batch)
            # TODO end
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        if epoch % 50 == 0:
            print('[%d] loss: %.8f' %
                (epoch, running_loss / 10.))
        losses.append(loss.item())
        #print("Finished Training for episode length:", el)    
        #all_losses.append(losses)
        #loss_val.append(running_loss/num_batches)

    return losses
    print("Finished Training")
    #plt.plot(losses)
    #plt.xlabel('Epoch')
    #plt.ylabel('Loss')
    #plt.title('Training Loss-Behavior Cloning')
    #plt.savefig('C:/study/Spring24/CSE571/assignment/cse571_24sp_hw3/resultstraining_loss_plot_bc.png')
    

    