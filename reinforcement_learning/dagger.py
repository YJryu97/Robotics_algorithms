import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

from utils import rollout, relabel_action

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def simulate_policy_dagger(env, policy, expert_paths, expert_policy=None, num_epochs=500, episode_length=100,
                            batch_size=32, num_dagger_iters=10, num_trajs_per_dagger=10):
    
    # TODO: Fill in your dagger implementation here. 
    
    # Hint: Loop through num_dagger_iters iterations, at each iteration train a policy on the current dataset.
    # Then rollout the policy, use relabel_action to relabel the actions along the trajectory with "expert_policy" and then add this to current dataset
    # Repeat this so the dataset grows with states drawn from the policy, and relabeled actions using the expert.
    
    # Optimizer code
    optimizer = optim.Adam(list(policy.parameters()))
    #losses = []
    returns = []
    loss_val = []
    trajs = expert_paths
    # Dagger iterations
    for dagger_itr in range(num_dagger_iters):
        idxs = np.array(range(len(trajs)))
        num_batches = len(idxs)*episode_length // batch_size
        losses = []
        data_s = torch.flatten(torch.tensor(np.array([trajs[t]['observations'] for t in idxs]), dtype=torch.float32), start_dim = 0, end_dim = 1)
        data_a = torch.flatten(torch.tensor(np.array([trajs[t]['actions'] for t in idxs]), dtype=torch.float32), start_dim = 0, end_dim = 1)
        policy.train()
        criterion = torch.nn.MSELoss()
        # Train the model with Adam
        for epoch in range(num_epochs):
            running_loss = 0.0
            for i in range(num_batches):
                optimizer.zero_grad()
                # TODO start: Fill in your behavior cloning implementation here
                # batch_idxs = idxs[i*batch_size:(i+1)*batch_size]
                """batch_s = torch.tensor(np.array([trajs[j]['observations'] for j in batch_idxs]), dtype=torch.float32).to(device)
                batch_a = torch.tensor(np.array([trajs[j]['actions'] for j in batch_idxs]), dtype=torch.float32).to(device)"""
                s_batch = data_s[i*batch_size:(i+1)*batch_size].to(device)
                a_batch = data_a[i*batch_size:(i+1)*batch_size].to(device)
                # print(batch_s.shape)
                a_hat = policy(s_batch)
                loss = criterion(a_hat, a_batch)
                # TODO end
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
            # print('[%d, %5d] loss: %.8f' %(epoch + 1, i + 1, running_loss))
            losses.append(loss.item())
        #plt.plot(losses)
        #plt.xlabel('Epoch')
        #plt.ylabel('Loss')
        #plt.title('Training Loss - Dagger Iteration {}'.format(dagger_itr))
        #plt.savefig('training_loss_plot_dagger_itr_{}.png'.format(dagger_itr))
            
        # Collecting more data for dagger
        trajs_recent = []
        for k in range(num_trajs_per_dagger):
            env.reset()
            # TODO start: Rollout the policy on the environment to collect more data, relabel them, add them into trajs_recent
            rollouts = rollout(env, policy, 'dagger')
            relabeling = relabel_action(rollouts, expert_policy)
            trajs_recent.append(relabeling)
            # TODO end

        trajs += trajs_recent
        mean_return = np.mean(np.array([traj['rewards'].sum() for traj in trajs_recent]))
        print("Average DAgger return is " + str(mean_return))
        returns.append(mean_return)
        
        # Plot the training loss at the end of each Dagger iteration
    
        
        
    print("Finished Training")
  