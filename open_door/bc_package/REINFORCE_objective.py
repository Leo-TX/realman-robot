import torch
import clip
import numpy as np
from PIL import Image
import sys

from PolicyNetwork import PolicyNetworkHighLow, gen_dataloader, resnet_img2feature, N_SEQUENCES, TYPES_SEQUENCES, N_PARAMS
sys.path.append('../')
from arm import Arm as Env ## TODO: reset() and step()

# Hyperparameters for RL
N_EPISODES = 5
N_ROLLOUTS = 5
BATCH_SIZE = 16 # Should be equal sized batches for online and offline
LR = 1e-4 # for RL
ALPHA = 0.2 # the proportion coefficient of the loss_online and loss_offline
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Paths
train_data_dir = './data/train_data'
bc_model_save_path = "./checkpoints/PolicyNetworkBC.pt"
rl_model_save_path = "./checkpoints/PolicyNetworkRL.pt"

def CLIP(img_path, text_prompt, model_name = "ViT-B/32", if_p = True):
    ## Load CLIP model
    clip_model, preprocess = clip.load(model_name, device=DEVICE)
    ## get img_input
    img_input = preprocess(Image.open(img_path)).unsqueeze(0).to(DEVICE)
    ## get text_input
    text_input = clip.tokenize(text_prompt).to(DEVICE)
    with torch.no_grad():
        image_features = clip_model.encode_image(img_input)
        text_features = clip_model.encode_text(text_input)
        logits_per_image, logits_per_text = clip_model(img_input, text_input)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()
        if if_p:
            print("Label probs:", probs)  # prints: [[0.9927937  0.00421068]]
    return probs

def reward_annotation_CLIP(img_path = "CLIP.png", if_p = True):
    text_prompt = ["door that is closed", "door that is open"]
    ## using CLIP
    probs = CLIP(img_path, text_prompt)
    ## get reward
    reward = 1 if probs[0][1] > probs[0][0] else 0 # Reward is 1 if closer to 'open door' prompt, 0 otherwise
    if if_p:
        print("reward:", reward)  # prints: 1/0
        print(text_prompt[reward]) # door is open/closed
    return reward

def reward_annotation(mode='A'): # A: autonomously M: manually
    if mode == 'A':
        img_path = "CLIP.png"
        reward = reward_annotation_CLIP(img_path)
    elif mode == 'M':
        pass
    return reward

def calculate_discounted_rewards(rewards):
    """
    Calculates discounted rewards from a list of rewards: \sum_{t^{\prime}=t+1}^{T} \gamma^{t^{\prime}-t-1} r_{t^{\prime}}
    
    How to use the discounted rewards:
    for log_prob, Gt in zip(log_probs, discounted_rewards):
        loss.append(-log_prob * Gt)
    policy_network.optimizer.zero_grad()
    loss = torch.stack(loss).sum()
    loss.backward()
    policy_network.optimizer.step()
    """
    GAMMA = 0.99  # Discount factor
    discounted_rewards = []
    for t in range(len(rewards)):
        Gt = 0 
        pw = 0
        for r in rewards[t:]:
            Gt = Gt + GAMMA**pw * r # y^{t'-t-1} * r_{t'}
            pw = pw + 1
        discounted_rewards.append(Gt)
    discounted_rewards = torch.tensor(discounted_rewards)
    discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-9) # normalize discounted rewards
    return discounted_rewards
    
def get_action(img_path,model):
    encoded_feature = resnet_img2feature(img_path)
    encoded_feature = torch.tensor(encoded_feature, dtype=torch.float32).unsqueeze(0).to(DEVICE)  # Convert to tensor
    high_level_probs, high_level_sq, low_level_params = model(encoded_feature)
    return high_level_probs, high_level_sq, low_level_params
                           
def train_rl(env, model, optimizer, dataloader_offline):
    """
    Trains the policy network using REINFORCE algorithm.

    Args:
        env: Instance of the environment.
        model: Policy network model.
        optimizer: Optimizer for the model.
        dataloader_offline: Dataloader for offline demonstration data.
    """

    model.train()

    for episode in range(N_EPISODES):
        ## Online Data Collection
        episode_rewards = []
        episode_high_level_log_probs = []
        episode_low_level_params = []

        for rollout in range(N_ROLLOUTS):
            state = env.reset()
            done = False
            episode_reward = 0
            high_level_log_probs_list = []
            low_level_params_list = []

            while not done:
                # Predict actions from this state(image)
                high_level_probs, high_level_sq, low_level_params = get_action(state,model)
                
                # get the high_level_log_probs
                high_level_log_probs = torch.gather(torch.log(high_level_probs), 2, high_level_sq.unsqueeze(2)).squeeze(2) 

                high_level_log_probs_list.append(high_level_log_probs.detach())   # Store log probability
                low_level_params_list.append(low_level_params.detach()) 

                # Execute actions in the environment
                next_state, reward, done, _ = env.step(high_level_sq.squeeze().cpu().numpy(), low_level_params.squeeze().cpu().numpy())

                # Handle safety violations
                if reward == -1:  # Assuming -1 reward for safety violations
                    done = True

                episode_reward += reward
                state = next_state

            episode_rewards.append(episode_reward)
            episode_high_level_log_probs.append(torch.stack(high_level_log_probs_list))  # Stack tensors along a new dimension
            episode_low_level_params.append(torch.stack(low_level_params_list))  # Stack tensors along a new dimension

        # Calculate discounted_rewards
        # discounted_rewards = calculate_discounted_rewards_for_every_episode(episode_rewards)

        ## Update Policy (REINFORCE)
        for rollout in range(N_ROLLOUTS):
            ## Calculate loss_online(RL)
            R = episode_rewards[rollout] # Reward is fixed for the single-timestep rollout
            high_level_loss = 0
            low_level_loss = 0
            for step in range(len(episode_high_level_log_probs[rollout])): # len(episode_high_level_log_probs[rollout]) represents the number of primitive in the sequence
                # Online loss for high-level policy (sum log probs of the sequence * R)
                high_level_loss += -torch.sum(episode_high_level_log_probs[rollout][step]) * R # or episode_high_level_log_probs[rollout][step] * R 
                # Online loss for low-level policy (consider adding a baseline for variance reduction)
                low_level_loss += -torch.norm(episode_low_level_params[rollout][step]) * R  
            
            loss_online = high_level_loss + low_level_loss 
         
            ## Calculate loss_offline(BC)
            data, labels_sq, labels_params = next(iter(dataloader_offline))  # Sample a batch from the offline data
            data = data.to(DEVICE)
            labels_sq = labels_sq.to(DEVICE)
            labels_params = labels_params.to(DEVICE)
            high_level_output, _, low_level_output = model(data)  # output: BATCH_SIZE * N_SEQUENCES * TYPES_SEQUENCES
            loss_offline = 0
            for i in range(len(high_level_output)):
                # high level loss
                loss_offline += torch.nn.functional.cross_entropy(high_level_output[i], labels_sq[i])  # output[i]: N_SEQUENCES * TYPES_SEQUENCES; labels[i]: N_SEQUENCES
                # low level loss
                loss_offline += torch.nn.functional.mse_loss(low_level_output[i], labels_params[i])
            loss_offline = loss_offline / len(high_level_output)

            ## Calculate all loss
            episode_loss = loss_online + ALPHA * loss_offline

            ## update the model
            optimizer.zero_grad()
            episode_loss.backward()
            optimizer.step()

        print(f"episode: {episode+1}, Average Reward: {np.mean(episode_rewards)}")

    torch.save(model, rl_model_save_path)
    print("RL Training Complete. Model saved!")

if __name__ == "__main__":
    ## Initialize environment
    env = Env() 

    ## Load pre-trained policy network or initialize a new one
    model = torch.load(bc_model_save_path).to(DEVICE)  # Load your trained model

    ## Load offline demonstration data
    dataloader_offline = gen_dataloader(data_dir=train_data_dir, batch_size=BATCH_SIZE, num_workers=0, shuffle=True)

    ## Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    ## Train the agent
    train_rl(env, model, optimizer, dataloader_offline)