import torch
high_level_probs = torch.tensor([[0.0975, 0.6180, 0.0981, 0.1025, 0.0839],[0.2485, 0.0963, 0.3108, 0.2084, 0.1360],[0.0631, 0.1095, 0.1124, 0.1003, 0.6146]])
# Sample high-level action from the probability distribution 
high_level_action_dist = torch.distributions.Categorical(high_level_probs)
print(f'high_level_action_dist:{high_level_action_dist}')

high_level_action = high_level_action_dist.sample()
print(f'high_level_action:{high_level_action}')

high_level_log_prob = high_level_action_dist.log_prob(high_level_action) 
print(f'high_level_log_prob:{high_level_log_prob}')

_, predicted_sq = torch.max(high_level_probs)
print(f'predicted_sq:{predicted_sq}')
