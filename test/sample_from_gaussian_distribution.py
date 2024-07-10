'''
Author: TX-Leo
Mail: tx.leo.wz@gmail.com
Date: 2024-05-26 22:24:49
Version: v1
File: 
Brief: 
'''
import numpy as np

# Mean and standard deviation values obtained from the low-level policy head
mean_values = np.array([0.2, -0.3, 0.5])  # Example mean values
std_values = np.array([0.1, 0.15, 0.2])  # Example standard deviation values

# Sample low-level actions from Gaussian distributions
low_level_actions = np.random.normal(mean_values, std_values)

# Clip the low-level actions between -1 and 1
low_level_actions = np.clip(low_level_actions, -1, 1)

print(low_level_actions)