# About

This is code for the paper "ReFill: Reinforcement Learning for Fill-In Minimization"

#Libraries needed:
1) stable-baselines3
2) gymnasium
3) numpy
4) sb3-contrib
5) torch_geometric

# How to run:

You can run the code using ``python main.py datasets/grid.n8.graph --output_file results/grid.n8.graph --policy_sizes 16 16 --total_timesteps 500_000 --learning_rate 0.00005 --parallel_envs 5 --node_dim 16 --ent_coef 0.0001 --action_masking 0`` and change the hyperparameters as you need. 
