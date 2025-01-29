
import numpy as np
import networkx as nx
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.common.maskable.utils import get_action_masks
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from main_env import MinFillEnv, minimum_fill_in_heuristic, minimum_degree_heuristic
import time
import torch_geometric as thg

import torch as th
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium import spaces
import numpy as np
import math
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, global_max_pool, BatchNorm
from stable_baselines3.common.callbacks import BaseCallback

from stable_baselines3.common.callbacks import BaseCallback

class SaveBestEliminationCallback(BaseCallback):
    def __init__(self, check_freq: int, verbose=0):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.best_fill_in = float("inf")
        self.best_order = None
        self.ep_rew_mean_values = []

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            num_envs = len(self.locals["infos"])
            ep_rews = []
            for info in self.locals["infos"]:
                # Track fill-in
                if "total_fill_in" in info:
                    current_fill = info["total_fill_in"]
                    if current_fill and current_fill < self.best_fill_in:
                        self.best_fill_in = current_fill
                        self.best_order = info["elimination_order"]
                        print(f"New best fill-in: {self.best_fill_in}")

                # Track episode rewards (updated for DummyVecEnv)
                if "episode" in info:
                    ep_rew_mean = info["episode"]["r"]
                    ep_rews.append(ep_rew_mean)
            if len(ep_rews):
                self.ep_rew_mean_values.append(sum(ep_rews)/len(ep_rews))

        return True


class GNNFeaturesExtractor(BaseFeaturesExtractor):
    """
    Graph Neural Network feature extractor for the minimum fill-in problem.
    """
    def __init__(
        self,
        observation_space: spaces.Box,
        features_dim: int = 1,
        n_hidden: int = 128,
        n_layers: int = 3
    ):
        super().__init__(observation_space, observation_space.shape[0])

        # Get the number of nodes from observation space
        self.N = observation_space.shape[0]  # N x (N+1) observation space

        # GNN layers
        self.conv_layers = nn.ModuleList()

        # First layer takes single feature per node
        self.conv_layers.append(
            nn.Linear(3, n_hidden)  # Initial node feature embedding
        )

        # Add graph convolution layers
        for _ in range(n_layers):
            self.conv_layers.append(
                nn.Linear(n_hidden * 2, n_hidden)  # Combines node features with neighbor features
            )

        # Final projection layer
        self.projection = nn.Sequential(
            nn.Linear(n_hidden, features_dim),
            nn.ReLU()
        )

    def message_passing(self, node_features, adj_matrix):
        """
        Performs message passing between nodes using the adjacency matrix
        """
        # Degree normalization
        degrees = adj_matrix.sum(dim=-1, keepdim=True) + 1
        neighbor_features = th.matmul(adj_matrix, node_features) / degrees
        return th.cat([node_features, neighbor_features], dim=-1)


    def forward(self, observations: th.Tensor) -> th.Tensor:
        device = observations.device

        # Split observation into adjacency matrix and node mask
        adj_matrix = observations[:, :, :-1]
        node_mask = observations[:, :, -1:]

        # Compute the initial node features from adj_matrix
        degrees = adj_matrix.sum(axis=-1).unsqueeze(-1)
        node_features_from_adj = degrees / self.N


        # Extract the diagonal: diag(result)
        diagonal = th.diagonal(th.matmul(th.matmul(adj_matrix, th.ones_like(adj_matrix).to(device) - adj_matrix), adj_matrix), dim1=-2, dim2=-1).unsqueeze(-1)  # Shape: (batch_size, num_nodes, 1)
        diagonal /= (self.N ** 2)


        # Concatenate node_features_from_adj and node_mask along the last dimension
        node_features = th.cat((node_features_from_adj, node_mask, diagonal), dim=-1)

        # Initial embedding
        node_features = self.conv_layers[0](node_features)

        # Apply graph convolution layers
        for conv in self.conv_layers[1:]:
            # Message passing
            combined_features = self.message_passing(node_features, adj_matrix)
            # Update node features
            node_features = F.relu(conv(combined_features))

        out = self.projection(node_features).squeeze(-1)
        return out



def make_env(graph, action_masking):
    def _init():
        env = MinFillEnv(graph, action_masking)
        return env
    return _init

def run_training(graph_location, policy_sizes, total_timesteps, learning_rate, output_file, parallel_envs, node_dim, ent_coef, action_masking):
    # Read the graph from the edge list
    G = nx.read_edgelist(graph_location, nodetype=int)
    G = nx.convert_node_labels_to_integers(G)


    # Define policy with custom feature extractor
    policy_kwargs = dict(
        features_extractor_class=GNNFeaturesExtractor,
        features_extractor_kwargs=dict(
            features_dim=1,
            n_hidden=args.node_dim,
            n_layers=2
        ),
        net_arch=dict(
            pi=policy_sizes,  # Policy network sizes after GNN
            vf=policy_sizes   # Value network sizes after GNN
        )
    )

    normalizer = len(G) * (len(G) - 1) / 2

    # Initialize the environment
    env = SubprocVecEnv([make_env(G, action_masking) for _ in range(parallel_envs)])
    env = VecMonitor(env)  # Wrap with VecMonitor

    # Initialize the model
    model = MaskablePPO(
        "MlpPolicy",
        env,
        n_steps=15*G.number_of_nodes(),           # Longer trajectories for complex graphs
        ent_coef=ent_coef,          # Slightly higher for exploration
        verbose=1,
        policy_kwargs=policy_kwargs,
        learning_rate=learning_rate,
        seed = 42,
        device="cuda"
    )

    # callback = MetricsPlotterCallback(plot_freq=1, verbose=0)
    checkpoint_callback = SaveBestEliminationCallback(check_freq=1)

    print(model.policy)

    # Train the model
    print("Starting training...")
    start_time = time.time()
    model.learn(total_timesteps=total_timesteps, callback=checkpoint_callback) #, callback=callback

    total_time = time.time() - start_time
    print(f"Training completed in {total_time:.2f} seconds.")

    obs = env.reset()
    done = False
    total_reward = 0
    total_fill_in = 0

    while not np.all(done):
        action_masks = get_action_masks(env)
        action, _ = model.predict(obs, deterministic=True, action_masks=action_masks)
        obs, reward, done, info = env.step(action)
        total_reward += reward
        total_fill_in += (-reward * normalizer)
    # # Calculate heuristic fill-ins
    mfinh = minimum_fill_in_heuristic(G)
    mdh = minimum_degree_heuristic(G)
    refill_fill_in = min(checkpoint_callback.best_fill_in, int(total_fill_in[0])) #minimum fill-in during training or from final policy
    # Write results to output file
    with open(output_file, "w") as f:
        f.write(f"{refill_fill_in}\n")
        f.write(f"{refill_fill_in / mdh:.6f}\n")
        f.write(f"{refill_fill_in / mfinh:.6f}\n")
        f.write(f"{total_time:.2f}\n")
        f.write(str(checkpoint_callback.best_order))
        f.write("\n")
        f.write(str(checkpoint_callback.ep_rew_mean_values))
    print(f"Results written to {output_file}")
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train MaskablePPO on a graph and compute fill-in heuristics.")
    parser.add_argument("graph_location", type=str, help="Path to the edge list file representing the graph.")
    parser.add_argument("--policy_sizes", nargs="*", type=int, default=[512, 256, 128], help="Policy sizes for the neural network.")
    parser.add_argument("--total_timesteps", type=int, default=150_000, help="Total timesteps for training.")
    parser.add_argument("--learning_rate", type=float, default=0.0005, help="Learning rate for training.")
    parser.add_argument("--output_file", type=str, default="results.txt", help="File to save the results.")
    parser.add_argument("--parallel_envs", type=int, default=8, help="How many parallel environments to run.")
    parser.add_argument("--node_dim", type=int, default=64, help="Dimension of the node.")
    parser.add_argument("--ent_coef", type=float, default=0.0, help="entropy coefficient for PPO.")
    parser.add_argument("--action_masking", type=int, default=0, help="0 or 1 on whether to do action masking.")
    args = parser.parse_args()
    # print(args.policy_sizes)
    run_training(
        graph_location=args.graph_location,
        policy_sizes=args.policy_sizes,
        total_timesteps=args.total_timesteps,
        learning_rate=args.learning_rate,
        output_file=args.output_file,
        parallel_envs=args.parallel_envs,
        node_dim=args.node_dim,
        ent_coef=args.ent_coef,
        action_masking=args.action_masking
    )
