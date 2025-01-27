import numpy as np
import networkx as nx
from gymnasium import spaces, Env
import networkx as nx 

def minimum_fill_in_heuristic(graph):
    G = graph.copy()
    total_fill_in = 0

    while G.number_of_nodes() > 0:
        # Compute fill-in cost for each node
        fill_in_costs = {}
        for node in G.nodes():
            neighbors = list(G.neighbors(node))
            missing_edges = 0
            for i in range(len(neighbors)):
                for j in range(i + 1, len(neighbors)):
                    if not G.has_edge(neighbors[i], neighbors[j]):
                        missing_edges += 1
            fill_in_costs[node] = missing_edges
        
        # Select the node with the minimum fill-in cost
        min_fill_in_node = min(fill_in_costs, key=fill_in_costs.get)
        total_fill_in += fill_in_costs[min_fill_in_node]
        
        # Make the neighbors of the selected node a clique
        neighbors = list(G.neighbors(min_fill_in_node))
        for i in range(len(neighbors)):
            for j in range(i + 1, len(neighbors)):
                if not G.has_edge(neighbors[i], neighbors[j]):
                    G.add_edge(neighbors[i], neighbors[j])
        
        # Remove the selected node
        G.remove_node(min_fill_in_node)

    return total_fill_in


def minimum_degree_heuristic(graph):
    G = graph.copy()  # Work on a copy of the graph to avoid modifying the original
    total_fill_in = 0

    while G.number_of_nodes() > 0:
        # Find the node with the minimum degree
        min_degree_node = min(G.nodes, key=G.degree)

        # Get the neighbors of the node to be deleted
        neighbors = list(G.neighbors(min_degree_node))

        # Compute the fill-in cost: count missing edges among neighbors
        missing_edges = 0
        for i, v in enumerate(neighbors):
            for w in neighbors[i + 1:]:
                if not G.has_edge(v, w):
                    missing_edges += 1
                    G.add_edge(v, w)

        # Add the fill-in cost to the total
        total_fill_in += missing_edges

        # Remove the selected node from the graph
        G.remove_node(min_degree_node)

    return total_fill_in


class MinFillEnv(Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self, G, action_masking=1):
        super(MinFillEnv, self).__init__()
        self.action_masking = action_masking
        self.G = G
        self.N = len(G)
        self.den = self.N * (self.N - 1) / 2
        
        print("Min degree heuristic is ", minimum_degree_heuristic(self.G))
        print("Min fill-in heuristic is ", minimum_fill_in_heuristic(self.G))

        self.elimination_order = []
        self.total_fill_in = 0

        
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.N, self.N + 1), dtype=np.int32)
        self.action_space = spaces.Discrete(self.N)
        self.reset()

    def reset(self, seed=None, options=None):
        self.elimination_order = []
        self.total_fill_in = 0

        self.adjacency_matrix = nx.adjacency_matrix(self.G, dtype=np.int64).toarray()
        self.deleted_mask = np.zeros(self.N, dtype=np.int32)
        return self._get_observation(), {}

    def step(self, action):
        selected_node = action
        undeleted = self.deleted_mask == 0
        neighbors = np.where((self.adjacency_matrix[selected_node] == 1) & undeleted)[0]
        missing_edges = 0
        
        if neighbors.size >= 2:
            # Extract the submatrix for the neighbors
            submatrix = self.adjacency_matrix[np.ix_(neighbors, neighbors)]
            # Count missing edges in the upper triangle
            triu_indices = np.triu_indices(len(neighbors), 1)
            upper_triangle = submatrix[triu_indices]
            missing_edges = np.sum(upper_triangle == 0)
            # Add missing edges efficiently
            self.adjacency_matrix[neighbors[:, None], neighbors] = 1

        self.adjacency_matrix[selected_node, :] = 0
        self.adjacency_matrix[:, selected_node] = 0
        
        self.elimination_order.append(selected_node)
        self.total_fill_in += missing_edges
        
        reward = -missing_edges / self.den
        self.deleted_mask[selected_node] = 1
        done = np.sum(self.deleted_mask) == self.N

        info = {"elimination_order": None, "total_fill_in": None}
        if done:
            info = {
                "elimination_order": self.elimination_order.copy(),
                "total_fill_in": self.total_fill_in
            }

        return self._get_observation(), reward, done, False, info

    def _get_observation(self):
        mask_col = self.deleted_mask[:, None]
        return np.hstack((self.adjacency_matrix, mask_col)).astype(np.int32)

    def action_masks(self):
        if self.action_masking==0:
            return self.deleted_mask == 0
            
        undeleted_mask = self.deleted_mask == 0

        # Compute node degrees for undeleted nodes
        degrees = np.sum(self.adjacency_matrix * undeleted_mask[:, None], axis=1)
        min_degree = np.min(degrees[undeleted_mask])  # Minimum degree among undeleted nodes
        min_degree_mask = (degrees == min_degree) & undeleted_mask
    
        # Identify all undeleted nodes
        nodes = np.where(undeleted_mask)[0]
    
        # Compute fill-in costs using list comprehension for speed
        fill_in_costs = np.array([
            0 if (neighbors := np.where((self.adjacency_matrix[node] == 1) & undeleted_mask)[0]).size < 2
            else len(neighbors) * (len(neighbors) - 1) // 2 - np.sum(np.triu(self.adjacency_matrix[np.ix_(neighbors, neighbors)], 1))
            for node in nodes
        ], dtype=np.float32)
    
        # Initialize fill_in_costs with infinity and assign computed costs
        full_fill_in_costs = np.full(self.N, np.inf, dtype=np.float32)
        full_fill_in_costs[nodes] = fill_in_costs
    
        # Determine the minimum fill-in cost among undeleted nodes
        min_fill_in = np.min(full_fill_in_costs[undeleted_mask])  # Minimum fill-in among undeleted nodes
        min_fill_in_mask = (full_fill_in_costs == min_fill_in) & undeleted_mask
        
        # Combine conditions: undeleted AND (min degree OR min fill-in)
        action_mask = undeleted_mask & (min_degree_mask | min_fill_in_mask)
        return action_mask