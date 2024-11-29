import random

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Parameter
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm
import numpy as np

from sklearn.metrics import accuracy_score
import networkx as nx
from collections import deque
from dataset import *


class GCN_prop(MessagePassing):
    def __init__(self, L, **kwargs):
        super().__init__(aggr='add', **kwargs)
        self.L = L

    def forward(self, x, edge_index, edge_weight=None):
        edge_index, norm = gcn_norm(
            edge_index, edge_weight, num_nodes=x.size(0), dtype=x.dtype)

        reps = []
        for k in range(self.L):
            x = self.propagate(edge_index, x=x, norm=norm)
            reps.append(x)

        return reps

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def __repr__(self):
        return '{}(K={})'.format(self.__class__.__name__, self.L,)



class HeterGCL(nn.Module):
    def __init__(self, dataset, args):
        super().__init__()

        self.L = args.L
        self.dropout = args.dropout
        self.hidden_size = args.hidden_size
        self.output_size = args.output_size
        self.input_size = dataset.graph['node_feat'].shape[1]

        if args.Init=='random':
            # random
            bound = np.sqrt(3/(self.L))
            logits = np.random.uniform(-bound, bound, self.L)
            logits = logits/np.sum(np.abs(logits))
            self.logits = Parameter(torch.tensor(logits))
            print(f"init logits: {logits}")
        else:
            # fixed
            logits = np.array([1, float('-inf'), float('-inf')])
            self.logits = torch.tensor(logits)

        self.FFN = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(self.input_size, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size, self.output_size)
        )

        self.prop = GCN_prop(self.L)

    def forward(self, x):
        return self.FFN(x)

    @torch.no_grad()
    def get_embedding(self, x):
        self.FFN.eval()
        return self.FFN(x)

    def reset_parameters(self):
        torch.nn.init.zeros_(self.logits)
        bound = np.sqrt(3/(self.L))
        logits = np.random.uniform(-bound, bound, self.L)
        logits = logits/np.sum(np.abs(logits))
        for k in range(self.L):
            self.logits.data[k] = logits[k]
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.reset_parameters()

    def ANC_loss(self, h1, h2, gamma, temperature=1, bias=1e-8):
        z1 = F.normalize(h1, dim=-1, p=2)
        z2 = gamma*F.normalize(h2, dim=-1, p=2)

        numerator = torch.exp(
            torch.sum(z1 * z2, dim=1, keepdims=True) / temperature)

        E_1 = torch.matmul(z1, torch.transpose(z1, 1, 0))

        denominator = torch.sum(
            torch.exp(E_1 / temperature), dim=1, keepdims=True)

        return -torch.mean(torch.log(numerator / (denominator + bias) + bias))

    def ANC_total(self, h0, hs):
        loss = torch.tensor(0, dtype=torch.float32).cuda()
        gamma = F.softmax(self.logits, dim=0)
        # for i in range(len(hs)):
        for i in range(len(gamma)):

            loss += self.ANC_loss(h0, hs[i], gamma[i])

        return loss

class GB_prop(MessagePassing):
    def __init__(self, init_GB_num,labels):
        super(GB_prop, self).__init__(aggr='add')
        self.init_GB_num = init_GB_num
        self.labels = labels


    def calculate_homogeneity(self, graph, labels):
        """
        Calculate the homogeneity of a graph based on edge label agreement.
        Parameters:
        - graph: NetworkX graph object.
        - labels: List or array where indices are node identifiers and values are node labels.
        Returns:
        - homogeneity: A single homogeneity score for the graph.
        """

        labels = labels.numpy().tolist() if isinstance(labels, torch.Tensor) else labels

        label_homogeneity = {}
        total_edges = 0
        for u, v in graph.edges():
            if u < len(labels) and v < len(labels):
                label_u = labels[u]
                label_v = labels[v]
                total_edges += 1

                if label_u == label_v:
                    if label_u not in label_homogeneity:
                        label_homogeneity[label_u] = 0
                    label_homogeneity[label_u] += 1

        if not label_homogeneity:
            return 0

        max_homogeneity = max(label_homogeneity.values())

        homogeneity = max_homogeneity / total_edges if total_edges > 0 else 0

        return homogeneity

    def split_ball(self, graph, split_GB_list, min_size=5, min_improvement=0.001):
        """
        The logic of splitting granular-balls, recursively splitting based on homogeneity and the structure of the graph.

        Parameters:
        - graph: NetworkX subgraph, granular-ball to be split.
        - split_GB_list: A list used to store the results of the final split.
        - min_size: The minimum number of nodes at which the granular-ball stops splitting.
        - min_improvement: Minimum homogeneity boost from splitting.
        """
        # If the pellet size is less than or equal to 1 or the min_size, stop splitting
        if len(graph) <= min_size:
            split_GB_list.append(graph)
            return

        # Select the central node based on the node degree
        degree_dict = dict(graph.degree())
        sorted_nodes = sorted(degree_dict, key=degree_dict.get, reverse=True)
        center_nodes = sorted_nodes[:2]

        # Assign nodes to multiple centers
        center_nodes_dict = self.assign_nodes_to_multiple_centers(graph, center_nodes)
        clusters = [cluster for cluster in center_nodes_dict.values()]
        cluster_a, cluster_b = clusters[0], clusters[1]

        # Generate two subgraphs
        graph_a = graph.subgraph(cluster_a)
        graph_b = graph.subgraph(cluster_b)

        # Check whether the subgraph is empty
        if len(graph_a.edges()) == 0 or len(graph_b.edges()) == 0:
            split_GB_list.append(graph)
            return
        
        homogeneity_before = self.calculate_homogeneity(graph, self.labels)
        homogeneity_a = self.calculate_homogeneity(graph_a, self.labels)
        homogeneity_b = self.calculate_homogeneity(graph_b, self.labels)
        homogeneity_after = (homogeneity_a + homogeneity_b) / 2.0

        # To determine whether to continue the split
        improvement = homogeneity_after - homogeneity_before

        if homogeneity_before < max(homogeneity_b,homogeneity_a) :

            # If splitting results in a significant homogeneity improvement, continue splitting the subgraph
            self.split_ball(graph_a, split_GB_list, min_size, min_improvement)
            self.split_ball(graph_b, split_GB_list, min_size, min_improvement)
        else:
            # If the split does not bring significant improvement, stop the split
            split_GB_list.append(graph)
        ## Output the list of split pellets
        # print(f"Split GB list (Size: {len(split_GB_list)}): {[len(g) for g in split_GB_list]}")

    def split_graph(self, graph, init_GB_num):
        sqrt_n = init_GB_num

        # Gets the node with the largest degree
        max_degree_node = max(graph.degree, key=lambda x: x[1])[0]

        # Create a queue for BFS
        queue = deque([max_degree_node])
        visited = set()
        subgraph_nodes = set([max_degree_node]) 
        visited.add(max_degree_node)

        while queue:
            # Gets the number of nodes in the current layer
            current_level_size = len(queue)
            current_layer_nodes = []

            # Traverse all nodes of the current layer
            for _ in range(current_level_size):
                node = queue.popleft()
                for neighbor in graph.neighbors(node):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        current_layer_nodes.append(neighbor)

            # After traversing the first layer, check whether the number of subgraph nodes exceeds sqrt_n
            if len(subgraph_nodes) + len(current_layer_nodes) > sqrt_n:
                break

            # Adds the current tier node to the subgraph and queue
            subgraph_nodes.update(current_layer_nodes)
            queue.extend(current_layer_nodes)

        # Removes the selected node from the original image
        remaining_graph = graph.copy()
        remaining_graph.remove_nodes_from(subgraph_nodes)

        # Returns the maximum degree node and the remaining graph
        return max_degree_node, remaining_graph

    def assign_nodes_to_multiple_centers(self, G, centers):
        # Initializes a dictionary to store a collection of nodes for each central node
        center_nodes_dict = {center: set() for center in centers}

        # Initialize queues and access records for each central node
        queues = {center: deque([center]) for center in centers}
        visited = {center: center for center in centers}

        # Perform a multi-source Breadth-first search (BFS)
        while any(queues.values()):
            for center in centers:
                if queues[center]:
                    current_node = queues[center].popleft()
                    center_nodes_dict[center].add(current_node)

                    # Traverses all neighbors of the current node
                    for neighbor in G.neighbors(current_node):
                        if neighbor not in visited:
                            visited[neighbor] = center
                            queues[center].append(neighbor)

        # Returns a collection of nodes for each central node
        return center_nodes_dict

    def init_GB_graph(self, graph, init_GB_num):
        remaining_graph = graph
        center_nodes = []
        for i in range(init_GB_num):
            max_degree_node, remaining_graph = self.split_graph(remaining_graph, init_GB_num)
            center_nodes.append(max_degree_node)
        center_nodes_dict = self.assign_nodes_to_multiple_centers(graph, center_nodes)
        init_GB_list = [nx.subgraph(graph, cluster) for cluster in center_nodes_dict.values()]
        # print("Init GB list:", init_GB_list)
        return init_GB_list

    def get_GB_graph(self, graph, init_methods="two"):
        if init_methods == "two":
            init_GB_num = 2
        else:
            import math
            init_GB_num = math.isqrt(len(graph))

        init_GB_list = self.init_GB_graph(graph, init_GB_num)
        GB_list = []
        for init_GB in init_GB_list:

            split_GB_list = []
            self.split_ball(init_GB, split_GB_list)
            GB_list.extend(split_GB_list)
        # for GB in GB_list:
        #     print(GB.nodes())
        # print(f"Total number of generated balls: {len(GB_list)}")
        GB_graph = nx.Graph()
        if len(GB_list) == 1:
            return graph

        # Add edges for each pair of granular-balls
        for i in range(len(GB_list)):
            for j in range(i + 1, len(GB_list)):
                count = sum(graph.has_edge(a, b) for a in GB_list[i].nodes() for b in GB_list[j].nodes())
                if count > 0:
                    GB_graph.add_edge(i, j, weight=count)

        return GB_graph, GB_list

    def forward(self, x1, edge_index):
        # Convert PyTorch tensor to NetworkX graph
        graph = self.tensor_to_graph(x1, edge_index) # Convert the input node feature x1 and edge index edge_index into a NetworkX graph

        GB_graph, GB_list = self.get_GB_graph(graph, init_methods="two")

        # Obtain the connected components (clusters) of the coarse graph
        clusters = self.get_clusters(GB_list)

        # Convert the coarsened graph back to PyTorch tensor
        GB_x, GB_edge_index = self.graph_to_tensor(GB_graph, x1, clusters)

        # Returns the node features and edge indexes of the coarse graph
        return GB_x, GB_edge_index

    def tensor_to_graph(self, x, edge_index):
        # Create a NetworkX diagram object
        graph = nx.Graph()
        # Add nodes and edges
        num_nodes = x.size(0)
        graph.add_nodes_from(range(num_nodes))
        graph.add_edges_from(edge_index.t().tolist())
        return graph

    def graph_to_tensor(self, GB_graph, x, clusters):
        GB_x = []
        GB_edge_index = []

        # Traverse each cluster (connected component)
        for i, cluster in enumerate(clusters):
            # Gets the nodes in the cluster
            cluster_nodes = list(cluster)

            # Compute the node feature average of the cluster
            cluster_x = x[torch.tensor(cluster_nodes, dtype=torch.long)]
            cluster_mean = cluster_x.mean(dim=0)
            GB_x.append(cluster_mean)

            # Record connections between clusters
            for j, other_cluster in enumerate(clusters):
                if i < j:  # Avoid adding edges repeatedly
                    # Check for edges
                    if any(GB_graph.has_edge(node, other_node) for node in cluster_nodes for other_node in
                           list(other_cluster)):
                        # Add edge index
                        GB_edge_index.append([i, j])

        GB_x = torch.stack(GB_x)
        GB_edge_index = torch.tensor(GB_edge_index, dtype=torch.long).t().contiguous()

        return GB_x, GB_edge_index

    # Each granular-ball (subgraph) in GB_list acts as a separate cluster
    def get_clusters(self, GB_list):
        return [list(GB.nodes()) for GB in GB_list]