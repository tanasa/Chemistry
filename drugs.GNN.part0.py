#!/usr/bin/env python
# coding: utf-8

# In[1]:


# https://medium.com/data-science/building-a-graph-convolutional-network-for-molecular-property-prediction-978b0ae10ec4


# In[2]:


# https://medium.com/data-from-the-trenches/graph-neural-networks-graph-classification-part-iii-4fa0409eb9e1


# In[ ]:





# In[3]:


print('''

The input node features are nine-dimensional and edge features three-dimensional :

Atom Features (Node Features)

For every atom in the molecular graph, the following features are extracted:

atomic number
chirality
degree
formal charge
number of H (hydrogen atoms attached)
number radical e (radical electrons)
hybridization
is aromatic
is in ring

These become the node feature vector for each atom in the graph.

Bond Features (Edge Features)

For every bond between atoms, the following features are extracted:

bond type
stereo configuration
is conjugated

These become the edge feature vector for each bond in the graph.

''')


# In[4]:


print("GNN for Graph Classification: How Does It Work?")


# In[5]:


print('''

Learning From Multiple Graphs at Once :

As graphs tend to be small, it’s better to use batches of graphs instead of individual graphs before inputting them into a GNN.

In NLP or computer vision, this is typically done by rescaling or padding each element into a set of equally-sized shapes. 
For graphs, those approaches are not feasible. Instead, we can:

Stack adjacency matrices in a diagonal manner leading to a large graph with multiple isolated subgraphs.
Concatenate node features and the target.

''')


# In[6]:


print('''

The model learns to classify graphs using three main steps:

Embed nodes using several rounds of message passing.

Aggregate these node embeddings into a single graph embedding (called readout layer). 
In the code below, the average of node embeddings is used (global mean pool).

Train a classifier based on graph embeddings.

''')


# In[7]:


print('''

Model Architecture:

The model learns to classify graphs using three main steps:

Embed nodes using several rounds of message passing.

Aggregate these node embeddings into a single graph embedding (called readout layer). 
In the code below, the average of node embeddings is used (global mean pool).

Train a classifier based on graph embeddings

''')


# In[8]:


import torch 
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, GATv2Conv, TransformerConv
from torch_geometric.nn import global_mean_pool

class GNN(torch.nn.Module):
    def __init__(self, input_size, hidden_channels, conv, conv_params={}):
        super(GNN, self).__init__()
        torch.manual_seed(12345)
        
        self.conv1 = conv(
            input_size, hidden_channels, **conv_params)
        
        self.conv2 = conv(
            hidden_channels, hidden_channels, **conv_params)
        
        self.lin = Linear(hidden_channels, 2)
    
    def forward(self, x, edge_index, batch = None,  edge_col = None):
        
        # Node embedding 
        x = self.conv1(x, edge_index, edge_col)
        x = x.relu()
        x = self.conv2(x, edge_index, edge_col)
        
        # Readout layer
        batch = torch.zeros(data.x.shape[0],dtype=int) if batch is None else batch
        x = global_mean_pool(x, batch)
        
        # Final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)
    
        return x


# In[10]:


print('''

What About Edge Features?

The type of bond that links two atoms in a given molecule holds valuable information about the molecule 
such as its stability, its reactivity, the presence of some chemically organic functional groups, etc

But how can edge features be used when training the model?

If we take the example of GCN, it can easily be done by replacing the zeros and ones of the adjacency matrix with the edge weights.

In this context, each message-passing iteration through the GCN updates the hidden embedding of nodes based 
on the aggregated and now weighted information of their neighborhood.

https://www.youtube.com/watch?v=mdWQYYapvR8

''')


# In[ ]:





# In[11]:


print(''' What About Explainability for GNN?

For graphs, explicability is about three questions:

    Which nodes and features were relevant to making the prediction?
    How relevant were they?
    How relevant were the node and edge features of the graph?

''')


# In[12]:


print('''

Gradient or features-based methods: They rely on the gradients or hidden feature maps to approximate input importance. 
Gradients-based approaches compute the gradients of target prediction with respect to input features by back-propagation 
whereas feature-based methods map the hidden features to the input space via interpolation to measure importance scores. 
In this context, larger gradients or feature values mean higher importance.

Perturbation-based methods: They examine the variation in the model predictions with respect to different input perturbations. 
This is done by masking nodes or edges and observing the results for instance. 
Intuitively, predictions remain the same when important input information is kept.

Decomposition methods: They decompose prediction into the input space. 
Layer by layer the output is transferred back until the input layer is reached. 
The values then indicate which of the inputs had the highest importance on the outputs.

Surrogate: Train a simple and interpretable surrogate model to approximate 
the predictions of the model in the neighboring area of the input.

''')


# In[13]:


print(''' Graph Convolution : 

A Graph Neural Network (GNN) learns not only from a node itself, but also from its neighboring nodes.

One simple way to include neighbor information is to combine the features of a node with the features of its neighbors 
using an operation such as a sum.

This operation can be represented mathematically using:

a node feature matrix
and 
an adjacency matrix describing which nodes are connected.

Multiplying the adjacency matrix by the node feature matrix produces a new node representation 
where each node contains information aggregated from its neighbors.

In practice, the summed neighbor information is often normalized by the node degree (the number of neighbors).

This normalization is performed using the inverse of the diagonal degree matrix, 
effectively transforming the operation from a sum into a mean over neighboring nodes.

After aggregation, the result is multiplied by a learnable weight matrix, 
allowing the model to learn useful transformations during training.

This entire operation is called a graph convolution.

Graph convolutions allow information to propagate through the graph, 
enabling each node to gradually learn about its local graph structure.

''')


# In[14]:


print('''

The ConvolutionLayer essentially does three things — 

(1) calculation of the inverse diagonal degree matrix from the adjacency matrix, 
(2) multiplication of the four matrices (D⁻¹ANW), 
and 
(3) application of a non-linear activation function to the layer output. 

D−1 = inverse diagonal degree matrix
A = adjacency matrix
N = node feature matrix
W = learnable weight matrix
σ = activation function
N = updated node matrix

''')

# https://arxiv.org/abs/1609.02907


# In[15]:


print('''

Tools for the GNN Explanation Methods : 

Duvenaud et al. in [2]. managed to create a gradient-attribution method 
that highlights certain molecular substructures that trigger the predictions. 

GNNExplainer : https://arxiv.org/abs/1903.03894

https://github.com/KacperKubara/ml-cookbook/blob/master/drug_discovery_with_gnns/gnn_explainer_setup.md


''')


# In[16]:


# https://medium.com/@aishweta/graph-networks-traditional-methods-to-extract-features-from-the-of-graph-2e6cd86e5c10
# https://medium.com/@aishweta/graph-networks-traditional-methods-to-extract-features-from-the-of-graph-2e6cd86e5c10


# In[17]:


print('''

1. Node-Level Features : 

a. Node Degree: 

This metric is often used as initialization of algorithms to generate more complex graph-level features such as Weisfeiler-Lehman Kernel.

b. Node Centrality: 

Between Centrality:

Eigenvector centrality metric takes into account 2 aspects:

how important is the node u
how important are the neighbours of the node u

Importance of the node is normalized sum of the importances of the nodes that it links to.

c. Clustering Coefficient: 

A ratio of the number of edges between neighbours and the number of node’s neighbours (node degree). 
Values that are close to 1 means that all neighbours of the node u are connected to each other, 
whereas values close to 0 mean that there is barely any connection between node’s neighbours ..

Or using Graphlets: by counting #pre-specified subgraphs (graphlets)
 
2. Link-Level Features :

Approaches to calculate links:

1. Distance based features:
2. Local Neighborhood Overlap:v
3. Global neighborhood features:

Katz index: it counts the number of paths of all different lengths between given pair of nodes.

Question. How to compute #paths between two nodes?
Ans: use powers of the graph adjacency matrix.

We talked about 3 link level features:

1. Distance based feature — Uses shortest path length between two nodes but does not capture how neighborhood overlaps.
2. Local neighborhood feature — captures how many neighboring nodes are shared by two nodes. 
it becomes zero when no neighbor nodes are shared.
3. Global neighborhood feature — uses graph structure to score two nodes. katz index counts #paths of all lengths between two nodes.

''')


# In[18]:


# https://medium.com/data-science/feature-extraction-for-graphs-625f4c5fb8cd


# In[19]:


# https://medium.com/analytics-vidhya/an-intuitive-explanation-of-deepwalk-84177f7f2b72

print('''

DeepWalk algorithm : 

DeepWalk to learn node embeddings

a. DeepWalk : 

We generate a fixed number (k) of random walks starting at each node. 
The length of each walk (l) is pre-determined. 
Thus, when this stage is finished, we obtain k node sequences of length l. 
    
b. SkipGram :

SkipGram algorithm is a popular technique used to learn word embeddings. 
It was introduced by Mikolov et. al. in their well-known paper on word2vec[2].

Given a corpus and a window size, 
SkipGram aims to maximize the similarity of word embeddings’ of the words that occur in the same window. 

We can think of each walk produced in the previous step as a context or word window in a text. 

Thus, we can maximize the similarities of embeddings of nodes that occur on the same walks.

''')


# In[20]:


print('''

Graph Level Features

1.  Adjacency matrix. 

2.  Laplacian Matrix : = Degree Matrix - Adjacency Matrix

a. Weisfeiler-Lehman Kernel :

WL Kernel is an improvement of the Bag of Nodes approach where we iteratively aggregate information from the node’s neighbourhoods

b. Graphlet Kernels : 

Graphlet is defined as a small subgraph of size k ∈ {3,4,5}. 

c.Path-based Kernels

Path-based kernels create feature vectors by applying random walks or shortest paths over labelled nodes and edges of the graph.

''')


# In[21]:


print(''' 

Local Overlap Measures : metrics that quantify the similarity of the neighbourhood between two nodes.

Sorenson index
Salton index, 
Hub Promoted index, 
Jaccard index j

Global Overlap Measures : the information if certain nodes belong to the same community in the graph.

Instead of only focusing on two adjacent nodes, 
we look at nodes from a more distant neighbourhood and check if they belong to the same community in the graph.

Katz Index : it counts all possible paths between two specific nodes:

The Katz index is biased and will generate higher similarity scores for nodes with higher node degrees. 

To overcome this problem, LHN similarity metric was proposed that takes this bias into account.

''')


# In[ ]:





# In[ ]:




