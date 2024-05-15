## ComBSAGE 

Basic DGL implmentation for the *Community-Based Sample and Aggregation* architecture described in "Graph Embedding for Mapping Interdisciplinary Research Networks" -- https://doi.org/10.1145/3543873.3587570.

The approach is also implemented in https://doi.org/10.48550/arXiv.2309.14984 in a research paper recommender systems framework. 

A basic implementation of the architecture is provided in the ComBSAGE class and another with the jumping knowledge connections is included in ComBSAGEJK. 

An example using the Cora dataset is included in *combsage_embedding_tutorial.ipynb*. This example trains the model in an autoencoder/unsupervised manner using a network reconstruction decoder. 

*requirements.yml* contains the python environment used in the experiments described in the paper. 

The training process is slow as it creates a Heterogenous graph to annotate edges in the network with their community memberships. Creation of this graph can be done upfront (see example) but managing samples from the heterogenous graph and original homogenous graphs creates a bottleneck for the training process. 
