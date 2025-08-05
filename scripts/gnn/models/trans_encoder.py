import os
import sys

import tqdm as tqdm
import wandb

import torch
from torch import nn
from torch_geometric.data import Batch, Data
from torch_geometric.nn import GCNConv, GATConv, GraphConv, TransformerConv, GraphNorm

# Add the 'scripts' directory to Python Path
scripts_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if scripts_path not in sys.path:
    sys.path.append(scripts_path)

from gnn.models.base_gnn import BaseGNN


"""
Transformer Encoder for Graph Neural Networks

This model implements a transformer encoder that can optionally incorporate graph structure
through various mechanisms. The base transformer operates on node features without explicit
graph awareness, but can be enhanced with:

1. Positional Information: Add node positions as additional features
2. Positional Encoding: Learn embeddings for node positions (like in transformers)
3. Graph Convolution Layers: Pre-process features using graph structure before transformer
"""

class TransEncoder(BaseGNN):
    def __init__(self, 
                in_channels: int = 5,
                out_channels: int = 1,
                embed_dim: int = 128,
                ff_dim: int = 512,
                num_heads: int = 4,
                num_layers: int = 5,
                num_nodes: int = 31635,
                use_pos: bool = False,
                use_pos_encoding: bool = False,
                dropout: float = 0.1,
                use_dropout: bool = False,
                use_graph_conv: bool = False,
                use_graph_norm: bool = True,
                num_graph_conv_layers: int = 2,
                graph_conv_type: str = 'trans_conv',
                predict_mode_stats: bool = False,
                dtype: torch.dtype = torch.float32,
                log_to_wandb: bool = False):
    
        # Call parent class constructor
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            dropout=dropout,
            use_dropout=use_dropout,
            predict_mode_stats=predict_mode_stats,
            dtype=dtype,
            log_to_wandb=log_to_wandb)
        
        # Model specific parameters
        self.use_pos = use_pos
        self.use_pos_encoding = use_pos_encoding
        self.embed_dim = embed_dim
        self.ff_dim = ff_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.num_nodes = num_nodes
        self.use_graph_conv = use_graph_conv
        self.use_graph_norm = use_graph_norm
        self.graph_conv_type = graph_conv_type
        self.num_graph_conv_layers = num_graph_conv_layers

        # Give some sense of Graph structure
        if self.use_pos:
            self.in_channels += 2 # x and y for mid pos, might blow up!

        if self.log_to_wandb:
            wandb.config.update({'use_pos': use_pos,
                                'use_pos_encoding': use_pos_encoding,
                                'in_channels': self.in_channels,
                                'embed_dim': embed_dim,
                                'ff_dim': ff_dim,
                                'num_heads': num_heads,
                                'num_layers': num_layers,
                                'num_nodes': num_nodes,
                                'use_graph_conv': use_graph_conv,
                                'use_graph_norm': use_graph_norm,
                                'graph_conv_type': graph_conv_type,
                                'num_graph_conv_layers': num_graph_conv_layers},
                                allow_val_change=True)
        
        # Define the layers of the model
        self.define_layers()

    def define_layers(self):
        
        # Step 1: Embed input into higher dimension
        self.embed = nn.Linear(self.in_channels, self.embed_dim)

        # Step 2: Transformer Encoder (batch_first = True: input = (B, S, D))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=self.num_heads,
            dim_feedforward=self.ff_dim,
            dropout=self.dropout if self.use_dropout else 0.0,
            batch_first=True)
        
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)

        # Step 3: Project to scalar output per node
        self.output = nn.Linear(self.embed_dim, 1)

        # Optional: Positional Encoding
        if self.use_pos_encoding:
            assert self.num_nodes is not None, "num_nodes must be set for positional encoding"
            self.pos_embedding = nn.Embedding(self.num_nodes, self.embed_dim)
            self.node_indices = torch.arange(self.num_nodes).long().to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))  # Node indices for positional embedding

        # Optional: Graph Convolution Layers
        if self.use_graph_conv:
            for i in range(self.num_graph_conv_layers):

                in_channels = self.in_channels if i == 0 else self.embed_dim
                
                if self.graph_conv_type == 'gcn':
                    conv = GCNConv(in_channels, self.embed_dim)
                elif self.graph_conv_type == 'gat':
                    conv = GATConv(in_channels, self.embed_dim)
                elif self.graph_conv_type == 'graph':
                    conv = GraphConv(in_channels, self.embed_dim)
                elif self.graph_conv_type == 'trans_conv':
                    conv = TransformerConv(in_channels, self.embed_dim)

                setattr(self, f'conv_{i + 1}', conv)

                if self.use_graph_norm:
                    graph_norm = GraphNorm(self.embed_dim if i > 0 else self.in_channels)
                    setattr(self, f'graph_norm_{i + 1}', graph_norm)

        # Optional: Dropout Layer for Convolutional layers
        if self.use_dropout:
            self.dropout_layer = nn.Dropout(self.dropout)


    def forward(self, data):

        if self.use_graph_conv:

            # Unpack data
            x = data.x
            edge_index = data.edge_index

            if self.use_pos:
                pos = data.pos[:, 2, :] # Middle position
                x = torch.cat((x, pos), dim=1)  # Concatenate along the feature dimension

            x = x.to(self.dtype)

            for i in range(self.num_graph_conv_layers):

                if self.use_graph_norm:
                    graph_norm = getattr(self, f'graph_norm_{i + 1}')
                    x = graph_norm(x)

                conv = getattr(self, f'conv_{i + 1}')
                x = conv(x, edge_index)
                x = nn.functional.relu(x)
                
                if self.use_dropout:
                    x = self.dropout_layer(x) 

            data.x = x
        
        # Unpack data
        if isinstance(data, Batch):
            datalist = data.to_data_list()
        elif isinstance(data, Data):
            datalist = [data]
        else:
            raise ValueError("Input data must be a Batch or Data object")

        # Reshape x to (batch_size, num_nodes, in_channels)
        x = [data.x for data in datalist]
        x = torch.stack(x)

        if self.use_pos and not self.use_graph_conv:
            pos = [data.pos[:,2,:] for data in datalist]
            pos = torch.stack(pos)
            x = torch.cat((x, pos), dim=2)

        if not self.use_graph_conv:
            x = x.to(self.dtype)
            x = self.embed(x)
        
        if self.use_pos_encoding:
            pos_emb = self.pos_embedding(self.node_indices)  # shape: [num_nodes, embed_dim]
            pos_emb = pos_emb.unsqueeze(0).expand(x.size(0), -1, -1) # Expand to match batch size
            x = x + pos_emb  # Add positional embedding to embedded features
        
        x = self.transformer(x)
        x = self.output(x)
        
        return x.reshape(-1, 1)