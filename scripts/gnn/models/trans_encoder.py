import os
import sys

import tqdm as tqdm
import wandb

import torch
from torch import nn
from torch_geometric.data import Batch, Data
from torch_geometric.nn import GCNConv, GATConv, GraphConv

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

Usage Examples:
    # Basic transformer (no graph structure)
    python run_models.py --gnn_arch trans_encoder --unique_model_description "transformer_basic"
    
    # With positional features
    python run_models.py --gnn_arch trans_encoder --use_pos True --unique_model_description "transformer_pos"
    
    # With learned positional encoding
    python run_models.py --gnn_arch trans_encoder --use_pos True --pos_encoding True --unique_model_description "transformer_pos_encoding"
    
    # With graph convolution preprocessing
    python run_models.py --gnn_arch trans_encoder --use_graph_conv True --graph_conv_type gcn --unique_model_description "gcn_transformer"
    python run_models.py --gnn_arch trans_encoder --use_graph_conv True --graph_conv_type gat --unique_model_description "gat_transformer"
"""

class TransEncoder(BaseGNN):
    def __init__(self, 
                in_channels: int = 5,
                out_channels: int = 1,
                embed_dim: int = 128,
                ff_dim: int = 1024,
                num_heads: int = 4,
                num_layers: int = 5,
                num_nodes: int = 31635,
                use_pos: bool = False,
                pos_encoding: bool = False,
                dropout: float = 0.1,
                use_dropout: bool = True,
                predict_mode_stats: bool = False,
                dtype: torch.dtype = torch.float32,
                log_to_wandb: bool = True,
                use_graph_conv: bool = False,
                graph_conv_type: str = 'gcn'):
    
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
        self.pos_encoding = pos_encoding
        self.embed_dim = embed_dim
        self.ff_dim = ff_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.num_nodes = num_nodes
        self.use_graph_conv = use_graph_conv
        self.graph_conv_type = graph_conv_type

        # Give some sense of Graph structure
        if self.use_pos:
            self.in_channels += 2 # x and y for mid pos, might blow up!

        if self.log_to_wandb:
            wandb.config.update({'use_pos': use_pos,
                                'in_channels': self.in_channels,
                                'embed_dim': embed_dim,
                                'ff_dim': ff_dim,
                                'num_heads': num_heads,
                                'num_layers': num_layers,
                                'num_nodes': num_nodes},
                                allow_val_change=True)
        
        # Define the layers of the model
        self.define_layers()

        if self.use_pos and self.pos_encoding:
            assert num_nodes is not None, "num_nodes must be set for positional encoding"
            self.pos_embedding = nn.Embedding(num_nodes, self.embed_dim)

        if self.use_graph_conv:
            if graph_conv_type == 'gcn':
                self.graph_conv = GCNConv(self.embed_dim, self.embed_dim)
            elif graph_conv_type == 'gat':
                self.graph_conv = GATConv(self.embed_dim, self.embed_dim)
            elif graph_conv_type == 'graph':
                self.graph_conv = GraphConv(self.embed_dim, self.embed_dim)
            else:
                raise ValueError(f"Unknown graph_conv_type: {graph_conv_type}")

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


    def forward(self, data):
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

        if self.use_pos:
            if self.pos_encoding:
                # Get node indices for positional embedding
                node_indices = [data.pos[:,2,:].long().squeeze(-1) for data in datalist]
                node_indices = torch.stack(node_indices)
            else:
                pos = [data.pos[:,2,:] for data in datalist]
                pos = torch.stack(pos)
                x = torch.cat((x, pos), dim=2)

        x = x.to(self.dtype)

        x = self.embed(x)
        
        if self.use_pos and self.pos_encoding:
            pos_emb = self.pos_embedding(node_indices)  # shape: [batch, num_nodes, embed_dim]
            x = x + pos_emb  # Add positional embedding to embedded features

        # Apply graph convolution to incorporate graph structure
        if self.use_graph_conv:
            batch_size, num_nodes, features = x.shape
            x_reshaped = x.view(-1, features)  # [batch*num_nodes, features]
            # edge_index should be [2, num_edges]
            x_reshaped = self.graph_conv(x_reshaped, data.edge_index)  # TODO: add more layers if needed
            x = x_reshaped.view(batch_size, num_nodes, features)

        x = self.transformer(x)
        x = self.output(x)
        
        return x.reshape(-1, 1)