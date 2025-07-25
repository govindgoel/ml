import os
import sys

import tqdm as tqdm
import wandb

import torch
from torch import nn
from torch_geometric.data import Batch, Data

# Add the 'scripts' directory to Python Path
scripts_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if scripts_path not in sys.path:
    sys.path.append(scripts_path)

from gnn.models.base_gnn import BaseGNN

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
                dropout: float = 0.1,
                use_dropout: bool = True,
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
        self.embed_dim = embed_dim
        self.ff_dim = ff_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.num_nodes = num_nodes

        # Might give it some sence of Graph structure
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
            pos = [data.pos[:,2,:] for data in datalist]
            pos = torch.stack(pos)
            x = torch.cat((x, pos), dim=2)  # Concatenate along the feature dimension

        x = x.to(self.dtype)

        # Transformer forward pass
        x = self.embed(x)
        x = self.transformer(x)
        x = self.output(x)
        
        return x.reshape(-1, 1)