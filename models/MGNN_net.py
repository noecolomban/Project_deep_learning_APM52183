import torch
import torch.nn as nn
import dgl.nn.pytorch as dglnn


class MGNNLayer(nn.Module):
    """
    Une Layer pour le MGNN
    H_new = A @ (H @ W_msg) + H @ W_self
    """
    def __init__(self, in_features, out_features):
        super().__init__()
        self.msg_proj = nn.Linear(in_features, out_features, bias=False)
        self.self_proj = nn.Linear(in_features, out_features, bias=True)
        
    def forward(self, h, A):
        # h: (bs, n_vertices, in_features)
        # A: (bs, n_vertices, n_vertices)
        msg = torch.bmm(A, self.msg_proj(h))
        self_loop = self.self_proj(h)
        return msg + self_loop


class MGNN(nn.Module):
    def __init__(self, original_features_num, num_blocks, in_features, out_features, depth_of_mlp, input_embed=False, **kwargs):
        
        super().__init__()
        
        # On reprend les mêmes attributs que BaseModel
        self.original_features_num = original_features_num
        self.num_blocks = num_blocks
        self.in_features = in_features
        self.out_features = out_features
        self.depth_of_mlp = depth_of_mlp #non utilisé ici
        self.embed = input_embed
        
        # Logique d'embedding copiée de BaseModel
        if self.embed:
            self.embedding = nn.Embedding(2, in_features)
            first_layer_features = self.in_features
        else:
            first_layer_features = self.original_features_num
            
        #Initialiser les noeuds à partir des arêtes
        self.edge_to_node = nn.Linear(first_layer_features, self.in_features)
        
        # Blocs de message passing
        self.layers = nn.ModuleList()
        for i in range(self.num_blocks):
            layer_out_feat = self.out_features if i == self.num_blocks - 1 else self.in_features
            self.layers.append(MGNNLayer(self.in_features, layer_out_feat))

    def forward(self, x):          
        #Matrice d'adjacence
        A = x.mean(dim=-1) 
        
        #Comme dans BaseModel
        if self.embed:
            x_feat = self.embedding(x[:,:,:,1].long())
        else:
            x_feat = x
            
        h = x_feat.sum(dim=2) 
        h = self.edge_to_node(h) 
        
        for i, layer in enumerate(self.layers):
            h = layer(h, A)
            
            if i < self.num_blocks - 1:
                h = torch.relu(h)
        
        return h

