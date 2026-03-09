import torch
import torch.nn as nn
import dgl.nn.pytorch as dglnn


class MGNNLayer(nn.Module):
    """
    Couche unitaire de Message Passing pour le MGNN.
    Applique la formule : H_new = A @ (H @ W_msg) + H @ W_self
    """
    def __init__(self, in_features, out_features):
        super().__init__()
        self.Wmsg = nn.Linear(in_features, out_features, bias=False)
        self.Wself = nn.Linear(in_features, out_features, bias=True)
        
    def forward(self, h, A):
        # h: (bs, n_vertices, in_features)
        # A: (bs, n_vertices, n_vertices)
        msg = torch.bmm(A, self.Wmsg(h))
        self_loop = self.Wself(h)
        return msg + self_loop


class MGNN(nn.Module):
    def __init__(self, original_features_num, num_blocks, in_features, out_features, depth_of_mlp, input_embed=False, **kwargs):
        """
        Prend un batch de graphes (bs, n_vertices, n_vertices, original_features_num)
        et renvoie un batch d'embeddings de noeuds (bs, n_vertices, out_features)
        """
        super().__init__()
        
        # On reprend les mêmes attributs que BaseModel
        self.original_features_num = original_features_num
        self.num_blocks = num_blocks
        self.in_features = in_features
        self.out_features = out_features
        self.depth_of_mlp = depth_of_mlp # Gardé pour la signature, même si non utilisé ici
        self.embed = input_embed
        
        # Logique d'embedding copiée de BaseModel
        if self.embed:
            self.embedding = nn.Embedding(2, in_features)
            first_layer_features = self.in_features
        else:
            first_layer_features = self.original_features_num
            
        # Couche pour initialiser les noeuds à partir des arêtes
        self.edge_to_node = nn.Linear(first_layer_features, self.in_features)
        
        # Création des blocs de Message Passing
        self.layers = nn.ModuleList()
        for i in range(self.num_blocks):
            layer_out_feat = self.out_features if i == self.num_blocks - 1 else self.in_features
            self.layers.append(MGNNLayer(self.in_features, layer_out_feat))

    def forward(self, x):
        # Validation de la taille d'entrée (comme dans BaseModel)
        if x.size(3) != self.original_features_num:
            print("expected input feature {} and got {}".format(self.original_features_num, x.shape[3]))
            return
            
        # 1. Extraction de la matrice d'adjacence (A)
        # On le fait avant l'embedding éventuel pour garder le sens de la matrice
        A = x.mean(dim=-1) 
        
        # 2. Gestion de l'input_embed (comme dans BaseModel)
        if self.embed:
            # On passe la deuxième feature (index 1) dans l'Embedding
            x_feat = self.embedding(x[:,:,:,1].long())
        else:
            x_feat = x
            
        # 3. Création des features initiales des noeuds (H)
        h = x_feat.sum(dim=2) 
        h = self.edge_to_node(h) 
        
        # 4. Boucle de Message Passing
        for i, layer in enumerate(self.layers):
            h = layer(h, A)
            
            if i < self.num_blocks - 1:
                h = torch.relu(h)
        
        return h

