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


"""
class MGNN(nn.Module):
    def __init__(self, **kwargs):
        super(MGNN, self).__init__()
        
        # Dimensions (original_features_num vaut 2 selon vos logs)
        self.in_feat = kwargs.get('original_features_num', 2)
        self.h_feat = kwargs.get('in_features', 64)
        self.out_feat = kwargs.get('out_features', 64)
        
        # Couche pour initialiser les noeuds à partir des arêtes
        self.edge_to_node = nn.Linear(self.in_feat, self.h_feat)
        
        # Poids pour la couche 1
        self.layer1_msg = nn.Linear(self.h_feat, self.h_feat, bias=False)
        self.layer1_self = nn.Linear(self.h_feat, self.h_feat)
        
        # Poids pour la couche 2
        self.layer2_msg = nn.Linear(self.h_feat, self.out_feat, bias=False)
        self.layer2_self = nn.Linear(self.h_feat, self.out_feat)

    def forward(self, x):
        # 'x' est un tenseur de dimension (Batch, N, N, in_feat)
        # in_feat vaut 2 : ça contient généralement l'adjacence et un masque.
        
        # 1. Création des features initiales des noeuds (H)
        # On somme les arêtes connectées à chaque noeud : (Batch, N, in_feat)
        h = x.sum(dim=2) 
        h = self.edge_to_node(h) # Projection vers hidden_features : (Batch, N, h_feat)
        
        # 2. Extraction de la matrice d'adjacence (A)
        # On fait la moyenne sur la dernière dimension pour avoir un poids de connexion
        # A devient un tenseur (Batch, N, N)
        A = x.mean(dim=-1) 
        
        # --- COUCHE 1 : Message Passing ---
        # Messages des voisins : A * (H * W_msg) -> (Batch, N, h_feat)
        msg1 = torch.bmm(A, self.layer1_msg(h))
        # Mise à jour du noeud lui-même : H * W_self
        self1 = self.layer1_self(h)
        # Activation
        h = torch.relu(msg1 + self1)
        
        # --- COUCHE 2 : Message Passing ---
        msg2 = torch.bmm(A, self.layer2_msg(h))
        self2 = self.layer2_self(h)
        h = msg2 + self2 # Pas de ReLU sur la dernière couche en général
        
        # On renvoie les embeddings des noeuds (Batch, N, out_feat)
        return h
"""