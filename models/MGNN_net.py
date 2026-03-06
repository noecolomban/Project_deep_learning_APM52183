import torch
import torch.nn as nn
import dgl.nn.pytorch as dglnn

class MGNNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(MGNNLayer, self).__init__()
        self.W1 = nn.Linear(in_features, out_features, bias=False)
        self.W2 = nn.Linear(in_features, out_features, bias=True)
        self.activation = nn.ReLU()

    def forward(self, H, A):
        # H : tenseur de dimension (batch_size, n, in_features)
        # A : matrice d'adjacence (batch_size, n, n)
        
        # Agrégation du voisinage : A * H * W1
        neighbors_aggr = torch.bmm(A, self.W1(H)) 
        
        # Transformation du noeud lui-même : H * W2
        self_feat = self.W2(H)
        
        return self.activation(neighbors_aggr + self_feat)
    
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