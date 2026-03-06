from typing import Tuple
from numpy import isin
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.base_model import Simple_Node_Embedding

from models.gated_gcn import GatedGCNNet_Node, GatedGCNNet_Edge


from models.MGNN_net import MGNNLayer, MGNN


class Siamese_Model(nn.Module):
    def __init__(self, original_features_num, num_blocks, in_features,out_features, depth_of_mlp, embedding_class=Simple_Node_Embedding):
        """
        take a batch of pair of graphs 
        ((bs, n_vertices, n_vertices, in_features) (bs,n_vertices, n_vertices, in_features))
        and return a batch of node similarities (bs, n_vertices, n_vertices)
        for each node the sum over the second dim should be one: sum(torch.exp(out[b,i,:]))==1
        graphs must have same size inside the batch
        """
        super().__init__()

        self.original_features_num = original_features_num
        self.num_blocks = num_blocks
        self.in_features = in_features
        self.out_features = out_features
        self.depth_of_mlp = depth_of_mlp
        self.node_embedder = embedding_class(original_features_num, num_blocks, in_features,out_features, depth_of_mlp)

    def forward(self, x):
        """
        Data should be given with the shape (x1,x2)
        """
        assert x.shape[1]==2, f"Data given is not of the shape (x1,x2) => data.shape={x.shape}"
        x = x.permute(1,0,2,3,4)
        x1 = x[0]
        x2 = x[1]
        x1 = self.node_embedder(x1)
        x2 = self.node_embedder(x2)
        raw_scores = torch.matmul(x1,torch.transpose(x2, 1, 2))
        return raw_scores

class Siamese_Model_Gen(nn.Module):
    def __init__(self, Model_class,**kwargs):
        """
        General class enforcing a Siamese architecture.
        The forward usually takes in a pair of graphs of shape:
        ((bs, n_vertices, n_vertices, in_features) (bs, n_vertices, n_vertices, in_features))
        and return a batch of node similarities (bs, n_vertices, n_vertices).
        That was the base use, but model can be anything and return mostly anything as long as the helper is taken into account
        """
        self.model=''
        super().__init__()
        self.node_embedder = Model_class(**kwargs)
        if isinstance(self.node_embedder,GatedGCNNet_Node):
            self.model='gatedgcnnode'
        elif isinstance(self.node_embedder,GatedGCNNet_Edge):
            self.model='gatedgcnedge'

    def forward(self,x):
        """
        Data should be given with the shape (x1,x2)
        """
        if isinstance(x,torch.Tensor):
            assert x.shape[1]==2, f"Data given is not of the shape (x1,x2) => data.shape={x.shape}"
            x = x.permute(1,0,2,3,4)
            x1 = x[0]
            x2 = x[1]
        else:
            assert len(x)==2, f"Data given is not of the shape (x1,x2) => data.shape={x.shape}"
            x1 = x[0]
            x2 = x[1]
        x1_out = self.node_embedder(x1)
        x2_out = self.node_embedder(x2)
        if self.model=='gatedgcnedge':
            #We're in dgl territory
            N = x1.number_of_nodes()
            N_features = x1_out.shape[1]
            src1,dst1 = x1.edges(form='uv')
            src2,dst2 = x2.edges(form='uv')
            assert src1.shape[0] == x1_out.shape[0] and src2.shape[0] == x2_out.shape[0]
            assert x1.number_of_nodes()==x2.number_of_nodes()
            fullt = torch.zeros((N,N),dtype=bool,requires_grad=False)
            fullt1 = torch.zeros((N,N,N_features))
            fullt2 = torch.zeros_like(fullt1)
            for i,(in1,out1,in2,out2) in enumerate(zip(src1,dst1,src2,dst2)):
                fullt[in1,out1] = True
                fullt[in2,out2] = True
                fullt1[in1,out1] = x1_out[i]
                fullt2[in2,out2] = x2_out[i]
            raw_scores = torch.matmul(fullt1,torch.transpose(fullt2,1,2)).unsqueeze(0) #unsqueeze to imitate a batch of 1
        elif self.model=='gatedgcnnode':
            raw_scores = torch.matmul(x1_out,torch.transpose(x2_out,-2,-1)).unsqueeze(0)
        else:
            raw_scores = torch.matmul(x1_out,torch.transpose(x2_out, 1, 2))
        return raw_scores
    

class SiameseMGNN(nn.Module):
    def __init__(self, **kwargs):
        super(SiameseMGNN, self).__init__()

        # ÉTAPE 2 : On récupère les valeurs depuis les kwargs envoyés par la configuration.
        # S'ils n'y sont pas, on donne une valeur par défaut (ex: 64 et 3).
        hidden_features = kwargs.get('features', 64)   
        num_layers = kwargs.get('depth', 3)            
        out_features = kwargs.get('features', 64)
        in_features = 1 # Pas de features de noeuds initiales pour ce dataset
        
        # Astuce de debug : pour voir les vrais noms envoyés par le programme
        print("====== DEBUG KWARGS ======")
        print(kwargs)
        print("==========================")

        # ÉTAPE 3 : On instancie le modèle MGNN de base
        # (Vérifiez bien que la classe MGNN, elle, accepte ces 4 paramètres !)
        self.node_embedder = MGNN(in_features, hidden_features, out_features, num_layers)
        
        # On garde le paramètre de température
        self.tau = nn.Parameter(torch.tensor(1.0))

    def forward(self, A1, A2, X1=None, X2=None):
        """
        A1, A2 : Matrices d'adjacence des deux graphes (batch, n, n)
        X1, X2 : Features des noeuds (batch, n, d_in). 
                 Si les graphes n'ont pas de features, on peut créer un tenseur de 1.
        """
        batch_size, n, _ = A1.shape
        
        # Si aucune feature n'est fournie, on initialise avec des 1 pour chaque noeud
        if X1 is None:
            X1 = torch.ones(batch_size, n, 1, device=A1.device)
        if X2 is None:
            X2 = torch.ones(batch_size, n, 1, device=A2.device)

        # --- ÉTAPE 1 & 2 : Passage dans les deux branches jumelles ---
        # H1 shape: (batch, n, d_out)
        H1 = self.gnn(X1, A1) 
        # H2 shape: (batch, n, d_out)
        H2 = self.gnn(X2, A2) 

        # --- ÉTAPE 3 : Calcul de la matrice d'alignement (Croisement) ---
        # On fait le produit scalaire entre tous les noeuds de G1 et tous les noeuds de G2
        # H1 : (batch, n, d_out)
        # H2 transposé : (batch, d_out, n)
        # Résultat S : (batch, n, n)
        
        S = torch.bmm(H1, H2.transpose(1, 2))
        
        # On divise par la température (aide à l'apprentissage)
        S = S / self.tau

        # --- ÉTAPE 4 : Softmax ou Sinkhorn ---
        # Pour forcer la matrice S à ressembler à une matrice de permutation (probabilités)
        # on applique souvent un LogSoftmax sur les lignes. 
        # Le code d'origine de Lelarge utilise peut-être une loss spécifique,
        # donc renvoyer S brut ou S passé dans un log_softmax dépendra de la loss utilisée dans toolbox/losses.py.
        
        # Exemple basique pour sortir des log-probabilités d'assignement :
        # return torch.nn.functional.log_softmax(S, dim=-1)
        
        return S