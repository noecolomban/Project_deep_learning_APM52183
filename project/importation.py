import os
from pathlib import Path
import math
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from models import get_model, get_model_gen
from toolbox.losses import triplet_loss
from toolbox import metrics
from loaders.data_generator import QAP_Generator
from loaders.siamese_loaders import siamese_loader
from toolbox.metrics import all_losses_acc, accuracy_linear_assignment
from toolbox.utils import check_dir

#Fonctions déjà présentes sur le notebook initial. Nous avons ajouté un paramètre 'MGNN':bool pour prendre en compte ce modèle.

def get_device_config(model_path):
    config_file = os.path.join(model_path,'config.json')
    with open(config_file) as json_file:
        config_model = json.load(json_file)
    use_cuda = not config_model['cpu'] and torch.cuda.is_available()
    device = 'cuda' if use_cuda else 'cpu'
    return config_model, device

#Modifié pour pouvoir charger notre modèle MGNN
def load_model(model_path, config, device, MGNN=False):
    if MGNN: #Eviter les clés inexistantes
        model = get_model_gen(config['arch'])
        model.to(device)
        model_file = os.path.join(model_path,'model_best.pth.tar')
        if device == 'cpu':
            checkpoint = torch.load(model_file,map_location=torch.device('cpu'))
        else:
            checkpoint = torch.load(model_file)
    else:
        model = get_model(config['arch'])
        model.to(device)
        model_file = os.path.join(model_path,'model_best.pth.tar')
        if device == 'cpu':
            checkpoint = torch.load(model_file,map_location=torch.device('cpu'), weights_only=False)
        else:
            checkpoint = torch.load(model_file, weights_only=False)
    model.load_state_dict(checkpoint['state_dict'])
    return model

def creat_args(config, num_ex = 1000, MGNN=False):
    if not MGNN:
        args = config['data']
        args['num_examples_test'] = num_ex
        n_vertices = args['n_vertices']
        edge_density = args['edge_density']
        deg = (n_vertices)*edge_density
        print(f'graphs with {n_vertices} vertices and average degree {deg}')
    else: #Pour gérer les clés manquantes et définir des valeurs par défaut
        base_data = config.get('data', {})
        if 'train' in base_data:
            args = base_data['train'].copy()
        else:
            args = base_data.copy()
        for key in ['generative_model', 'noise_model', 'vertex_proba']:
            if key in base_data and key not in args:
                args[key] = base_data[key]
        args.setdefault('generative_model', 'ErdosRenyi')
        args.setdefault('noise_model', 'ErdosRenyi')
        args.setdefault('vertex_proba', 1.0)
        args['num_examples_test'] = num_ex
        n_vertices = args.get('n_vertices', 50)
        args['n_vertices'] = n_vertices 
        edge_density = args.get('density', args.get('edge_density', 0.2))
        args['edge_density'] = edge_density
        args['density'] = edge_density

        deg = n_vertices * edge_density
      
    return args, deg

def acc_2_error(mean_acc, q_acc):
    # q_acc[:,0] = upper quantile, q_acc[:,1] = lower quantile
    error = np.zeros((len(mean_acc), 2))
    error[:, 0] = mean_acc - q_acc[:, 1]  # lower error (mean - q_low), positive
    error[:, 1] = q_acc[:, 0] - mean_acc  # upper error (q_up - mean), positive
    return error

def compute_all(criterion, device, list_noise,args,path_dataset,model,bs=50):
    num_batches = math.ceil(args['num_examples_test']/bs)
    all_losses = np.zeros((len(list_noise),num_batches))
    all_acc = np.zeros((len(list_noise),args['num_examples_test']))
    for i,noise in enumerate(list_noise):
        args['noise'] = noise
        gene_test = QAP_Generator('test', args, path_dataset)
        gene_test.load_dataset()
        test_loader = siamese_loader(gene_test, bs, gene_test.constant_n_vertices)
        all_losses[i,:], all_acc[i,:] = all_losses_acc(test_loader,model,criterion,device,eval_score=accuracy_linear_assignment)
    return all_losses, all_acc

def compute_quant(all_acc,quant_low=0.1,quant_up=0.9):
    mean_acc = np.mean(all_acc,1)
    num = len(mean_acc)
    q_acc = np.zeros((num,2))
    for i in range(num):
        q_acc[i,:] = np.quantile(all_acc[i,:],[quant_up, quant_low])
    return mean_acc, q_acc

def compute_all_with_metric(list_noise, args, path_dataset, model, criterion, device, metric_fn, bs=50):
    num_batches = math.ceil(args['num_examples_test'] / bs)
    all_acc = np.zeros((len(list_noise), args['num_examples_test']))
    for i, noise in enumerate(list_noise):
        args['noise'] = noise
        gene_test = QAP_Generator('test', args, path_dataset)
        gene_test.load_dataset()
        test_loader = siamese_loader(gene_test, bs, gene_test.constant_n_vertices)
        _, all_acc[i, :] = all_losses_acc(test_loader, model, criterion, device, eval_score=metric_fn)
    return all_acc