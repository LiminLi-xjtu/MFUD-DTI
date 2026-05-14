import json
import pandas as pd
import torch
import numpy as np
import os
import random
# Utils
from utils.utils import DataLoader, virtual_screening
from utils.dataset import *  # data
from utils.trainer import Trainer
from utils.metrics import *
# Preprocessing
from utils import protein_init, ligand_init
# Model
from models.net import net
# Config
from tqdm import tqdm
import argparse
parser = argparse.ArgumentParser()

## Device and batch size
parser.add_argument('--device', type=str, default='cuda:0', help='')
# parser.add_argument('--trained_model_path',type=str, default='/ssd2/lxy_code/DTI/GTBAN_copy/result/ALL_r/save_model_seed3')
parser.add_argument('--trained_model_path',type=str, default='/ssd2/lxy_code/DTI/GTBAN_copy/result/cdk4_6_2/save_model_seed3')

parser.add_argument('--batch_size', type=int, default=256)
## Data and Pre-processing
parser.add_argument('--datafolder', type=str, default='/ssd2/lxy_code/DTI/GTBAN_copy/dataset/yamanishi_08/cdk4_6_1')
parser.add_argument('--screenfile', type=str, default='/ssd2/lxy_code/DTI/GTBAN_copy/dataset/yamanishi_08/cdk4_6_1/test.csv', help='csv file')  
parser.add_argument('--result_path', type=str,default='vision/cdk_2',help='path to save results')
parser.add_argument('--save_interpret', type=bool,default=True,help='Save interpretation from PSICHIC?')
parser.add_argument('--save_cluster', type=bool,default=True,help='Save Residue-to-Region Assignment Matrix from PSICHIC?')

args = parser.parse_args()


with open(os.path.join(args.trained_model_path,'config.json'),'r') as f:
    config = json.load(f)

print("Screening the csv file: {}".format(args.screenfile))
# device
device = torch.device(args.device)


if not os.path.exists(args.result_path):
    os.makedirs(args.result_path)

print(args)
with open(os.path.join(args.result_path, 'screening_params_cdk_2.txt'), 'w') as f:
    f.write(str(args))

# degree_dict = torch.load(os.path.join(args.trained_model_path,'degree.pt'))
param_dict = os.path.join(args.trained_model_path,'model.pt')
# mol_deg, prot_deg = degree_dict['ligand_deg'],degree_dict['protein_deg']

model = net(# MOLECULE
            mol_in_channels=config['params']['mol_in_channels'],  prot_in_channels=config['params']['prot_in_channels'], 
            prot_evo_channels=config['params']['prot_evo_channels'],
            hidden_channels=config['params']['hidden_channels'],total_layer=config['params']['total_layer'], 
            num_layers=config['params']['num_layer'],               
            K = config['params']['K'],heads=config['params']['heads'], 
            dropout=config['params']['dropout'],
            dropout_attn_score=config['params']['dropout_attn_score'],
            # output
            regression_head=config['tasks']['regression_task'],
            classification_head=config['tasks']['classification_task'] ,
            multiclassification_head=config['tasks']['mclassification_task'],
            device=device).to(device)
model.reset_parameters()    
model.load_state_dict(torch.load(param_dict,map_location=args.device))


screen_df = pd.read_csv(os.path.join(args.screenfile))



protein_seqs = screen_df['Protein'].unique().tolist()
# print('Initialising protein sequence to Protein Graph')
# protein_dict = protein_init(protein_seqs)
ligand_smiles = screen_df['Ligand'].unique().tolist()
# print('Initialising ligand SMILES to Ligand Graph')
# ligand_dict = ligand_init(ligand_smiles)


protein_path = os.path.join(args.datafolder,'protein.pt')
if os.path.exists(protein_path):
    print('Loading Protein Graph data...')
    protein_dict = torch.load(protein_path)
else:
    print('Initialising Protein Sequence to Protein Graph...')
    protein_dict = protein_init(protein_seqs)
    torch.save(protein_dict,protein_path)

ligand_path = os.path.join(args.datafolder,'ligand.pt')
if os.path.exists(ligand_path):
    print('Loading Ligand Graph data...')
    ligand_dict = torch.load(ligand_path)
else:
    print('Initialising Ligand SMILES to Ligand Graph...')
    ligand_dict = ligand_init(ligand_smiles)
    torch.save(ligand_dict,ligand_path)


torch.cuda.empty_cache()
## drop any invalid smiles
screen_df = screen_df[screen_df['Ligand'].isin(list(ligand_dict.keys()))].reset_index(drop=True)
screen_dataset = ProteinMoleculeDataset(screen_df, ligand_dict, protein_dict, device=args.device)
screen_loader = DataLoader(screen_dataset, batch_size=args.batch_size, shuffle=False,
                            follow_batch=['mol_x', 'clique_x', 'prot_node_aa'])

print("Screening starts now!")
screen_df = virtual_screening(screen_df, model, screen_loader,
                 result_path=os.path.join(args.result_path, "interpretation_result"), save_interpret=args.save_interpret, 
                 ligand_dict=ligand_dict, device=args.device,
                 save_cluster=args.save_cluster)

screen_df.to_csv(os.path.join(args.result_path,'cdk_2.csv'),index=False)
print('Screening completed and saved to {}'.format(args.result_path))