import pickle
import argparse
import os.path as osp
import torch
import torch.nn as nn
from utils import mesh_sampling
from utils.mesh_sampling import Mesh
import vtk
# import torch.backends.cudnn as cudnn
# import torch_geometric.transforms as T

from models import AE

parser = argparse.ArgumentParser(description='mesh autoencoder')
parser.add_argument('--exp_name', type=str, default='arap')
parser.add_argument('--checkpoint', type=str, default=None)
parser.add_argument('--test_checkpoint', type=str, default=None)

parser.add_argument('--in_channels', type=int, default=3)
# network hyperparameters
parser.add_argument('--out_channels',
                    nargs='+',
                    #default=[32, 32, 32, 64],
                    default=[16, 16, 16, 32],
                    type=int)

parser.add_argument('--ds_factors',
                    nargs='+',
                    #default=[32, 32, 32, 64],
                    default=[2,2,2,2],
                    type=int)
parser.add_argument('--latent_channels', type=int, default=72)

parser.add_argument('--K', type=int, default=6)

# optimizer hyperparmeters
args = parser.parse_args()



device = torch.device('cpu')
# torch.set_num_threads(args.n_threads)


# load dataset
template_fp = osp.join('template', 'smpl_male_template.ply')

# generate/load transform matrices
transform_fp = osp.join('data', 'transform.pkl')

if not osp.exists(transform_fp):
    print('Generating transform matrices...')
    reader = vtk.vtkPLYReader()
    reader.SetFileName(template_fp)
    reader.Update()

    mesh = Mesh()
    mesh.SetPolyData(reader.GetOutput())
    
    ds_factors = args.ds_factors    
    _, A, D, U, F = mesh_sampling.generate_transform_matrices(mesh, ds_factors)
    tmp = {'face': F, 'adj': A, 'down_transform': D, 'up_transform': U}

    with open(transform_fp, 'wb') as fp:
        pickle.dump(tmp, fp)
    print('Done!')
    print('Transform matrices are saved in \'{}\''.format(transform_fp))
else:
    with open(transform_fp, 'rb') as f:
        tmp = pickle.load(f, encoding='latin1')

exit()

edge_index_list = [utils.to_edge_index(adj).to(device) for adj in tmp['adj']]
down_transform_list = [
    utils.to_sparse(down_transform).to(device)
    for down_transform in tmp['down_transform']
]
up_transform_list = [
    utils.to_sparse(up_transform).to(device)
    for up_transform in tmp['up_transform']
]





model = AE(args.in_channels,
        args.out_channels,
        args.latent_channels,
        edge_index_list,
        down_transform_list,
        up_transform_list,
        K=args.K)    

data = torch.load(latest_checkpoint)
model.load_state_dict(data["model_state_dict"])