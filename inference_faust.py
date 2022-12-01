import pickle
import argparse
import os.path as osp
import torch
from utils import mesh_sampling, utils
from utils.mesh_sampling import Mesh
import vtk
from vtk.util import numpy_support

from models import AE, AE_single

parser = argparse.ArgumentParser(description='mesh autoencoder')
parser.add_argument('--exp_name', type=str, default='arap')
parser.add_argument('--checkpoint', type=str, default=None)
parser.add_argument('--test_checkpoint', type=str, default=None)

parser.add_argument('--in_channels', type=int, default=3)
# network hyperparameters
parser.add_argument('--out_channels',
                    nargs='+',
                    default=[32, 32, 32, 64],                    
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
reader = vtk.vtkPLYReader()
reader.SetFileName(template_fp)
reader.Update()
template_poly = reader.GetOutput()

# generate/load transform matrices
transform_fp = osp.join('data', 'transform.pkl')

if not osp.exists(transform_fp):
    print('Generating transform matrices...')
    
    mesh = Mesh()
    mesh.SetPolyData(template_poly)
    
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

latest_checkpoint = 'work_dir/FAUST/checkpoint_0960.pt'
data = torch.load(latest_checkpoint, map_location=device)
state_dict = data['model_state_dict']
state_dict = {k.partition('module.')[2]: v for k,v in state_dict.items()} # Remove Prefix
model.load_state_dict(state_dict)


train_latent_vectors = data['train_latent_vecs']['weight']

sample_idx = 0
sample_input = train_latent_vectors[sample_idx:sample_idx+1]
sample_output = model(sample_input)

# Need to denormalize!!
print(sample_output.shape)


## Visualize
iren = vtk.vtkRenderWindowInteractor()
iren.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
renWin = vtk.vtkRenderWindow()
renWin.SetSize(1000, 1000)
iren.SetRenderWindow(renWin)
ren = vtk.vtkRenderer()
renWin.AddRenderer(ren)

output_polydata = vtk.vtkPolyData()
output_polydata.DeepCopy(template_poly)
output_polydata.GetPoints().SetData(numpy_support.numpy_to_vtk(sample_output[0].detach().numpy()))


mapper = vtk.vtkPolyDataMapper()
mapper.SetInputData(output_polydata)
actor = vtk.vtkActor()
actor.SetMapper(mapper)

ren.AddActor(actor)
ren.ResetCamera()
renWin.Render()
iren.Start()