import pickle
import argparse
import os.path as osp
import torch
from utils import mesh_sampling, utils
from utils.mesh_sampling import Mesh
import vtk
from vtk.util import numpy_support
from models import AE
import numpy as np


class ServiceModel(AE):
    def __init__(self, in_channels, out_channels, latent_channels, edge_index, down_transform, up_transform, K, mean, std):
        super().__init__(in_channels, out_channels, latent_channels, edge_index, down_transform, up_transform, K)

        self.mean = mean
        self.std = std

    def forward(self,x):
        y = super().forward(x)
        

        return y * self.std + self.mean

class LatentInteractorStyle(vtk.vtkInteractorStyleTrackballCamera):

    def __init__(self, model, vectors):
        self.AddObserver("LeftButtonPressEvent", self.LeftButtonPressed)
        self.AddObserver("MouseMoveEvent", self.MouseMove)
        self.AddObserver("LeftButtonReleaseEvent", self.LeftButtonReleased)

        #Initialize model
        self.model = model

        #Sample vectors
        self.vectors = vectors
        
    def initialize(self, polydata):
        ren = self.GetInteractor().GetRenderWindow().GetRenderers().GetFirstRenderer()
        self.polydata = vtk.vtkPolyData()
        self.polydata.DeepCopy(polydata)
        self.actor = self.make_actor(self.polydata)
        self.actor.SetScale(0.5,0.5,0.5)
        ren.AddActor(self.actor)

        #Initialize Plane        
        planeSource = vtk.vtkPlaneSource()
        planeSource.SetCenter(0, 0, 0)

        planeSource.Update()
        planePoly = planeSource.GetOutput()
        planePoly.GetPointData().RemoveArray("Normals")
        self.planeActor = self.make_actor(planePoly)        
        self.planeActor.GetProperty().SetRepresentationToWireframe()
        self.planeActor.GetProperty().SetColor(1, 0, 0)        

        
        ren.AddActor(self.planeActor)


        # #Initialize Template Polydata
        bounds = self.planeActor.GetBounds()

        self.latentPositions = np.array( [
            [bounds[0], bounds[2], 0],
            [bounds[0], bounds[3], 0],
            [bounds[1], bounds[2], 0],
            [bounds[1], bounds[3], 0]
        ])
    


        template_pos = [
            [bounds[0], bounds[2], 0],
            [bounds[0], bounds[3], 0],
            [bounds[1], bounds[2], 0],
            [bounds[1], bounds[3], 0]
        ]

        output_v = model(self.vectors).detach().cpu().numpy()
        print(output_v.shape)

        for idx, pos in enumerate(template_pos):
            template_poly = vtk.vtkPolyData()
            template_poly.DeepCopy(polydata)
            template_poly.GetPoints().SetData(numpy_support.numpy_to_vtk(output_v[idx]))

            template_actor = self.make_actor(template_poly)
            template_actor.SetPosition(pos[0], pos[1], pos[2])
            template_actor.SetScale(0.5, 0.5, 0.5)
            ren.AddActor(template_actor)          

        self.pickedPosition = -1


    def LeftButtonPressed(self, obj, ev):
        
        self.OnLeftButtonDown()

        pos = obj.GetInteractor().GetEventPosition()

        picker = vtk.vtkCellPicker()
        picker.PickFromListOn()
        picker.AddPickList(self.planeActor)
        picker.Pick(pos[0], pos[1], 0, ren)

    

        position = picker.GetPickPosition()
        
        if picker.GetActor() == self.planeActor:
            self.pickedPosition = position

    def MouseMove(self, obj, ev):
        renWin = self.GetInteractor().GetRenderWindow()
        if self.pickedPosition == -1:
            self.OnMouseMove()
            return

        pos = obj.GetInteractor().GetEventPosition()

        picker = vtk.vtkCellPicker()
        picker.PickFromListOn()
        picker.AddPickList(self.planeActor)
        picker.Pick(pos[0], pos[1], 0, ren)
        
        
        position = picker.GetPickPosition()
        targetPos = np.array([position[0], position[1], 0])

        if targetPos[0] < -50 : targetPos[0] = -50
        elif targetPos[0] > 50 : targetPos[0] = 50
        if targetPos[1] < -50 : targetPos[1] = -50
        elif targetPos[1] > 50 : targetPos[1] = 50




        #Maximum length sqrt(2) = 1.4142135623730951
        distances = []
        for sample in self.latentPositions:            
            distances.append(np.linalg.norm(targetPos-sample))

        weights = np.array(distances)
        weights[weights > 1] = 1
        weights = 1 - weights


        calculatedLatent = torch.zeros(72, dtype=torch.float32)
        for idx, weight in enumerate(weights):
            calculatedLatent += self.vectors[idx] * weight
        calculatedLatent.unsqueeze(0)
        
        pred = self.model(calculatedLatent).detach().numpy()

        self.polydata.GetPoints().SetData(numpy_support.numpy_to_vtk(pred[0]))
        self.polydata.GetPoints().Modified()        

        renWin.Render()

    def LeftButtonReleased(self, obj, ev):

        self.pickedPosition = -1
        self.OnLeftButtonUp()


    def make_actor(self, polydata):
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(polydata)
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)

        return actor

if __name__ == "__main__":

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
    
    meanstd = torch.load("data/DFaust/raw/meanstd.pt")
    mean = meanstd['mean']
    std = meanstd['std']

    model = ServiceModel(args.in_channels,
            args.out_channels,
            args.latent_channels,
            edge_index_list,
            down_transform_list,
            up_transform_list,
            K=args.K,
            mean=mean,
            std=std)

    

    latest_checkpoint = 'work_dir/FAUST/checkpoint_0960.pt'
    data = torch.load(latest_checkpoint, map_location=device)
    state_dict = data['model_state_dict']
    state_dict = {k.partition('module.')[2]: v for k,v in state_dict.items()} # Remove Prefix
    model.load_state_dict(state_dict)


    train_latent_vectors = data['train_latent_vecs']['weight']
    
    sample_vectors = train_latent_vectors.index_select(0, torch.tensor([1000,2000,3000,4000]))
    

    sample_idx = 4000
    sample_input = train_latent_vectors[sample_idx:sample_idx+1]
    out_v = model(sample_input)
    # out_v = sample_output*std + mean

    ## Visualize
    iren = vtk.vtkRenderWindowInteractor()    
    interactorStyle = LatentInteractorStyle(model, sample_vectors)
    iren.SetInteractorStyle(interactorStyle)
    renWin = vtk.vtkRenderWindow()
    renWin.SetSize(1000, 1000)
    iren.SetRenderWindow(renWin)
    ren = vtk.vtkRenderer()
    ren.GradientBackgroundOn()
    ren.SetBackground( 235/255,206/255, 135/255)
    ren.SetBackground2( 158/255,125/255, 44/255,)
    renWin.AddRenderer(ren)

    #Initialzize
    interactorStyle.initialize(polydata=template_poly)

    # output_polydata = vtk.vtkPolyData()
    # output_polydata.DeepCopy(template_poly)
    # output_polydata.GetPoints().SetData(numpy_support.numpy_to_vtk(out_v[0].detach().numpy()))


    # mapper = vtk.vtkPolyDataMapper()
    # mapper.SetInputData(output_polydata)
    # actor = vtk.vtkActor()
    # actor.SetMapper(mapper)

    # ren.AddActor(actor)
    ren.ResetCamera()
    renWin.Render()
    iren.Start()