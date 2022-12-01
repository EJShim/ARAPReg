import vtk

if __name__ == "__main__":
    iren = vtk.vtkRenderWindowInteractor()
    iren.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
    renWin = vtk.vtkRenderWindow()
    iren.SetRenderWindow(renWin)
    ren = vtk.vtkRenderer()
    renWin.AddRenderer(ren)

    reader = vtk.vtkPLYReader()
    reader.SetFileName("template/smpl_male_template.ply")
    reader.Update()
    polydata = reader.GetOutput()
    print(polydata.GetNumberOfPoints())

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(reader.GetOutput())
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)

    ren.AddActor(actor)
    ren.ResetCamera()
    renWin.Render()
    iren.Start()

