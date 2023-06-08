import vtk


if __name__ == "__main__":
    iren = vtk.vtkRenderWindowInteractor()
    iren.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
    renWin = vtk.vtkRenderWindow()
    renWin.SetSize(1000, 1000)
    iren.SetRenderWindow(renWin)
    ren = vtk.vtkRenderer()
    renWin.AddRenderer(ren)


    renWin.Render()
    iren.Start()