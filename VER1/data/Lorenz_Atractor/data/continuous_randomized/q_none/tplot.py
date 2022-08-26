import numpy as np
import plotly.graph_objs as go
import plotly.io as pio
pio.renderers.default = "browser"
import torch



def plot3D(firstPlot, secondPlot=False, thieredPlot=False):


    firstPlot = firstPlot.detach().numpy()

    fig = go.Figure()

    fig.add_trace( go.Scatter3d(
            x=firstPlot[0,0,:],
            y=firstPlot[0,1,:],
            z=firstPlot[0,2,:],
            mode='lines',
            name='firstPlot'))

    if not isinstance(secondPlot,bool):
        secondPlot = secondPlot.detach().numpy()
        fig.add_trace(go.Scatter3d(
            x=secondPlot[1, 0, :],
            y=secondPlot[1, 1, :],
            z=secondPlot[1, 2, :],
            mode='lines',
            name='secondPlot'))

    if not isinstance(thieredPlot, bool):
        thieredPlot = thieredPlot.detach().numpy()
        fig.add_trace(go.Scatter3d(
            x=thieredPlot[2, 0, :],
            y=thieredPlot[2, 1, :],
            z=thieredPlot[2, 2, :],
            mode='lines',
            name='thieredPlot'))


    fig.show()

GT_train = torch.load('GT_train.pt')
GT_test = torch.load('GT_test.pt')
obs_identity_0 = torch.load('obs_identity_0.pt')
GT_test_long = torch.load('GT_test_long.pt')

plot3D(GT_test)