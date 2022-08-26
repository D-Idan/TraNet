import numpy as np
import plotly.graph_objs as go
import plotly.io as pio
pio.renderers.default = "browser"

def plot3D_1part(part_num,firstPlot, secondPlot=False, thieredPlot=False):


    firstPlot = firstPlot.detach().cpu().numpy()

    fig = go.Figure()

    fig.add_trace( go.Scatter3d(
            x=firstPlot[part_num,0,:],
            y=firstPlot[part_num,1,:],
            z=firstPlot[part_num,2,:],
            mode='lines',
            marker=dict(line_width=3,size=5),
            name='lines+markers'))

    if not isinstance(secondPlot,bool):
        secondPlot = secondPlot.detach().cpu().numpy()
        fig.add_trace(go.Scatter3d(
            x=secondPlot[part_num, 0, :],
            y=secondPlot[part_num, 1, :],
            z=secondPlot[part_num, 2, :],
            mode='markers',
            marker=dict(line_width=1,size=2),
            name='secondPlot'))

    if not isinstance(thieredPlot, bool):
        thieredPlot = thieredPlot.detach().cpu().numpy()
        fig.add_trace(go.Scatter3d(
            x=thieredPlot[part_num, 0, :],
            y=thieredPlot[part_num, 1, :],
            z=thieredPlot[part_num, 2, :],
            mode='markers',
            marker=dict(line_width=1,size=2),
            name='thieredPlot'))


    fig.show()

def plot3D_Allpart(firstPlot, secondPlot=False, thieredPlot=False):


    firstPlot = firstPlot.detach().cpu().numpy()

    fig = go.Figure()

    fig.add_trace( go.Scatter3d(
            x=firstPlot[1,0,:],
            y=firstPlot[1,1,:],
            z=firstPlot[1,2,:],
            mode='lines',
            name='firstPlot'))

    if not isinstance(secondPlot,bool):
        secondPlot = secondPlot.detach().cpu().numpy()
        fig.add_trace(go.Scatter3d(
            x=secondPlot[1, 0, :],
            y=secondPlot[1, 1, :],
            z=secondPlot[1, 2, :],
            mode='lines',
            name='secondPlot'))

    if not isinstance(thieredPlot, bool):
        thieredPlot = thieredPlot.detach().cpu().numpy()
        fig.add_trace(go.Scatter3d(
            x=thieredPlot[1, 0, :],
            y=thieredPlot[1, 1, :],
            z=thieredPlot[1, 2, :],
            mode='lines',
            name='thieredPlot'))


    fig.show()

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)