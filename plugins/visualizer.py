# visualizer.py

import visdom
import numpy as np
import plotly.graph_objs as go
# import plotly.graph_objects as go

# import plotly.express as go
# import plotly.express as go
from utils import plotlify
from collections import OrderedDict

class Visualizer:
    def __init__(self, port, env, title):
        self.keys = []
        self.values = {}
        self.env = env
        self.viz = visdom.Visdom(port=port, env=env)
        self.iteration = 0
        self.title = title

    def register(self, modules):

        # here modules are assumed to be a dictionary
        for key in modules:

            # import pdb
            # pdb.set_trace()

            self.keys.append(key)
            self.values[key] = {}
            self.values[key]['dtype'] = modules[key]['dtype']
            self.values[key]['vtype'] = modules[key]['vtype']
            self.values[key]['win'] = modules[key]['win'] \
                if 'win' in modules[key].keys() \
                else None


            if modules[key]['vtype'] == 'plot':
                self.values[key]['layout'] = modules[key]['layout'] \
                    if 'layout' in modules[key].keys() \
                    else {'windows': [key], 'id': 0}
                self.values[key]['value'] = []
                # Create traces
                data = [go.Scatter(
                    x=[],
                    y=[],
                    mode='lines',
                    name=self.values[key]['layout']['windows'][i]
                ) for i in range(len(self.values[key]['layout']['windows']))]
                # Edit the layout
                layout = dict(
                    title=key,
                    xaxis=dict(title='Epoch'),
                    yaxis=dict(title=key),
                )
                fig = go.Figure(data=data, layout=layout)
                fig = fig.to_dict()
                self.values[key]['win'] = self.viz._send(
                    plotlify(fig, env=self.env, win=self.values[key]['win']))
            elif modules[key]['vtype'] == 'scatter':

                    self.values[key]['layout'] = modules[key]['layout'] \
                        # if 'layout' in modules[key].keys() \
                        # else {'windows': [key], 'id': 2}
                    self.values[key]['value'] = []
                    # Create traces
                    data = [go.Scatter(
                        x=[],
                        y=[],
                        mode='markers',
                        name=self.values[key]['layout']['windows'][i]
                    ) for i in range(len(self.values[key]['layout']['windows']))]
                    # Edit the layout
                    layout = dict(
                        # title=key,
                        # xaxis=dict(title='Epoch'),
                        # yaxis=dict(title=key),
                    )
                    fig = go.Figure(data=data, layout=layout)
                    # fig['data'][1]['marker']['symbol'] = 'triangle-left'
                    fig = fig.to_dict()
                    self.values[key]['win'] = self.viz._send(
                        plotlify(fig, env=self.env, win=self.values[key]['win']))
            elif modules[key]['vtype'] == 'image':
                self.values[key]['value'] = None
            elif modules[key]['vtype'] == 'images':
                self.values[key]['value'] = None
            else:
                raise Exception('Data type not supported, please update the '
                                'visualizer plugin and rerun !!')

    def update(self, modules):
        # import pdb
        # pdb.set_trace()

        for key in modules:
            if self.values[key]['dtype'] == 'scalar':
                self.values[key]['value'].append(modules[key])
            elif self.values[key]['dtype'] == 'vector':
                self.values[key]['value'] = modules[key]
            elif self.values[key]['dtype'] == 'image':
                self.values[key]['value'] = modules[key]
            elif self.values[key]['dtype'] == 'images':
                self.values[key]['value'] = modules[key]
            else:
                raise Exception('Data type not supported, please update the '
                                'visualizer plugin and rerun !!')

        for key in self.keys:

            if self.values[key]['vtype'] == 'plot':
                # Create traces
                data = [go.Scatter(
                    x=[],
                    y=[],
                ) for i in range(len(self.values[key]['layout']['windows']))]
                data[self.values[key]['layout']['id']] = go.Scatter(
                    x=np.array([self.iteration]).tolist(),
                    y=np.array([self.values[key]['value'][-1]]).tolist(),
                )
                data = go.Figure(data=data)
                fig = data.to_dict()
                fig['append'] = True
                self.viz._send(
                    plotlify(fig, env=self.env,
                             win=self.values[key]['win']), endpoint='update')
            elif self.values[key]['vtype'] == 'scatter':

                # Create traces
                data = [go.Scatter(
                    x=[],
                    y=[],
                ) for i in range(len(self.values[key]['layout']['windows']))]


                # for i in range(len(np.array([self.values[key]['value']])[0,0])):


                if self.values[key]['layout']['id']==0:
                    data[self.values[key]['layout']['id']] = go.Scatter(
                        x=[self.values[key]['value']][0][0][:,0].tolist(),
                        # x=[self.values[key]['value']][0][0],
                        y=[self.values[key]['value']][0][0][:,1].tolist(),
                        # y=[self.values[key]['value']][0][1],
                        mode='markers',
                        marker=dict(size=6, color=([self.values[key]['value']][0][1][:,0]+1).tolist(),
                                    symbol=([self.values[key]['value']][0][2][:,0]).tolist())
                    # x=[1,2],
                    # y=[3,4]
                    )
                # else:
                #
                #     data[self.values[key]['layout']['id']] = go.Scatter(
                #         x=[self.values[key]['value']][0][0][:,0].tolist(),
                #         # x=[self.values[key]['value']][0][0],
                #         y=[self.values[key]['value']][0][0][:,1].tolist(),
                #         # y=[self.values[key]['value']][0][1],
                #         mode='markers',
                #         marker=dict(color=([self.values[key]['value']][0][2][:,0]).tolist(),
                #                     symbol=([self.values[key]['value']][0][1][:,0]).tolist())
                #     # x=[1,2],
                #     # y=[3,4]
                #     )
                # import pdb
                # pdb.set_trace()
                data = go.Figure(data=data)
                fig = data.to_dict()
                fig['append'] = False

                # fig['data'][0]['marker']['symbol'] = 'square'
                # self.viz._send(
                #     plotlify(fig, env=self.env,
                #              win=self.values[key]['win']), endpoint='update')
                self.viz._send(
                    plotlify(fig, env=self.env, win=self.values[key]['win']))
            elif self.values[key]['vtype'] == 'image':
                temp = self.values[key]['value'].numpy()
                for i in range(temp.shape[0]):
                    temp[i] = temp[i] - temp[i].min()
                    temp[i] = temp[i] / temp[i].max()
                if self.iteration == 0:
                    self.values[key]['win'] = self.viz.image(
                        temp,
                        win=self.values[key]['win'],
                        opts=dict(title=key, caption=self.iteration)
                    )
                else:
                    self.viz.image(
                        temp,
                        win=self.values[key]['win'],
                        opts=dict(title=key, caption=self.iteration)
                    )
            elif self.values[key]['vtype'] == 'images':
                temp = self.values[key]['value'].numpy()
                for i in range(temp.shape[0]):
                    for j in range(temp.shape[1]):
                        temp[i][j] = temp[i][j] - temp[i][j].min()
                        temp[i][j] = temp[i][j] / temp[i][j].max()
                if self.iteration == 0:
                    self.values[key]['win'] = self.viz.images(
                        temp,
                        win=self.values[key]['win'],
                        opts=dict(title=key, caption=self.iteration)
                    )
                else:
                    self.viz.images(
                        temp,
                        opts=dict(
                            title=key,
                            caption=self.iteration
                        ),
                        win=self.values[key]['win']
                    )
            else:
                raise Exception('Visualization type not supported, please '
                                'update the visualizer plugin and rerun !!')
        self.iteration = self.iteration + 1

    def getvalues(self, key=None):
        if key is not None:
            return self.values[key]['value']
        else:
            return OrderedDict(
                [(key, self.values[key]['value']) for key in self.keys])
