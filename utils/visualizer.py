'''
This file is modified from:
https://github.com/zhoubenjia/RAAR3DNet/blob/master/Network_Train/utils/visualizer.py
'''


#coding: utf8

import numpy as np
import time


class Visualizer():
    def __init__(self, env='default', **kwargs):
        import visdom
        self.vis = visdom.Visdom(env=env, use_incoming_socket=False, **kwargs)

        self.index = {}
        self.log_text = ''

    def reinit(self, env='defult', **kwargs):
        self.vis = visdom.Visdom(env=env, use_incoming_socket=False, **kwargs)
        return self

    def plot_many(self, d, modality, epoch=None):
        colmu_stac = []
        for k, v in d.items():
            colmu_stac.append(np.array(v))
        if epoch:
            x = epoch
        else:
            x = self.index.get(modality, 0)
        # self.vis.line(Y=np.column_stack((np.array(dicts['loss1']), np.array(dicts['loss2']))),
        self.vis.line(Y=np.column_stack(tuple(colmu_stac)),
                        X=np.array([x]),
                        win=(modality),
                        # opts=dict(title=modality,legend=['loss1', 'loss2'], ylabel='loss value'),
                      opts=dict(title=modality, legend=list(d.keys()), ylabel='Value', xlabel='Iteration'),
                      update=None if x == 0 else 'append')
        if not epoch:
            self.index[modality] = x + 1

    def plot(self, name, y):
        """
        self.plot('loss',1.00)
        """
        x = self.index.get(name, 0)
        self.vis.line(Y=np.array([y]), X=np.array([x]),
                      win=(name),
                      opts=dict(title=name),
                      update=None if x == 0 else 'append'
                      )
        self.index[name] = x + 1

    def log(self, info, win='log_text'):
        """
        self.log({'loss':1,'lr':0.0001})
        """

        self.log_text += ('[{time}] {info} <br>'.format(
            time=time.strftime('%m.%d %H:%M:%S'),
            info=info))
        self.vis.text(self.log_text, win=win)

    def img_grid(self, name, input_3d, heatmap=False):
        self.vis.images(
            # np.random.randn(20, 3, 64, 64),
            show_image_grid(input_3d, name, heatmap),
            win=name,
            opts=dict(title=name, caption='img_grid.')
        )
    def img(self, name, input):
        self.vis.images(
            input,
            win=name,
            opts=dict(title=name, caption='RGB Images.')
        )

    def draw_curve(self, name, data):
        self.vis.line(Y=np.array(data), X=np.array(range(len(data))),
                      win=(name),
                      opts=dict(title=name),
                      update=None
                      )

    def featuremap(self, name, input):
        self.vis.heatmap(input, win=name, opts=dict(title=name))

    def draw_bar(self, name, inp):
        self.vis.bar(
            X=np.abs(np.array(inp)), 
            win=name,
            opts=dict(
                stacked=True,  
                legend=list(map(str, range(inp.shape[-1]))),
                rownames=list(map(str, range(inp.shape[0])))  
            )
        )


    def __getattr__(self, name):
        return getattr(self.vis, name)
