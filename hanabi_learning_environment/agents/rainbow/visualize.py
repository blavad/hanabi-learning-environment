import plotly
import plotly.graph_objects as go
import numpy as np
import pickle
import tensorflow as tf
import sys
import re
import os

models = {re.split('/', path)[-3] : path for path in sys.argv[1:-1]}
data = []
for url in models.values():
    with tf.gfile.GFile(url, 'rb') as f:
          data.append(pickle.load(f))

rews = []
for d in data:
    rew = []
    for k,v in d.items():
        rew+=v["average_return"]
    rews.append(rew)

delta=1
rews_ma = []
for rew_l in rews:
    rew_ma = []
    for i in range(delta, len(rew_l)-delta):
        rew_ma.append(np.mean(rew_l[i-delta:i+delta]))
    rews_ma.append(rew_ma)

names=list(models.keys())


fig = go.Figure()
# fig.add_trace(go.Scatter(y=[13.96 for i in range(11000)], mode="lines", name="simple bot", showlegend=True))
for i in range(len(rews_ma)):
    fig.add_trace(go.Scatter(y=rews_ma[i], mode="lines", name="{}".format(names[i]), showlegend=True))
   

fig.update_layout(title='Training process',
                   xaxis_title='Iteration (150-200 episodes/iter)',
                   yaxis_title='Average Return',
                   height= 700,
                   width = 1000)

fig.update_xaxes(range=[-1, 10100])
fig.update_yaxes(range=[0, 18])

# fig.show('browser')


if not os.path.exists("images"):
    os.mkdir("images")

res = plotly.offline.plot(fig, include_plotlyjs=False, output_type='div')
# print(res)

tmp=""
with open("images/{}.html".format(sys.argv[-1]), 'w+') as f:
    tmp+="<script src='https://cdn.plot.ly/plotly-latest.min.js'></script>"
    tmp+=res
    f.write(tmp)