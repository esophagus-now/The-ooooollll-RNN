import numpy as np
dbg_data = [
{
	"tag":"before_bp[0]",
	"file":"main.cpp",
	"line":87,
	"data":{"fc_1": {
"W": np.array([[-0.999624, -0.686186, -0.730945], [-0.984794, 0.572827, -0.489959], [-0.741858, -0.414494, -0.393773]]),
"bias": np.array([-0.999624, -0.686186, -0.730945])
}},
},
{
	"tag":"bp_inputs[0]",
	"file":"main.cpp",
	"line":88,
	"data":np.array([[-0.1, -0.3, 0.4]])
},
{
	"tag":"bp_dy_in[0]",
	"file":"main.cpp",
	"line":89,
	"data":np.array([[0.0276324, -2.31108, -8.37984]])
},
{
	"tag":"z2",
	"file":"./layers.h",
	"line":172,
	"data":np.array([[0.0276324, -2.31108, -8.37984]])
},
{
	"tag":"dErr_dW",
	"file":"./layers.h",
	"line":177,
	"data":np.array([[-0.00276324, -0.00828971, 0.0110529], [0.231108, 0.693323, -0.924431], [0.837984, 2.51395, -3.35194]])
},
{
	"tag":"after_bp[0]",
	"file":"main.cpp",
	"line":91,
	"data":{"fc_1": {
"W": np.array([[-0.999583, -0.686062, -0.73111], [-0.98826, 0.562428, -0.476093], [-0.754428, -0.452203, -0.343494]]),
"bias": np.array([-1.00004, -0.65152, -0.605247])
}},
},
{
	"tag":"bp_outputs[0]",
	"file":"main.cpp",
	"line":92,
	"data":np.array([[8.46497, 2.13058, 4.41189]])
},
{
	"tag":"before_bp[0]",
	"file":"main.cpp",
	"line":87,
	"data":{"fc_1": {
"W": np.array([[-0.999583, -0.686062, -0.73111], [-0.98826, 0.562428, -0.476093], [-0.754428, -0.452203, -0.343494]]),
"bias": np.array([-1.00004, -0.65152, -0.605247])
}},
},
{
	"tag":"bp_inputs[0]",
	"file":"main.cpp",
	"line":88,
	"data":np.array([[-0.1, -0.3, 0.4]])
},
{
	"tag":"bp_dy_in[0]",
	"file":"main.cpp",
	"line":89,
	"data":np.array([[0.026588, -2.22372, -8.06308]])
},
{
	"tag":"z2",
	"file":"./layers.h",
	"line":172,
	"data":np.array([[0.026588, -2.22372, -8.06308]])
},
{
	"tag":"dErr_dW",
	"file":"./layers.h",
	"line":177,
	"data":np.array([[-0.0026588, -0.00797639, 0.0106352], [0.222372, 0.667116, -0.889487], [0.806308, 2.41892, -3.22523]])
},
{
	"tag":"after_bp[0]",
	"file":"main.cpp",
	"line":91,
	"data":{"fc_1": {
"W": np.array([[-0.999543, -0.685942, -0.73127], [-0.991596, 0.552421, -0.46275], [-0.766523, -0.488487, -0.295115]]),
"bias": np.array([-1.00044, -0.618164, -0.484301])
}},
},
{
	"tag":"bp_outputs[0]",
	"file":"main.cpp",
	"line":92,
	"data":np.array([[8.25405, 2.37723, 3.80887]])
},
{
	"tag":"before_bp[0]",
	"file":"main.cpp",
	"line":87,
	"data":{"fc_1": {
"W": np.array([[-0.999543, -0.685942, -0.73127], [-0.991596, 0.552421, -0.46275], [-0.766523, -0.488487, -0.295115]]),
"bias": np.array([-1.00044, -0.618164, -0.484301])
}},
},
{
	"tag":"bp_inputs[0]",
	"file":"main.cpp",
	"line":88,
	"data":np.array([[-0.1, -0.3, 0.4]])
},
{
	"tag":"bp_dy_in[0]",
	"file":"main.cpp",
	"line":89,
	"data":np.array([[0.0255828, -2.13966, -7.7583]])
},
{
	"tag":"z2",
	"file":"./layers.h",
	"line":172,
	"data":np.array([[0.0255828, -2.13966, -7.7583]])
},
{
	"tag":"dErr_dW",
	"file":"./layers.h",
	"line":177,
	"data":np.array([[-0.00255828, -0.00767484, 0.0102331], [0.213966, 0.641899, -0.855865], [0.77583, 2.32749, -3.10332]])
},
{
	"tag":"after_bp[0]",
	"file":"main.cpp",
	"line":91,
	"data":{"fc_1": {
"W": np.array([[-0.999505, -0.685827, -0.731423], [-0.994805, 0.542792, -0.449912], [-0.77816, -0.523399, -0.248565]]),
"bias": np.array([-1.00082, -0.586069, -0.367926])
}},
},
{
	"tag":"bp_outputs[0]",
	"file":"main.cpp",
	"line":92,
	"data":np.array([[8.04302, 2.59028, 3.26101]])
},
]

tags = set()
for x in dbg_data:
    tags.add(x["tag"])
tags = list(tags)

tag_groups = {}
for tag in tags:
    tag_groups[tag] = [x["data"] for x in dbg_data if x["tag"] == tag]
