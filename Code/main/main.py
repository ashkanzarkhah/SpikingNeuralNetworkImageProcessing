##################################################


# Imports
import numpy as np
import matplotlib.pyplot as plt
import torch
from pymonntorch import NeuronGroup, SynapseGroup, NeuronDimension, Recorder
from conex import (
    CorticalLayer,
    CorticalColumn,
    CorticalLayerConnection,
    Synapsis,
    Neocortex,
    InputLayer,
    replicate,
    prioritize_behaviors,
    Port,
    save_structure,
    create_structure_from_dict,
)
from conex.behaviors.neurons import (
    SimpleDendriteStructure,
    SimpleDendriteComputation,
    LIF,
    SpikeTrace,
    NeuronAxon,
    KWTA,
)
from conex.behaviors.synapses import (
    SynapseInit,
    WeightInitializer,
    SimpleDendriticInput,
    SimpleSTDP,
    Conv2dDendriticInput,
    Conv2dSTDP,
    LateralDendriticInput,

)
from conex.helpers.transforms.encoders import Poisson
from conex.helpers.transforms.misc import SqueezeTransform
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import torch.nn.functional as F
from visualisation import visualize_network_structure
from conex.helpers.filters import DoGFilter, GaborFilter


##################################################


# Parameters
DEVICE = "cpu"
DTYPE = torch.float32
DT = 1
NUMBER_OF_ITERATIONS = 100

POISSON_TIME = 30
POISSON_RATIO = 5 / 30
MNIST_ROOT = "MNIST/"

SENSORY_SIZE_HEIGHT = 28
SENSORY_SIZE_WIDTH = 28
SENSORY_TRACE_TAU_S = 2.7

# Layer 1
L1_EXC_DEPTH = 8
L1_EXC_HEIGHT = 24
L1_EXC_WIDTH = 24
L1_EXC_R = 5.0
L1_EXC_THRESHOLD = 0.0
L1_EXC_TAU = 10.0
L1_EXC_V_RESET = 0.0
L1_EXC_V_REST = 0.0
L1_EXC_TRACE_TAU = 10.0

L1_INH_SIZE = 576
L1_INH_R = 5.0
L1_INH_THRESHOLD = 0.0
L1_INH_TAU = 10.0
L1_INH_V_RESET = 0.0
L1_INH_V_REST = 0.0
L1_INH_TRACE_TAU = 10.0

L1_EXC_EXC_MODE = "random"
L1_EXC_EXC_COEF = 1
L1_EXC_INH_MODE = "random"
L1_EXC_INH_COEF = 1
L1_INH_INH_MODE = "random"
L1_INH_INH_COEF = 1
L1_INH_EXC_MODE = "random"
L1_INH_EXC_COEF = 1

L2_EXC_SIZE = 400
L2_INH_SIZE = 100

L1_L2_MODE = "random"
L1_L2_COEF = 1
L1_L2_A_PLUS = 0.01
L1_L2_A_MINUS = 0.002


INP_CC_MODE = "random"
INP_CC_WEIGHT_SHAPE = (8, 1, 5, 5)
INP_CC_COEF = 1
INP_CC_A_PLUS = 0.01
INP_CC_A_MINUS = 0.002


DOG_SIZE = 28
DOG_SIGMA_1 = 2
DOG_SIGMA_2 = 1


GABOR_SIZE = 28
GABOR_LABDA = 10
GABOR_THETA = np.pi / 4
GABOR_SIGMA = 4.0
GABOR_GAMMA = 2.5
##################################################


# TTFS simulator
class TimeToFirstSpikeLIF:
    def __init__(self, max_time=1.0, tau_m=20.0, v_th=1.0, v_reset=0.0):
        self.max_time = max_time
        self.tau_m = tau_m
        self.v_th = v_th
        self.v_reset = v_reset

    def __call__(self, img):
        img = img / 255.0 if img.max() > 1 else img
        img_flat = img.flatten()
        V = torch.zeros_like(img_flat)
        time_steps = int(self.max_time * 1000)
        dt = 1.0
        I = (img_flat + 0.5) * (img_flat + 0.5) * 255
        spike_times = self.max_time * torch.ones_like(img_flat)
        for t in range(time_steps):
            dV = (dt / self.tau_m) * (-V + I)
            V += dV
            spiking_neurons = (V >= self.v_th)
            spike_times[spiking_neurons] = t * dt / 1000.0
            V[spiking_neurons] = self.v_reset
            I[spiking_neurons] = 0
        spike_times = spike_times.reshape(img.shape)
        return spike_times


##################################################


# A class to apply DoGFilter and GaborFilter
class ApplyFilter:
    def __init__(self, filter_matrix):
        self.filter_matrix = filter_matrix.unsqueeze(
            0).unsqueeze(0)

    def __call__(self, img):
        img = img.unsqueeze(0)
        img = F.conv2d(img, self.filter_matrix,
                       padding='same')
        return img.squeeze()


##################################################


# Filters
DoG_CB_filter = DoGFilter(
    size=DOG_SIZE, sigma_1=DOG_SIGMA_1 + 2, sigma_2=DOG_SIGMA_2 + 4)
DoG_CW_filter = DoGFilter(
    size=DOG_SIZE, sigma_1=DOG_SIGMA_2, sigma_2=DOG_SIGMA_1)
GaborFilter_BC = GaborFilter(
    size=GABOR_SIZE, labda=GABOR_LABDA, theta=GABOR_THETA, sigma=GABOR_SIGMA, gamma=GABOR_GAMMA)
GaborFilter_WC = -GaborFilter(
    size=GABOR_SIZE, labda=GABOR_LABDA, theta=GABOR_THETA, sigma=GABOR_SIGMA, gamma=GABOR_GAMMA)


##################################################


# Transformation on the Data
transformation = transforms.Compose(
    [
        transforms.ToTensor(),
        SqueezeTransform(dim=0),
        ApplyFilter(abs(DoG_CB_filter)),
        # TimeToFirstSpikeLIF(max_time=1.0, tau_m=2.0, v_th=60.0, v_reset=0.0)
        Poisson(time_window=POISSON_TIME, ratio=POISSON_RATIO),
    ]
)


##################################################


# Loading the data
dataset = MNIST(root=MNIST_ROOT, train=True,
                download=True, transform=transformation)

dl = DataLoader(dataset, batch_size=16)


##################################################


# Visualizing the data
# data_iter = iter(dl)
# images, labels = next(data_iter)

# Visualize normal images
# fig, axes = plt.subplots(1, 16, figsize=(16, 2))
# for idx, (img, label) in enumerate(zip(images, labels)):
#     axes[idx].imshow(img.squeeze(), cmap='gray')
#     axes[idx].set_title(f'Label: {label}')
#     axes[idx].axis('off')

# Visualize the images with poisson filter
# fig, axes = plt.subplots(3, 10, figsize=(10, 10))
# for idx, img in enumerate(images[0]):
#     axes[idx // 10][idx % 10].imshow(img.squeeze(), cmap='gray')
#     axes[idx // 10][idx % 10].set_title(f'Label: {labels[0]}')
#     axes[idx // 10][idx % 10].axis('off')
# plt.show()


##################################################


# Defining the network
net = Neocortex(dt=DT, device=DEVICE, dtype=DTYPE)


##################################################


# Defining the input layer
input_layer = InputLayer(
    net=net,
    input_dataloader=dl,
    sensory_size=NeuronDimension(
        depth=1, height=SENSORY_SIZE_HEIGHT, width=SENSORY_SIZE_WIDTH
    ),
    sensory_trace=SENSORY_TRACE_TAU_S,
    instance_duration=POISSON_TIME,
    output_ports={"data_out": (None, [("sensory_pop", {})])}
)


##################################################


# Defining first layer's excitory population
pop_exc = NeuronGroup(
    net=net,
    size=NeuronDimension(depth=L1_EXC_DEPTH,
                         height=L1_EXC_HEIGHT, width=L1_EXC_WIDTH),
    behavior=prioritize_behaviors(
        [
            SimpleDendriteStructure(),
            SimpleDendriteComputation(),
            LIF(
                R=L1_EXC_R,
                threshold=L1_EXC_THRESHOLD,
                tau=L1_EXC_TAU,
                v_reset=L1_EXC_V_RESET,
                v_rest=L1_EXC_V_REST,
            ),
            KWTA(k=10),
            SpikeTrace(tau_s=L1_EXC_TRACE_TAU),
            NeuronAxon(),
        ]
    ),
)

# Defining first layer's inhibitory population
pop_inh = NeuronGroup(
    net=net,
    size=L1_INH_SIZE,
    tag="inh",
    behavior=prioritize_behaviors(
        [
            SimpleDendriteStructure(),
            SimpleDendriteComputation(),
            LIF(
                R=L1_INH_R,
                threshold=L1_INH_THRESHOLD,
                tau=L1_INH_TAU,
                v_reset=L1_INH_V_RESET,
                v_rest=L1_INH_V_REST,
            ),
            SpikeTrace(tau_s=L1_INH_TRACE_TAU),
            NeuronAxon(),
        ]
    ),
)

# Defining first layer's excitory population's connection with itself
syn_exc_exc = SynapseGroup(
    net=net,
    src=pop_exc,
    dst=pop_exc,
    tag="Proximal",
    behavior=prioritize_behaviors(
        [
            SynapseInit(),
            WeightInitializer(mode=L1_EXC_EXC_MODE),
            SimpleDendriticInput(current_coef=L1_EXC_EXC_COEF),
        ]
    ),
)

# Defining first layer's excitory population's connection with the inhibitory population
syn_exc_inh = SynapseGroup(
    net=net,
    src=pop_exc,
    dst=pop_inh,
    tag="Proximal",
    behavior=prioritize_behaviors(
        [
            SynapseInit(),
            WeightInitializer(mode=L1_EXC_INH_MODE),
            SimpleDendriticInput(current_coef=L1_EXC_INH_COEF),
        ]
    ),
)

# Defining first layer's inhibitory population's connection with the excitory population
syn_inh_exc = SynapseGroup(
    net=net,
    src=pop_inh,
    dst=pop_exc,
    tag="Proximal,inh",
    behavior=prioritize_behaviors(
        [
            SynapseInit(),
            WeightInitializer(mode=L1_INH_EXC_MODE),
            SimpleDendriticInput(current_coef=L1_INH_EXC_COEF),
        ]
    ),
)

# Defining first layer's inhibitory population's connection with itself
syn_inh_inh = SynapseGroup(
    net=net,
    src=pop_inh,
    dst=pop_inh,
    tag="Proximal,inh",
    behavior=prioritize_behaviors(
        [
            SynapseInit(),
            WeightInitializer(mode=L1_INH_INH_MODE),
            SimpleDendriticInput(current_coef=L1_INH_INH_COEF),
        ]
    ),
)

# Defining the first layer
layer_l1 = CorticalLayer(
    net=net,
    excitatory_neurongroup=pop_exc,
    inhibitory_neurongroup=pop_inh,
    synapsegroups=[syn_exc_exc, syn_exc_inh, syn_inh_inh, syn_inh_exc],
    input_ports={
        "input": (
            None,
            [Port(object=pop_exc, label=None)],
        )
    },
)
##################################################


# Saving first layer's structure to copy it
l1_dict = save_structure(
    layer_l1,
    save_device=True,
    built_structures=None,
    save_structure_tag=True,
    save_behavior_tag=True,
    save_behavior_priority=True,
    all_structures_required=None,
)

# Using first layer structure for the second layer
l2_dict = l1_dict

# Changing the output ports for the second layer
l2_dict["output_ports"] = {"output": (None, [(0, None, None)])}

# Removing neurondimension for the second layer
l2_dict["built_structures"][0]["behavior"] = [
    beh_save
    for beh_save in l2_dict["built_structures"][0]["behavior"]
    if beh_save["key"] != 0
]

# Resizing the populations sizes
l2_dict["built_structures"][0]["size"] = L2_EXC_SIZE
l2_dict["built_structures"][1]["size"] = L2_INH_SIZE

# Defining the second layer
layer_l2 = create_structure_from_dict(
    net=net, structure_dict=l2_dict, built_structures=None
)

# Defining third layer based on the second layer
layer_l3, _ = replicate(layer_l2, net)


##################################################


# Defining the connections between the first two layers
cortical_connection_l1_l2 = CorticalLayerConnection(
    net=net,
    src=layer_l1,
    dst=layer_l2,
    connections=[
        (
            "exc_pop",
            "exc_pop",
            {
                **prioritize_behaviors(
                    [
                        SynapseInit(),
                        WeightInitializer(mode=L1_L2_MODE),
                        SimpleDendriticInput(current_coef=L1_L2_COEF),
                        SimpleSTDP(a_plus=L1_L2_A_PLUS, a_minus=L1_L2_A_MINUS),
                    ]
                ),
                1000: Recorder(variables=['weights']),
            },
            "Proximal"
        )
    ],
)

# Defining the connections between the second two layers based on the previous connection
cortical_connection_l2_l3, _ = replicate(cortical_connection_l1_l2, net)
cortical_connection_l2_l3.connect_src(layer_l2)
cortical_connection_l2_l3.connect_dst(layer_l3)


##################################################


# Defining the cotical column
cc1 = CorticalColumn(
    net=net,
    layers={"l1": layer_l1, "l2": layer_l2, "l3": layer_l3},
    layer_connections=[
        ("l1", "l2", cortical_connection_l1_l2),
        ("l2", "l3", cortical_connection_l2_l3),
    ],
    input_ports={"input": (None, [Port(object=layer_l1, label="input")])},
)


##################################################


# Connecting the input layer into the cortical column
synapsis_input_cc1 = Synapsis(
    net=net,
    src=input_layer,
    dst=cc1,
    input_port="data_out",
    output_port="input",
    synapsis_behavior={
        **prioritize_behaviors(
            [
                SynapseInit(),
                WeightInitializer(
                    mode=INP_CC_MODE, weight_shape=INP_CC_WEIGHT_SHAPE, kernel_shape=INP_CC_WEIGHT_SHAPE),
                LateralDendriticInput(current_coef=INP_CC_COEF),
                Conv2dDendriticInput(current_coef=INP_CC_COEF),
                Conv2dSTDP(a_plus=INP_CC_A_PLUS, a_minus=INP_CC_A_MINUS),
            ]),
        1000: Recorder(variables=['weights']),
    },
    synaptic_tag="Proximal",
)


##################################################


# Simulating the network
net.initialize()
net.simulate_iterations(NUMBER_OF_ITERATIONS)


##################################################


# Visualizing the weights
def weight_threshhold(img, threshold_value=0.5):
    min_val = torch.min(img)
    max_val = torch.max(img)
    scaled = (img - min_val) / (max_val - min_val)
    binary = (scaled >= threshold_value).float()
    return binary


# input layer -> CC
fig, axes = plt.subplots(2, 8, figsize=(16, 2))
for idx in range(8):
    axes[0][idx].imshow(synapsis_input_cc1.synapses[0]['weights']
                        [0][NUMBER_OF_ITERATIONS - 1][idx][0], cmap='gray')
    axes[0][idx].axis('off')
    axes[1][idx].imshow(weight_threshhold(synapsis_input_cc1.synapses[0]['weights']
                                          [0][NUMBER_OF_ITERATIONS - 1][idx][0]), cmap='gray')
    axes[1][idx].axis('off')
plt.show()

# layer 1 -> layer 2
fig, axes = plt.subplots(1, 2)
axes[0].imshow(cortical_connection_l1_l2.synapses[0]
               ['weights'][0][NUMBER_OF_ITERATIONS - 1], cmap='gray')
axes[0].axis('off')
axes[1].imshow(
    weight_threshhold(cortical_connection_l1_l2.synapses[0]['weights'][0][NUMBER_OF_ITERATIONS - 1]), cmap='gray')
axes[1].axis('off')
plt.show()

# layer 2 -> layer 3
fig, axes = plt.subplots(1, 2)
axes[0].imshow(cortical_connection_l2_l3.synapses[0]
               ['weights'][0][NUMBER_OF_ITERATIONS - 1], cmap='gray')
axes[1].imshow(
    weight_threshhold(cortical_connection_l2_l3.synapses[0]['weights'][0][NUMBER_OF_ITERATIONS - 1]), cmap='gray')
plt.show()


##################################################

# Visualizing the network structure

visualize_network_structure(net)
