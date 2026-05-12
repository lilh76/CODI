REGISTRY = {}

from .mlp_agent import MLPAgent
REGISTRY["mlp"] = MLPAgent

from .rnn_agent import RNNAgent
REGISTRY["rnn"] = RNNAgent