# madml_python

Project for exascale data parallel and model parallel training and infernece pipeline.

## Current state: (Both Fwd & Bck Prop)

- **Convergence on identity and spiral datasets**
- needs currerent queques
- missing Device IDs
- Need to decide on conda or pip installations

## Primary OBj:

- vulkan impl of python fns, followed by performance, device, and nodal optimizations

## Installation

This requires numpy, tqdm, torchvision, tensorboardX to be installed

## Usage

For mnist run `Python madml_mnist.py`

For unitTest run 'Python -m unittest test_*.py'

## Reference

Base GPU Implementation: https://github.com/opencv/opencv/tree/master/modules/dnn/src/vkcom

- Needs Optimization Base Python Implementation CPU: https://github.com/pytorch/pytorch/tree/master/torch/nn/modules
- Needs Async and Memory Optimization

## Contributing

I would love help. If anyone is interested feel free to push an issue. 
