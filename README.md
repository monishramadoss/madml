# MADML
Multi-Architecture design tool for ML algorithms, (MADML for short)
The premise behind this is to create a universal platform for researchers to experiement with Ml algorithms in a familiar envionrment.

# Current state: (Both Fwd & Bck Prop)
    Training possible via CPU (numba/numpy)
    needs currerent queques and async functional calls
    missing Device IDs
    Need to decide on conda or pip installations
    Needs new build system

# Installation

This is currently not possible
# Usage

For mnist run Python madml_mnist.py

For unitTest run 'Python -m unittest test_*.py'
# Reference

Base GPU Implementation: https://github.com/opencv/opencv/tree/master/modules/dnn/src/vkcom

    Needs Optimization Base Python Implementation CPU: https://github.com/pytorch/pytorch/tree/master/torch/nn/modules
    Needs Async and Memory Optimization

# Contributing

I would love help. If anyone is interested feel free to push an issue.
