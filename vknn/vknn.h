#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

namespace py = pybind11;

#include "../kernel/kernel.h"

#include "spv_shader.h"
#include "activation.h"
#include "convolution.h"
#include "gemm.h"
#include "loss.h"
#include "math.h"
#include "normalization.h"
#include "pooling.h"
#include "rnn.h"
#include "transform.h"
