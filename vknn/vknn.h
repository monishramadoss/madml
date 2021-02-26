#pragma once

#include <pybind11/pybind11.h>
namespace py = pybind11;

#include "../kernel/kernel.h"

#include "activation.h"
#include "convolution.h"
#include "gemm.h"
#include "loss.h"
#include "math.h"
#include "normalization.h"
#include "pooling.h"
#include "rnn.h"
#include "transform.h"
