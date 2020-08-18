#ifndef HALMAL_H
#define HALMAL_H

#include <vector>
#include <pybind11/pybind11.h>
namespace py = pybind11;

enum class Format
{
	kFormatInvalid = -1,
	kFormatFp16 = 0,
	kFormatFp32 = 1,
	kFormatFp64 = 2,
	kFormatInt8 = 3,
	kFormatInt16 = 4,
	kFormatInt32 = 5,
	kFormatInt64 = 6,
	kFormatUInt8 = 7,
	kFormatBool = 8,
	kFormatNum = -1
};

enum FusedActivationType { kNone, kRelu, kRelu1, kRelu6, kActivationNum };

typedef std::vector<int> Shape;
bool isAvailable();

//TODO need work on deallocations
//TODO work on static members
//TODO worker threads for first hand latency

#include "tensor.h"
#include "buffer.h"
#include "layer.h"

#include "math.h"
#include "matmul.h"
#include "activation.h"
#include "transform.h"
#include "rnn.h"
#include "nn_layers.h"

#endif