#include <vector>
#include "backend.h"
#include "layer.h"

#ifndef ACTIVATION_H
#define ACTIVATION_H

struct activation_param 
{
    int total;
    float alpha;
};

class celu : public Base_Layer<activation_param>
{
public:
    std::shared_ptr<tensor> alpha;
    explicit celu(float alpha, bool in_place = false);
    std::shared_ptr<tensor>& operator()(const std::shared_ptr<tensor>& x);
};

class elu : public Base_Layer<activation_param>
{
public:
    explicit elu(float alpha, bool in_place = false);
    std::shared_ptr<tensor>& operator()(const std::shared_ptr<tensor>& x);
};

class gelu : public Base_Layer<activation_param>
{
public:
    explicit gelu(float alpha, bool in_place = false);
    std::shared_ptr<tensor>& operator()(const std::shared_ptr<tensor>& x);
};

class hardshrink : public Base_Layer<activation_param>
{
public:
    explicit hardshrink(float lambda, bool in_place = false);
    std::shared_ptr<tensor>& operator()(const std::shared_ptr<tensor>& x);
};

class hardsigmoid : public Base_Layer<activation_param>
{
public:
    explicit hardsigmoid(bool in_place = false);
    std::shared_ptr<tensor>& operator()(const std::shared_ptr<tensor>& x);
};

class hardswish : public Base_Layer<activation_param>
{
public:
    explicit hardswish(bool in_place = false);
    std::shared_ptr<tensor>& operator()(const std::shared_ptr<tensor>& x);
};

struct two_param
{
    int total;
    float alpha;
    float beta;
};

class hardtanh : public Base_Layer<two_param>
{
public:
    explicit hardtanh(float min_val = -1, float max_val = 1, bool in_place = false);
    std::shared_ptr<tensor>& operator()(const std::shared_ptr<tensor>& x);
};

class leakyrelu : public Base_Layer<activation_param>
{
public:
    explicit leakyrelu(float slope = -0.01, bool in_place = false);
    std::shared_ptr<tensor>& operator()(const std::shared_ptr<tensor>& x);
};

class logsigmoid : public Base_Layer<activation_param>
{
public:
    explicit logsigmoid(float alpha = -0.01, bool in_place = false);
    std::shared_ptr<tensor>& operator()(const std::shared_ptr<tensor>& x);
};

class prelu : public Base_Layer<two_param>
{
public:
    explicit prelu(float alpha = -0.01, bool in_place = false);
    std::shared_ptr<tensor>& operator()(const std::shared_ptr<tensor>& x);
};

class relu : public Base_Layer<activation_param>
{
public:
    explicit relu(bool in_place = false);
    std::shared_ptr<tensor>& operator()(const std::shared_ptr<tensor>& x);
};

class rrelu : public Base_Layer<activation_param>
{
    float min, max;
public:
    explicit rrelu(float lower, float upper, bool in_place = false);
    std::shared_ptr<tensor>& operator()(const std::shared_ptr<tensor>& x);
};

class selu : public Base_Layer<activation_param>
{
public:
    explicit selu(bool in_place = false);
    std::shared_ptr<tensor>& operator()(const std::shared_ptr<tensor>& x);
};

class sigmoid : public Base_Layer<activation_param>
{
public:
    explicit sigmoid(bool in_place = false);
    std::shared_ptr<tensor>& operator()(const std::shared_ptr<tensor>& x);
};

class softplus : public Base_Layer<activation_param>
{
public:
    explicit softplus(float alpha, bool in_place = false);
    std::shared_ptr<tensor>& operator()(const std::shared_ptr<tensor>& x);
};

class softshrink : public Base_Layer<activation_param>
{
public:
    explicit softshrink(float alpha, bool in_place = false);
    std::shared_ptr<tensor>& operator()(const std::shared_ptr<tensor>& x);
};

class softsign : public Base_Layer<activation_param>
{
public:
    explicit softsign(bool in_place = false);
    std::shared_ptr<tensor>& operator()(const std::shared_ptr<tensor>& x);
};

class tanhshrink : public Base_Layer<activation_param>
{
public:
    explicit tanhshrink(bool in_place = false);
    std::shared_ptr<tensor>& operator()(const std::shared_ptr<tensor>& x);
};

void init_celu(py::module& m);

void init_elu(py::module& m);

void init_gelu(py::module& m);

void init_hardshrink(py::module& m);

void init_hardsigmoid(py::module& m);

void init_hardswish(py::module& m);

void init_hardtanh(py::module& m);

void init_leakyrelu(py::module& m);

void init_logsigmoid(py::module& m);

void init_prelu(py::module& m);

void init_relu(py::module& m);

void init_rrelu(py::module& m);

void init_selu(py::module& m);

void init_sigmoid(py::module& m);

void init_softplus(py::module& m);

void init_softshrink(py::module& m);

void init_softsign(py::module& m);

void init_tanshrink(py::module& m);

#endif //!activation
