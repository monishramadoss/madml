#pragma once

#include "vknn.h"
#include "../engine/layer.h"


struct sgd_param{
    int total;
    float lr;
    float momentum;
    float dampening;
    float weight_decay;
    bool nestrov;    
};

class sgd : public layer
{
    sgd_param m_param;
public:
    sgd(float lr, float momentum, float dampening, float weight_decay, bool nestrov);
    void forward(tensor& p, tensor& dp, tensor& v);
};

struct adam_param
{
    int total;
    float lr;
    float beta_a;
    float beta_b;
    float eps;
    float weight_decay;
    bool amsgrad;
    int counter;
};


class adam : public layer
{
    adam_param m_param;
public:
    adam(float lr, float beta_a, float beta_b, float eps, float weight_decay, bool amsgrad);
    void forward(int counter, tensor& p, tensor& dp, tensor& m, tensor& r, tensor& m_k_hat, tensor& r_k_hat);
};

struct adagrad_param
{
    int total;
    float lr;
    float eps;
    float lr_decay;
    float weight_decay;
    int counter;
};

class adagrad : public layer
{
    adagrad_param m_param;
public:
    adagrad(float lr, float eps, float lr_decay, float weight_decay);
    void forward(int counter, tensor& p, tensor& dp, tensor& v);
};

struct rmsprop_param
{
    int total;
    float lr;
    float alpha;
    float eps;
    float momentum;
    float weight_decay;
    bool centered;
};

class rmsprop : public layer
{
    rmsprop_param m_param;
public:
    rmsprop(float lr, float alpha, float eps, float weight_decay, float momentum, bool centered);
    void forward(tensor& p, tensor& dp, tensor& v);
};



