#ifndef NORM_H
#define NORM_H
#include <vector>
#include "backend.h"
#include "layer.h"

struct norm_param
{
    int total;
    int batchsize;
    int C;
    float eps;
    float momentum;
    int upper_offset;
    int lower_offset;
};

class BatchNormalization : Base_Layer<norm_param>
{
private:
    void computeGroupCount() override;
public:
    explicit BatchNormalization(float eps = 1e-05, float momentum = 0.1, int num_features = -1, bool affine = true);
    std::shared_ptr<tensor>& operator()(const std::shared_ptr<tensor>& x);
    int set_backward() override;
};

class InstanceNormalization : Base_Layer<norm_param>
{
private:
    void computeGroupCount() override;
public:
    explicit InstanceNormalization(float eps = 1e-05, float momentum = 0.1, int num_features = -1, bool affine = true);
    std::shared_ptr<tensor>& operator()(const std::shared_ptr<tensor>& x);
    int set_backward() override;
};

class LayerNormalization : Base_Layer<norm_param>
{
private:
    void computeGroupCount() override;
public:
    explicit LayerNormalization(float eps = 1e-05, float momentum = 0.1, int num_features = -1, bool affine = true);
    std::shared_ptr<tensor>& operator()(const std::shared_ptr<tensor>& x);
    int set_backward() override;
};
#endif