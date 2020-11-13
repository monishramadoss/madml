#ifndef TRANSFORM_H
#define TRANSFORM_H

#include <vector>
#include <utility>
#include "backend.h"
#include "layer.h"

class copy : public Base_Layer<>
{
private:
    void computeGroupCount() override;
public:
    copy();
    std::shared_ptr<tensor>& operator()(const std::shared_ptr<tensor>& x);
};

struct transpose_param
{
    int total;
    int num_axes;
};

class transpose : public Base_Layer<transpose_param>
{
private:
    void computeGroupCount() override;
    std::vector<int> new_shape;
    std::vector<int> old_shape;
    std::vector<int> stride;
    std::vector<int> d_stride;
public:
    transpose(const std::vector<int>& order);
    std::shared_ptr<tensor>& operator()(const std::shared_ptr<tensor>& x);
    int set_backward() override;
};

void init_transpose(py::module& m);

#endif //!TRANSFORM_H
