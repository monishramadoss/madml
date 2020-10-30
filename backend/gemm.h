#ifndef MATMUL_H
#define MATMUL_H

#include "backend.h"
#include "layer.h"

struct gemm_param
{
    int total;
    int batchsize;
    float alpha;
    float beta;
    bool use_bias;
    int m;
    int n;
    int k;
};

class gemm : public Base_Layer<gemm_param>
{
private:
    void computeGroupCount() override;
    std::shared_ptr<transpose> t;
public:
    explicit gemm(float alpha, float beta, bool use_bias);
    void operator()(std::shared_ptr<tensor>& y, const std::shared_ptr<tensor>& x, const std::shared_ptr<tensor>& w,
        const std::shared_ptr<tensor>& b = nullptr);

    int set_backward() override;
};

void init_gemm(py::module& m);
#endif
