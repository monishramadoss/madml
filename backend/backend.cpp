//#include <Python.h>

#include "backend.h"
#include "test.h"
#include <math.h>

PYBIND11_MODULE(backend, m)
{
    m.def("im2col_cpu", &im2col_cpu);
    m.def("col2im_cpu", &col2im_cpu);
    init_tensor(m);
    init_celu(m);
    init_elu(m);
    init_gelu(m);
    init_hardshrink(m);
    init_hardsigmoid(m);
    init_hardswish(m);
    init_hardtanh(m);
    init_leakyrelu(m);
    init_logsigmoid(m);
    init_prelu(m);
    init_relu(m);
    init_rrelu(m);
    init_selu(m);
    init_sigmoid(m);
    init_softplus(m);
    init_softshrink(m);
    init_softsign(m);
    init_tanshrink(m);

    init_abs(m);
    init_ceil(m);
    init_clip(m);
    init_exp(m);
    init_floor(m);
    init_ln(m);
    init_round(m);
    init_sqrt(m);

    init_acos(m);
    init_acosh(m);
    init_asin(m);
    init_asinh(m);
    init_atan(m);
    init_atanh(m);
    init_cos(m);
    init_cosh(m);
    init_sin(m);
    init_sinh(m);
    init_tan(m);
    init_tanh(m);

    init_add(m);
    init_sub(m);
    init_mul(m);
    init_div(m);

    init_mod(m);
    init_pow(m);
    init_min(m);
    init_max(m);

    init_eq(m);
    init_ne(m);
    init_lt(m);
    init_le(m);
    init_gt(m);
    init_ge(m);

    init_gemm(m);
    init_vol2col(m);
    init_col2vol(m);
    //init_transpose(m);

}