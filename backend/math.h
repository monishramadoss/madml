#ifndef MATH_H
#define MATH_H

#include <vector>
#include <future>
#include "backend.h"
#include "layer.h"

namespace math
{
    // Unary Operators
    class abs : public Base_Layer<>
    {
    public:
        abs(bool in_place = false);
        std::shared_ptr<tensor>& operator()(const std::shared_ptr<tensor>& x);
    };

    class ceil : public Base_Layer<>
    {
    public:
        ceil(bool in_place = false);
        std::shared_ptr<tensor>& operator()(const std::shared_ptr<tensor>& x);
    };

    struct clip_operator_param
    {
        int total;
        float min;
        float max;
    };

    class clip : public Base_Layer<clip_operator_param>
    {
    public:
        clip(float min = 0.0f, float max = 1.0f, bool in_place = false);
        std::shared_ptr<tensor>& operator()(const std::shared_ptr<tensor>& x);
    };

    class exp : public Base_Layer<>
    {
    public:
        exp(bool in_place = false);
        std::shared_ptr<tensor>& operator()(const std::shared_ptr<tensor>& x);
    };

    class floor : public Base_Layer<>
    {
    public:
        floor(bool in_place = false);
        std::shared_ptr<tensor>& operator()(const std::shared_ptr<tensor>& x);
    };

    class ln : public Base_Layer<>
    {
    public:
        ln(bool in_place = false);
        std::shared_ptr<tensor>& operator()(const std::shared_ptr<tensor>& x);
    };

    class round : public Base_Layer<>
    {
    public:
        round(bool in_place = false);
        std::shared_ptr<tensor>& operator()(const std::shared_ptr<tensor>& x);
    };

    class sqrt : public Base_Layer<>
    {
    public:
        sqrt(bool in_place = false);
        std::shared_ptr<tensor>& operator()(const std::shared_ptr<tensor>& x);
    };

    class acos : public Base_Layer<>
    {
    public:
        acos(bool in_place = false);
        std::shared_ptr<tensor>& operator()(const std::shared_ptr<tensor>& x);
    };

    class acosh : public Base_Layer<>
    {
    public:
        acosh(bool in_place = false);
        std::shared_ptr<tensor>& operator()(const std::shared_ptr<tensor>& x);
    };

    class asin : public Base_Layer<>
    {
    public:
        asin(bool in_place = false);
        std::shared_ptr<tensor>& operator()(const std::shared_ptr<tensor>& x);
    };

    class asinh : public Base_Layer<>
    {
    public:
        asinh(bool in_place = false);
        std::shared_ptr<tensor>& operator()(const std::shared_ptr<tensor>& x);
    };

    class atan : public Base_Layer<>
    {
    public:
        atan(bool in_place = false);
        std::shared_ptr<tensor>& operator()(const std::shared_ptr<tensor>& x);
    };

    class atanh : public Base_Layer<>
    {
    public:
        atanh(bool in_place = false);
        std::shared_ptr<tensor>& operator()(const std::shared_ptr<tensor>& x);
    };

    class cos : public Base_Layer<>
    {
    public:
        cos(bool in_place = false);
        std::shared_ptr<tensor>& operator()(const std::shared_ptr<tensor>& x);
    };

    class cosh : public Base_Layer<>
    {
    public:
        cosh(bool in_place = false);
        std::shared_ptr<tensor>& operator()(const std::shared_ptr<tensor>& x);
    };

    class sin : public Base_Layer<>
    {
    public:
        sin(bool in_place = false);
        std::shared_ptr<tensor>& operator()(const std::shared_ptr<tensor>& x);
    };

    class sinh : public Base_Layer<>
    {
    public:
        sinh(bool in_place = false);
        std::shared_ptr<tensor>& operator()(const std::shared_ptr<tensor>& x);
    };

    class tan : public Base_Layer<>
    {
    public:
        tan(bool in_place = false);
        std::shared_ptr<tensor>& operator()(const std::shared_ptr<tensor>& x);
    };

    class tanh : public Base_Layer<>
    {
    public:
        tanh(bool in_place = false);
        std::shared_ptr<tensor>& operator()(const std::shared_ptr<tensor>& x);
    };

    // BINARY OPERATORS
    class add : public Base_Layer<>
    {
    public:
        add(bool in_place = false);
        std::shared_ptr<tensor>& operator()(const std::shared_ptr<tensor>& x, const std::shared_ptr<tensor>& w);
    };

    class sub : public Base_Layer<>
    {
    public:
        sub(bool in_place = false);
        std::shared_ptr<tensor>& operator()(const std::shared_ptr<tensor>& x, const std::shared_ptr<tensor>& w);
    };

    class mul : public Base_Layer<>
    {
    public:
        mul(bool in_place = false);
        std::shared_ptr<tensor>& operator()(const std::shared_ptr<tensor>& x, const std::shared_ptr<tensor>& w);
    };

    class div : public Base_Layer<>
    {
    public:
        div(bool in_place = false);
        std::shared_ptr<tensor>& operator()(const std::shared_ptr<tensor>& x, const std::shared_ptr<tensor>& w);
    };

    class mod : public Base_Layer<>
    {
    public:
        mod(bool in_place = false);
        std::shared_ptr<tensor>& operator()(const std::shared_ptr<tensor>& x, const std::shared_ptr<tensor>& w);
    };

    class pow : public Base_Layer<>
    {
    public:
        pow(bool in_place = false);
        std::shared_ptr<tensor>& operator()(const std::shared_ptr<tensor>& x, const std::shared_ptr<tensor>& w);
    };

    class max : public Base_Layer<>
    {
    public:
        max(bool in_place = false);
        std::shared_ptr<tensor>& operator()(const std::shared_ptr<tensor>& x, const std::shared_ptr<tensor>& w);
    };

    class min : public Base_Layer<>
    {
    public:
        min(bool in_place = false);
        std::shared_ptr<tensor>& operator()(const std::shared_ptr<tensor>& x, const std::shared_ptr<tensor>& w);
    };

    class eq : public Base_Layer<>
    {
    public:
        eq(bool in_place = false);
        std::shared_ptr<tensor>& operator()(const std::shared_ptr<tensor>& x, const std::shared_ptr<tensor>& w);
    };

    class ne : public Base_Layer<>
    {
    public:
        ne(bool in_place = false);
        std::shared_ptr<tensor>& operator()(const std::shared_ptr<tensor>& x, const std::shared_ptr<tensor>& w);
    };

    class lt : public Base_Layer<>
    {
    public:
        lt(bool in_place = false);
        std::shared_ptr<tensor>& operator()(const std::shared_ptr<tensor>& x, const std::shared_ptr<tensor>& w);
    };

    class le : public Base_Layer<>
    {
    public:
        le(bool in_place = false);
        std::shared_ptr<tensor>& operator()(const std::shared_ptr<tensor>& x, const std::shared_ptr<tensor>& w);
    };

    class gt : public Base_Layer<>
    {
    public:
        gt(bool in_place = false);
        std::shared_ptr<tensor>& operator()(const std::shared_ptr<tensor>& x, const std::shared_ptr<tensor>& w);
    };

    class ge : public Base_Layer<>
    {
    public:
        ge(bool in_place = false);
        std::shared_ptr<tensor>& operator()(const std::shared_ptr<tensor>& x, const std::shared_ptr<tensor>& w);
    };

    class xr : public Base_Layer<>
    {
    public:
        xr(bool in_place = false);
        std::shared_ptr<tensor>& operator()(const std::shared_ptr<tensor>& x, const std::shared_ptr<tensor>& w);
    };
}

//unary op
void init_abs(py::module & m);
void init_ceil(py::module & m);
void init_clip(py::module & m);
void init_exp(py::module & m);
void init_floor(py::module & m);
void init_ln(py::module & m);
void init_round(py::module & m);
void init_sqrt(py::module & m);

//unary trig
void init_acos(py::module & m);
void init_acosh(py::module & m);
void init_asin(py::module & m);
void init_asinh(py::module & m);
void init_atan(py::module & m);
void init_atanh(py::module & m);
void init_cos(py::module & m);
void init_cosh(py::module & m);
void init_sin(py::module & m);
void init_sinh(py::module & m);
void init_tan(py::module & m);
void init_tanh(py::module & m);

//binary
void init_add(py::module & m);
void init_sub(py::module & m);
void init_mul(py::module & m);
void init_div(py::module & m);

//binary misc
void init_mod(py::module & m);
void init_pow(py::module & m);
void init_min(py::module & m);
void init_max(py::module & m);

//binary boolean
void init_eq(py::module & m);
void init_ne(py::module & m);
void init_lt(py::module & m);
void init_le(py::module & m);
void init_gt(py::module & m);
void init_ge(py::module & m);

#endif //MATH_H
