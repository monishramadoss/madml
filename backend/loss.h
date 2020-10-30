#ifndef LOSS_H
#define LOSS_H

#include "backend.h"
#include "layer.h"

namespace loss
{
    struct loss_param
    {
        int total;
    };

    class Loss : public Base_Layer<loss_param>
    {
    public:
        Loss();
        void hook(const std::shared_ptr<tensor>& x, const std::shared_ptr<tensor>& w);
        void backward();
    };

    class MSE : public Loss
    {
    public:
        MSE();
        void operator()(const std::shared_ptr<tensor>& y_true, const std::shared_ptr<tensor>& y_pred);
    };
}

#endif
