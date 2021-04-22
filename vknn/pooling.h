#pragma once

#include "vknn.h"
#include "../engine/layer.h"
/*
 y = y.reshape([self.in_channels * self.batch_size, -1])
        for i in range(self.in_channels * self.batch_size):
            tmp = self.col.host_data[i]
            m_idx = np.argmax(tmp, axis=0)
            self.max_idx[i] = m_idx
            tmp = self.col.host_data[i][m_idx, range(m_idx.size)]
            y[i] = tmp

*/

struct max_reduce_param
{
    uint32_t y_size;
    uint32_t channel_offset;
    uint32_t out_size;
};

class max_reduce : public layer
{
    max_reduce_param m_param;
    bool m_derivative;
public:
    max_reduce(int in_channels, int batch_size, bool derivative);
    void forward(tensor& y, tensor& col, tensor& mdx_idx);
};