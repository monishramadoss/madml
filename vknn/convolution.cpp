#include "common.h"
#include "utils.h"
#include "convolution.h"


constexpr int local_sz_x_conv = 16;
constexpr int local_sz_y_conv = 64;

vol2col::vol2col(std::vector<int>& params){
    initVulkanThing(2);
    m_type = "vol2col";
    m_param = {
        0, 1, params[0], params[1], params[2], params[3], params[4], params[5], params[6], params[7], params[8], params[9], params[10],
        params[11], params[12], 0, 0, 0, 0, 0, 0
    };
}


void vol2col::forward(std::shared_ptr<tensor>& col, const std::shared_ptr<tensor>& vol){
    
    if (m_pipeline == nullptr)
    {
        m_param.total = vol->count();
        const int depth = static_cast<int>(vol->getShape()[vol->getShape().size() - 3]);
        const int height = static_cast<int>(vol->getShape()[vol->getShape().size() - 2]);
        const int width = static_cast<int>(vol->getShape()[vol->getShape().size() - 1]);
        m_param.batchsize = vol->getShape()[0];
        m_param.channels = vol->getShape()[1];
        m_param.depth_vol = depth;
        m_param.height_vol = height;
        m_param.width_vol = width;
        m_param.depth_col = (depth + 2 * m_param.pad_d - (m_param.dilation_d * (m_param.kernel_d - 1) - 1)) / m_param.stride_d + 1;
        m_param.height_col = (height + 2 * m_param.pad_h - (m_param.dilation_h * (m_param.kernel_h - 1) - 1)) / m_param.stride_h + 1;
        m_param.width_col = (width + 2 * m_param.pad_w - (m_param.dilation_w * (m_param.kernel_w - 1) - 1)) / m_param.stride_w + 1;
        const int n_out_plane = static_cast<int>(m_param.channels * m_param.kernel_d * m_param.kernel_h * m_param.kernel_w);
        const int output_length = static_cast<int>(m_param.batchsize * m_param.depth_col * m_param.height_col * m_param.width_col);
        
        size_t tmp = m_param.channels;
        tmp *= static_cast<size_t>(m_param.kernel_h);
        tmp *= static_cast<size_t>(m_param.kernel_w);
        tmp *= static_cast<size_t>(m_param.kernel_d);

        m_group_x = static_cast<int>(alignSize(tmp, local_sz_x_conv)) / local_sz_x_conv;
        if (m_group_x > max_compute_work_group_count)
            m_group_x = max_compute_work_group_count;
        m_group_y = static_cast<int>(alignSize(m_param.batchsize, local_sz_y_conv)) / local_sz_y_conv;
        if (m_group_y > max_compute_work_group_count)
            m_group_y = max_compute_work_group_count;
        m_group_z = 1;
        if(!col)
            col = std::make_shared<tensor>(tensor(0., std::vector<int>{n_out_plane, output_length}));

        //createShaderModule(fwd_shader, fwd_codeSize);
        createPipeline(sizeof(vol2col_param));
    }

    bindtensor(vol, 0);
    bindtensor(col, 1);

    recordCommandBuffer(static_cast<void*>(&m_param), sizeof(vol2col_param));
    runCommandBuffer(); 
}

col2vol::col2vol(std::vector<int>& params){
    m_param = {
        0, 1, params[0], params[1], params[2], params[3], params[4], params[5], params[6], params[7], params[8], params[9], params[10],
        params[11], params[12], 0, 0, 0, 0, 0, 0
    };
}

void col2vol::forward(std::shared_ptr<tensor>& vol, const std::shared_ptr<tensor>& col){
    
    if (m_pipeline == nullptr)
    {
        m_param.total = vol->count();
        const int depth = static_cast<int>(vol->getShape()[vol->getShape().size() - 3]);
        const int height = static_cast<int>(vol->getShape()[vol->getShape().size() - 2]);
        const int width = static_cast<int>(vol->getShape()[vol->getShape().size() - 1]);
        m_param.batchsize = vol->getShape()[0];
        m_param.channels = vol->getShape()[1];
        m_param.depth_col = depth;
        m_param.height_col = height;
        m_param.width_col = width;
        m_param.depth_vol = (depth - 1) * m_param.stride_d - 2 * m_param.pad_d + m_param.dilation_d * (m_param.kernel_d - 1) + m_param.pad_d + 1;
        m_param.height_vol = (height - 1) * m_param.stride_h - 2 * m_param.pad_h + m_param.dilation_h * (m_param.kernel_h - 1) + m_param.pad_h + 1;
        m_param.width_vol = (width - 1) * m_param.stride_w - 2 * m_param.pad_w + m_param.dilation_w * (m_param.kernel_w - 1) + m_param.pad_w + 1;
        if(m_param.depth_vol != vol->getShape()[2] || m_param.height_col != vol->getShape()[3] || m_param.width_vol != vol->getShape()[4])
            return;


        const int n_out_plane = static_cast<int>(m_param.channels * m_param.kernel_d * m_param.kernel_h * m_param.kernel_w);
        const int output_length = static_cast<int>(m_param.batchsize * m_param.depth_col * m_param.height_col * m_param.width_col);
        
        size_t tmp = m_param.channels;
        tmp *= static_cast<size_t>(m_param.kernel_h);
        tmp *= static_cast<size_t>(m_param.kernel_w);
        tmp *= static_cast<size_t>(m_param.kernel_d);

        m_group_x = static_cast<int>(alignSize(tmp, local_sz_x_conv)) / local_sz_x_conv;
        if (m_group_x > max_compute_work_group_count)
            m_group_x = max_compute_work_group_count;
        m_group_y = static_cast<int>(alignSize(m_param.batchsize, local_sz_y_conv)) / local_sz_y_conv;
        if (m_group_y > max_compute_work_group_count)
            m_group_y = max_compute_work_group_count;
        m_group_z = 1;
        
        //createShaderModule(fwd_shader, fwd_codeSize);
        createPipeline(sizeof(vol2col_param));
    }

    bindtensor(vol, 0);
    bindtensor(col, 1);

    recordCommandBuffer(static_cast<void*>(&m_param), sizeof(vol2col_param));
    runCommandBuffer(); 
}
