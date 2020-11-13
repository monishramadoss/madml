#include "common.h"
#include "utils.h"
#include "gemm.h"

#define TSM 128                     // The tile-size in dvolension M
#define TSN 128                     // The tile-size in dvolension N
#define TSK 16
#define WPTM 8u                     // The amount of work-per-thread in dvolension M
#define WPTN 8u
#define RTSM (TSM/WPTM)     // The reduced tile-size in dvolension M (TSM/WPTM number of threads)
#define RTSN (TSN/WPTN)     // The reduced tile-size in dvolension N (TSN/WPTN number of threads)
#define CEIL_DIV(x,y) (((x) + (y) - 1) / (y))

gemm::gemm(float alpha, float beta, bool use_bias) : Base_Layer<gemm_param>(3)
{
    m_type = "gemm";
    m_param = { 0, 0, alpha, beta, use_bias, 0, 0 };
    bck_shader = kernel::shaders::d_gemm_spv;
    bck_codeSize = sizeof(kernel::shaders::d_gemm_spv);
    fwd_shader = kernel::shaders::gemm_1_spv;
    fwd_codeSize = sizeof(kernel::shaders::gemm_1_spv);
    t = std::make_shared<transpose>(transpose(std::vector<int>{1, 0}));
}

void gemm::computeGroupCount()
{
    m_group_x = static_cast<int>(alignSize(m_param.m, 64)) / 64;
    m_group_y = static_cast<int>(alignSize(m_param.n, 64)) / 64;
    m_group_z = static_cast<int>(alignSize(m_param.batchsize, 1)) / 1;

    if (m_group_x > max_compute_work_group_count)
        m_group_x = max_compute_work_group_count - 1;
    if (m_group_y > max_compute_work_group_count)
        m_group_y = max_compute_work_group_count - 1;
    if (m_group_z > max_compute_work_group_count)
        m_group_z = max_compute_work_group_count - 1;
}

void gemm::operator()(std::shared_ptr<tensor>& _y, const std::shared_ptr<tensor>& _x, const std::shared_ptr<tensor>& _w,
    const std::shared_ptr<tensor>& _b)
{
    if (x->getShape().size() == w->getShape().size() + 1)
    {
        m_param.batchsize = x->getShape()[0];
        m_param.m = x->getShape()[1];
        m_param.k = x->getShape()[2];
        m_param.n = w->getShape()[1];
        if (!y)
            y = std::make_shared<tensor>(tensor(0., std::vector<int>{m_param.batchsize, m_param.m, m_param.n}));
    }
    else
    {
        m_param.batchsize = 1;
        m_param.m = x->getShape()[0];
        m_param.k = x->getShape()[1];
        m_param.n = w->getShape()[1];
        if (!y)
            y = std::make_shared<tensor>(tensor(0., std::vector<int>{m_param.m, m_param.n}));
    }

    if (_b == nullptr)
        b = std::make_shared<tensor>(tensor(0., std::vector<int>{1}));
    if (!_b && m_param.use_bias)
    {
        b = std::make_shared<tensor>(tensor(1., std::vector<int>{m_param.m, m_param.n}));
    }

    if (m_pipeline == nullptr)
    {
        m_param.total = w->count();
        computeGroupCount();
        createShaderModule(fwd_shader, fwd_codeSize);
        createPipeline(sizeof(gemm_param));
    }

    bindTensor(x, 0);
    bindTensor(w, 1);
    bindTensor(b, 2);
    bindTensor(y, 3);

    recordCommandBuffer(static_cast<void*>(&m_param), sizeof(gemm_param));
    runCommandBuffer();
    _y = y;
}

int gemm::set_backward()
{
    // dx = dy * w.T  // MxK = MxN NxK
    // dw = I.T * dy  // KxN = KxM MxN
    // MxK KxN = MxN
    if (!dx)
        dx = std::make_shared<tensor>(tensor(0., x->getShape()));
    if (!dw)
        dw = std::make_shared<tensor>(tensor(0., w->getShape()));
    if (!dy)
        dy = std::make_shared<tensor>(tensor(0., y->getShape()));

    if (derivative->m_pipeline == nullptr)
    {
        m_param.total = x->count();
        derivative->createShaderModule(bck_shader, bck_codeSize);
        derivative->createPipeline(sizeof(gemm_param));
    }

    derivative->bindTensor(x, 0);
    derivative->bindTensor(w, 1);
    derivative->bindTensor(dy, 2);
    derivative->bindTensor(dw, 3);
    derivative->bindTensor(dx, 4);
    derivative->bindTensor(db, 5);
    derivative->recordCommandBuffer(static_cast<void*>(&m_param), sizeof(gemm_param));
    derivative->runCommandBuffer();
    return dy->getId();
}

void init_gemm(py::module& m)
{
    py::class_<gemm>(m, "gemm")
        .def(py::init<float&, float&, bool&>())
        .def("__call__", &gemm::operator());
}