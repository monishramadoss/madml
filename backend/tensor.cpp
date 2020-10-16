#include <functional>
#include <algorithm>
#include <iomanip>

#include "common.h"
#include "utils.h"

tensor::tensor(Format fmt) : size_in_byte(0), format(fmt)
{
    id = -1;
    createContext();
    if (!counted)
    {
        counted = true;
    }
    is_onDevice = false;
    m_device = kDevice;
}

tensor::tensor(char* data, const std::vector<int>& shape, Format fmt) : size_in_byte(0), format(fmt)
{
    createContext();
    if (!counted)
    {
        counted = true;
    }

    m_device = kDevice;
    is_onDevice = true;
    reshape(data, shape);
}

tensor::tensor(float c, const std::vector<int>& shape, Format fmt) : size_in_byte(0), format(fmt)
{
    createContext();
    if (!counted)
    {
        counted = true;
    }

    m_device = kDevice;
    std::shared_ptr<char> m_data;
    if (fmt == Format::kFormatBool)
        m_data = std::shared_ptr<char>(init::fill_memory_shape<int>(shape, static_cast<int>(c)));
    else
        m_data = std::shared_ptr<char>(init::fill_memory_shape<float>(shape, c));
    is_onDevice = true;
    reshape(m_data.get(), shape);
}

tensor::tensor(double c, const std::vector<int>& shape) : size_in_byte(0), format(Format::kFormatFp32)
{
    createContext();
    if (!counted)
    {
        counted = true;
    }

    m_device = kDevice;

    auto m_data = std::shared_ptr<char>(init::fill_memory_shape<float>(shape, static_cast<float>(c)));
    is_onDevice = true;
    reshape(m_data.get(), shape);
}

tensor::tensor(std::vector<float>& c, const std::vector<int>& shape) : size_in_byte(0), format(Format::kFormatFp32)
{
    createContext();
    if (!counted)
    {
        counted = true;
    }

    m_device = kDevice;
    float* data = c.data();

    auto m_data = std::shared_ptr<char>(reinterpret_cast<char*>(data));
    is_onDevice = true;
    reshape(m_data.get(), shape);
}

void* tensor::map() const
{
    void* p;
    VK_CHECK_RESULT(vkMapMemory(m_device, m_buffer->getVkMemory(), 0, size_in_byte, 0, (void**)&p));
    return p;
}

void tensor::unMap() const { vkUnmapMemory(m_device, m_buffer->getVkMemory()); }

Shape tensor::getShape() const { return m_shape; }

int tensor::getId() const { return id; }

int tensor::count(const int start_axis, const int end_axis) const
{
    return shapeCount(m_shape, start_axis, end_axis);
}

int tensor::dimSize(const int axis) const
{
    if (axis >= 0 || m_shape.size() > axis)
    {
        return -1;
    }
    return m_shape[axis];
}

int tensor::dimNum() const { return static_cast<int>(m_shape.size()); }

tensor tensor::reshape(const char* data, const std::vector<int>& shape, bool alloc, Format fmt)
{
    if (m_device == nullptr)
        return *this;
    if (m_shape != shape) m_shape = shape;
    if (checkFormat(fmt) && fmt != format) format = fmt;
    const size_t new_size = shapeCount(m_shape) * elementSize(format);
    if (alloc || new_size > size_in_byte)
        alloc = true;
    size_in_byte = new_size;
    if (alloc)
        m_buffer.reset(new buffer(m_device, size_in_byte, data));
    else if (data)
    {
        void* p = map();
        memcpy(p, data, size_in_byte);
        unMap();
    }
    return *this;
}

tensor tensor::reShape(const std::vector<int>& shape)
{
    const size_t _size = std::accumulate(std::begin(shape), std::end(shape), 1, std::multiplies<int>());
    if (count() != _size)
        std::cerr << "SHAPE ERROR" << std::endl;
    if (m_shape != shape) m_shape = shape;
    return *this;
}

void tensor::toDevice(const char* data)
{
    is_onDevice = true;
    reshape(data, m_shape, true, format);
}

Format tensor::getFormat() const { return format; }

void tensor::copyTo(tensor dst) const
{
    void* p = map();
    dst = dst.reshape(static_cast<const char*>(p), m_shape, false, getFormat());
    unMap();
}

char* tensor::toHost()
{
    char* p = (char*)map();
    char* d = new char[size_in_byte];
    std::copy(p, p + size_in_byte, d);
    unMap();

    //m_buffer.reset();
    is_onDevice = false;
    return d;
}

int& tensor::get_object_id()
{
    static int objId;
    return objId;
}

void tensor::update_id()
{
    auto& objId = get_object_id();
    id = objId++;
}

std::ostream& printMatrix_helper(std::ostream& os, float* data, std::vector<int> shape, size_t offset, std::string step, size_t stage)
{
    if (shape.size() == 2)
    {
        size_t m = shape[shape.size() - 2];
        size_t n = shape.back();
        os << "[";
        for (size_t x = 0; x < m; ++x)
        {
            os << (x != 0 ? step + " " : offset == 0 ? "" : "\n") << "[";
            for (size_t y = 0; y < n; ++y)
            {
                os << data[offset + x * n + y] << ((y + 1) == n ? "]" : ", ");
            }
            os << ((x + 1) == m ? "]" : "\n");
        }
        return os;
    }
    else
    {
        std::vector<int>new_shape;

        size_t new_offset = 1;
        for (int i = 1; i < shape.size(); ++i)
        {
            new_shape.push_back(shape[i]);
            new_offset *= shape[i];
        }
        os << "[";
        for (int i = 0; i < shape[0]; ++i)
            printMatrix_helper(os, data, new_shape, offset + i * new_offset, step + " ", stage);
        os << "]";
        return os;
    }
}

std::ostream& operator<<(std::ostream& os, tensor& t)
{
    os << std::fixed << std::setprecision(2);

    auto shape = t.getShape();
    for (auto s : shape)
        os << s << " ";
    os << "\n";
    auto fmt = t.getFormat();
    if (fmt == Format::kFormatFp32)
    {
        float* data = reinterpret_cast<float*>(t.toHost());
        printMatrix_helper(os, data, shape, 0, " ", shape.size());
    }
    if (fmt == Format::kFormatInt32 || fmt == Format::kFormatBool)
    {
        int* data = reinterpret_cast<int*>(t.toHost());
    }
    os << "\n";
    return os;
}

namespace init
{
    char* fill_memory_iter(std::vector<int> shape)
    {
        const size_t _shape = std::accumulate(std::begin(shape), std::end(shape), 1, std::multiplies<int>());
        auto ret = new float[_shape];
#pragma omp for
        for (int i = 0; i < _shape; ++i)
            ret[i] = static_cast<float>(i);
        return reinterpret_cast<char*>(ret);
    }

    char* normal_distribution_init(std::vector<int> shape, float mean, float std)
    {
        std::default_random_engine generator;
        std::normal_distribution<float> distribution(mean, std);

        const size_t _shape = std::accumulate(std::begin(shape), std::end(shape), 1, std::multiplies<int>());
        auto ret = new float[_shape];
#pragma omp for
        for (int i = 0; i < _shape; ++i)
        {
            auto number = distribution(generator);
            ret[i] = number;
        }
        return reinterpret_cast<char*>(ret);
    }

    char* uniform_distribution_init(std::vector<int> shape, float min, float max)
    {
        std::default_random_engine generator;
        std::uniform_real_distribution<float> distribution(min, max);

        const size_t _shape = std::accumulate(std::begin(shape), std::end(shape), 1, std::multiplies<int>());
        auto ret = new float[_shape];
#pragma omp for
        for (int i = 0; i < _shape; ++i)
        {
            auto number = distribution(generator);
            ret[i] = number;
        }
        return reinterpret_cast<char*>(ret);
    }

    char* xavier_uniform_init(std::vector<int> shape, float gain, float fan_in, float fan_out)
    {
        float a = gain * std::sqrtf(6 / (fan_in + fan_out));
        return uniform_distribution_init(shape, -a, a);
    }

    char* xavier_normal_init(std::vector<int> shape, float gain, float fan_in, float fan_out)
    {
        float a = gain * std::sqrtf(2 / (fan_in + fan_out));
        return normal_distribution_init(shape, 0, a * a);
    }
}

void init_tensor(py::module& m)
{
    py::class_<tensor, std::shared_ptr<tensor>>(m, "tensor")
        .def(py::init<std::vector<float>&, const std::vector<int>&>())
        .def("reshape", &tensor::reShape)
        .def_readonly("shape", &tensor::m_shape)
        .def("size", &tensor::count)
        .def("copy", &tensor::copyTo)
        .def("toHost", &tensor::toHost)
        ;
}

//PYBIND11_MODULE(backend, m)
//{
//	py::class_<tensor>(m, "tensor")
//		.def(py::init<float, const std::vector<int>&>());
//}