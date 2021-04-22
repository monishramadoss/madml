#include <functional>
#include <algorithm>
#include <iomanip>

#include "common.h"
#include "utils.h"

tensor::tensor(Format fmt) : m_format(fmt), m_size_in_byte(0)
{
    m_device_id = 0;
    createContext();
    m_device = kDevices[m_device_id];
}

tensor::tensor(char* data, const std::vector<int>& shape, Format fmt) : m_format(fmt), m_size_in_byte(0)
{
    m_device_id = 0;
    createContext();
    m_device = kDevices[m_device_id];
    reshape(data, shape);
}

tensor::tensor(std::vector<float>& c, const std::vector<int>& shape) : m_format(Format::kFormatFp32), m_size_in_byte(0)
{
    m_device_id = 0;
    createContext();
    m_device = kDevices[m_device_id];
    reshape((char*)c.data(), shape);
}

tensor::tensor(float c, const std::vector<int>& shape) : m_format(Format::kFormatFp32), m_size_in_byte(0)
{
    m_device_id = 0;
    createContext();
    m_device = kDevices[m_device_id];
    char* c_arr = init::fill_memory_shape<float>(shape, c);
    reshape(c_arr, shape);
}

void* tensor::map() const
{
    void* p;
    VK_CHECK_RESULT(vkMapMemory(m_device, m_buffer->getVkMemory(), 0, m_size_in_byte, 0, (void**)&p));
    return p;
}

void tensor::unMap() const { vkUnmapMemory(m_device, m_buffer->getVkMemory()); }

Shape tensor::getShape() const { return m_shape; }

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
    if (checkFormat(fmt) && fmt != m_format) m_format = fmt;
    const size_t new_size = shapeCount(m_shape) * elementSize(m_format);
    if (alloc || m_size_in_byte == 0)
        alloc = true;
    m_size_in_byte = new_size;
    if (alloc)
        m_buffer.reset(new buffer(m_device, m_size_in_byte, data));
    else if (data)
    {
        void* p = map();
        memcpy(p, data, m_size_in_byte);
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

void tensor::toDevice(const std::vector<char>& data)
{
    reshape(data.data(), m_shape, true, m_format);
}

void tensor::copyTo(tensor& dst) const
{
    void* p = map();
    dst = dst.reshape(static_cast<const char*>(p), m_shape, false, getFormat());
    unMap();
}

char* tensor::toHost() const
{
    char* host = new char[m_size_in_byte];
    char* device = static_cast<char*>(map());
    std::copy(device, device + m_size_in_byte, host);
    unMap(); // m_buffer.reset();
    return host;
}