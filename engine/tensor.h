#pragma once

#include <memory>
#include <numeric>
#include <random>
#include <iostream>
#include <vulkan/vulkan.h>
#include "engine.h"

class buffer;

class tensor
{
public:
    tensor(Format fmt = Format::kFormatFp32);
    tensor(char* data, const std::vector<int>& shape, Format fmt = Format::kFormatFp32);
    tensor(std::vector<float>& c, const std::vector<int>& shape);
    tensor(float c, const std::vector<int>& shape);

    void* map() const;
    void unMap() const;
    Shape getShape() const;
    int dimNum() const;
    int dimSize(int axis) const;
    int count(int start_axis = 0, int end_axis = -1) const;
    char* toHost() const;
    tensor reshape(const char* data, const std::vector<int>& shape, bool alloc = false, Format fmt = Format::kFormatInvalid);
    tensor reShape(const std::vector<int>& shape);
    void toDevice(const std::vector<char>& val);
    Format getFormat() const { return m_format; }
    size_t size() const { return m_size_in_byte; }
    bool isEmpty() const { return m_size_in_byte == 0; }

    void copyTo(tensor& dst) const;
    std::shared_ptr<buffer>& getBuffer() { return m_buffer; }
    std::vector<int> m_shape;

private:

    Format m_format;
    size_t m_size_in_byte;
    VkDevice m_device;
    std::shared_ptr<buffer> m_buffer;
    int m_device_id;
};

namespace init
{
    template <typename dType = float>
    char* fill_memory_shape(std::vector<int> shape, dType c)
    {
        const size_t _shape = std::accumulate(std::begin(shape), std::end(shape), 1, std::multiplies<int>());
        auto ret = new dType[_shape];
        for (int i = 0; i < _shape; ++i)
            ret[i] = reinterpret_cast<dType&>(c);
        return reinterpret_cast<char*>(ret);
    }

    template <typename dType = float>
    char* fill_memory_iter(std::vector<int> shape)
    {
        const size_t _shape = std::accumulate(std::begin(shape), std::end(shape), 1, std::multiplies<int>());
        auto ret = new dType[_shape];
        for (int i = 0; i < _shape; ++i)
            ret[i] = reinterpret_cast<dType&>(i);
        return reinterpret_cast<char*>(ret);
    }
}
