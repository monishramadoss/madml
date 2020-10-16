#ifndef TENSOR_H
#define TENSOR_H
#include <memory>
#include <numeric>
#include <random>
#include <iostream>
#include <vulkan/vulkan.h>
#include "backend.h"

class buffer;

class tensor
{
public:
    tensor(Format fmt = Format::kFormatFp32);
    tensor(char* data, const std::vector<int>& shape, Format fmt = Format::kFormatFp32);
    tensor(float c, const std::vector<int>& shape, Format fmt);
    tensor(double c, const std::vector<int>& shape);
    tensor(std::vector<float>& c, const std::vector<int>& shape);

    void* map() const;
    void unMap() const;
    Shape getShape() const;
    int getId() const;
    int dimNum() const;
    int dimSize(int axis) const;
    int count(int start_axis = 0, int end_axis = -1) const;
    char* toHost();
    tensor reshape(const char* data, const std::vector<int>& shape, bool alloc = false, Format fmt = Format::kFormatInvalid);
    tensor reShape(const std::vector<int>& shape);
    void toDevice(const char* val);
    Format getFormat() const;
    size_t size() const { return size_in_byte; }
    bool isEmpty() const { return size_in_byte == 0; }
    bool onDevice() const { return is_onDevice; }
    void copyTo(tensor dst) const;
    std::shared_ptr<buffer>& getBuffer() { return m_buffer; }
    friend std::ostream& operator<<(std::ostream& os, tensor& dt);
    bool is_trainable() const { return trainable; }
    bool set_trainable() { trainable = true; }
    void update_id();

    std::vector<int> m_shape;

private:

    int id;
    bool trainable = true;
    bool counted = false;
    bool is_onDevice;

    Format format;
    size_t size_in_byte;

    VkDevice m_device;
    std::shared_ptr<buffer> m_buffer;

    static int& get_object_id();
};

#endif

#ifndef INIT_H
#define INIT_H

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
    char* fill_memory_iter(std::vector<int> shape);
    char* normal_distribution_init(std::vector<int> shape, float mean, float std);
    char* uniform_distribution_init(std::vector<int> shape, float min, float max);
    char* xavier_uniform_init(std::vector<int> shape, float gain, float fan_in, float fan_out);
    char* xavier_normal_init(std::vector<int> shape, float gain, float fan_in, float fan_out);
}

#endif
