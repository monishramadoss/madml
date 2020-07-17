#include <functional>

#include "common.h"
#include "utils.h"

namespace kernel
{
	tensor::tensor(Format fmt) : size_in_byte(0), format(fmt)
	{
		id = 0;
		createContext();
		if (!counted)
		{
			update_id();
			counted = true;
		}

		m_device = kDevice;
	}

	tensor::tensor(char* data, const std::vector<int>& shape, Format fmt) : size_in_byte(0), format(fmt)
	{
		id = 0;
		createContext();
		if (!counted)
		{
			update_id();
			counted = true;
		}

		m_device = kDevice;
		std::shared_ptr<char> m_data = std::shared_ptr<char>(data);
		reshape(data, shape);
	}

	tensor::tensor(float c, const std::vector<int>& shape, Format fmt) : size_in_byte(0), format(fmt)
	{		
		createContext();
		if (!counted)
		{
			update_id();
			counted = true;
		}

		m_device = kDevice;
		std::shared_ptr<char> data;
		if (fmt == Format::kFormatBool)
			data = std::shared_ptr<char>(fill_memory_shape<int>(shape, static_cast<int>(c)));
		else
			data = std::shared_ptr<char>(fill_memory_shape<float>(shape, c));
		
		reshape(data.get(), shape);
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
		{
			return *this;
		}
				
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

	tensor tensor::reshape(const std::vector<int>& shape)
	{
		const size_t cnt = count();
		const size_t _shape = std::accumulate(std::begin(shape), std::end(shape), 1, std::multiplies<int>());
		if (cnt != _shape)
			std::cout << "SHAPE ERROR" << std::endl;
		if (m_shape != shape) m_shape = shape;
		return *this;
	}

	void tensor::set_to(float val) const
	{
		if (m_device == nullptr)
			return;

		auto* p = static_cast<float*>(map());
		const int cnt = count();
		for (int i = 0; i < cnt; ++i)
			*p++ = val;
		unMap();
	}

	Format tensor::getFormat() const { return format; }

	void tensor::copyTo(tensor dst) const
	{
		void* p = map();
		dst = dst.reshape(static_cast<const char*>(p), m_shape, false, getFormat());
		unMap();
	}

	char* tensor::toHost() const
	{
		char* data = new char[size_in_byte];
		void* p = map();
		memcpy(data, p, size_in_byte);
		unMap();
		return data;
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
}
