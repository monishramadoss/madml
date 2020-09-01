#ifndef TEST_H
#define TEST_H
namespace test
{
	template <typename T = float>
	void PrintDiffer(T* data, int size)
	{
		std::map<T, int> diff_freq;
		for (int i = 0; i < size; ++i)
			diff_freq[data[i]] += 1;
		std::cout << "{";
		for (const auto df : diff_freq)
			std::cout << df.first << ": " << df.second << ", ";
		std::cout << "}" << std::endl;
	}

	template <typename T = float>
	void PrintMatrix(T* data, std::vector<int> shape)
	{
		for (int i = 0; i < shape.size(); ++i)
			std::cout << shape[i] << ((shape.size() - 1) == i ? "" : ", ");
		std::cout << std::endl;
		int m_offset = shape[0] * shape[1];
		for (int offset = 0; offset < m_offset; ++offset)
		{
			std::cout << offset << std::endl;
			for (int i = 0; i < shape[shape.size() - 2]; ++i)
			{
				std::cout << "[ ";
				for (int j = 0; j < shape[shape.size() - 1]; ++j)
				{
					std::cout << " " << data[offset * shape[shape.size() - 2] * shape[shape.size() - 1] + i * shape[shape.size() - 1] + j] << ",";
				}
				std::cout << "]" << std::endl;
			}
		}
	}
}

#endif