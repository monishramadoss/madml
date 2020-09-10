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

		int shape_offset = 1;
		std::vector<int> buckets;
		int counter = 0;
		int stage_counter = 0;

		if (shape.size() > 2)
		{
			counter = static_cast<int>(shape.size()) - 3;
			stage_counter = shape[counter];
			for (int i = 0; i < shape.size() - 2; ++i)
			{
				shape_offset *= shape[i];
				buckets.push_back(0);
			}
			buckets.back() = -1;
		}

		int m = shape.size() < 2 ? 1 : shape[shape.size() - 2];
		int n = shape[shape.size() - 1];

		for (int offset = 0; offset < shape_offset; ++offset)
		{
			if (shape.size() > 2)
			{
				buckets[shape.size() - 3]++;
				if (offset >= stage_counter)
				{
					if (counter > 0)
						counter--;
					for (int c = counter; c < buckets.size(); ++c)
						buckets[c] = 0;
					buckets[counter]++;
					stage_counter *= shape[counter];
				}

				for (size_t b = 0; b < buckets.size(); ++b)
					std::cout << buckets[b] << ((b + 1) == buckets.size() ? "\n" : ":");
			}

			for (int i = 0; i < m; ++i)
			{
				std::cout << "[ ";
				for (int j = 0; j < n; ++j)
				{
					std::cout << " " << data[offset * m * n + i * n + j] << ((j + 1) == n ? " ]\n" : ", ");
				}
			}
			std::cout << std::endl;
		}
	}
}
#endif