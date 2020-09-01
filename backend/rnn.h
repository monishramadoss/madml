#ifndef RNN_H
#define RNN_H
#include <vector>
#include "backend.h"
#include "layer.h"

constexpr int local_sz_x_rnn = 32;
constexpr int local_sz_y_rnn = 32;

namespace layers
{
	namespace rnn
	{
		struct RNN_cell_param
		{
			int total;
			int vocab_size;
			int hidden_size;
			int output_size;
			int input_offset;
			int weight_offset;
			int output_offset;
		};

		class RNNCell : public Base_Layer<RNN_cell_param>
		{
		private:
			void computeGroupCount() override;
			RNN_cell_param m_param;
		public:
			RNNCell(int vocab_size, int hidden_size, int output_size = 0);
			void operator()(std::shared_ptr<tensor>& x, std::shared_ptr<tensor>& h, std::shared_ptr<tensor>& y,
			                std::shared_ptr<tensor>& hn, std::shared_ptr<tensor>& U, std::shared_ptr<tensor>& W,
			                std::shared_ptr<tensor>& V, std::shared_ptr<tensor>& b1,
			                std::shared_ptr<tensor>& b2, int input_offset, int weight_offset, int output_offset);

			void update_weight() override
			{
			};
		};

		class LSTMCell : public Base_Layer<RNN_cell_param>
		{
		private:
			void computeGroupCount() override;
			RNN_cell_param m_param;
		public:
			LSTMCell(int vocab_size, int hidden_size, int output_size);
			void operator()(std::shared_ptr<tensor>& x, std::shared_ptr<tensor>& h, std::shared_ptr<tensor>& c,
			                std::shared_ptr<tensor>& y, std::shared_ptr<tensor>& hn, std::shared_ptr<tensor>& cn,
			                std::shared_ptr<tensor>& U, std::shared_ptr<tensor>& W,
			                std::shared_ptr<tensor>& V, std::shared_ptr<tensor>& b1, std::shared_ptr<tensor>& b2,
			                int input_offset, int weight_offset, int output_offset);

			void update_weight() override
			{
			};
		};

		class GRUCell : public Base_Layer<RNN_cell_param>
		{
		private:
			void computeGroupCount() override;
			RNN_cell_param m_param;
		public:
			GRUCell(int vocab_size, int hidden_size, int output_size);
			void GRUCell::operator()(std::shared_ptr<tensor>& x, std::shared_ptr<tensor>& h, std::shared_ptr<tensor>& y,
			                         std::shared_ptr<tensor>& hn, std::shared_ptr<tensor>& U, std::shared_ptr<tensor>& W,
			                         std::shared_ptr<tensor>& V, std::shared_ptr<tensor>& b1,
			                         std::shared_ptr<tensor>& b2, int input_offset, int weight_offset, int output_offset);

			void update_weight() override
			{
			};
		};
	}
}

#endif
