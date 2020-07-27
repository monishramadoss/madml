#ifndef RNN_H
#define RNN_H
#include <vector>
#include "madml.h"
#include "layer.h"

namespace kernel
{
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

			class RNNCell : public Base_Layer
			{
			private:
				void computeGroupCount() override;
				RNN_cell_param m_param;
			public:
				RNNCell(int vocab_size, int hidden_size, int output_size = 0);
				void forward(tensor* x, tensor* h, tensor* y, tensor* hn, tensor* U, tensor* W, tensor* V, tensor* b1,
					tensor* b2, int input_offset, int weight_offset, int output_offset);

				void update_weight() override
				{
				};
			};

			class LSTMCell : public Base_Layer
			{
			private:
				void computeGroupCount() override;
				RNN_cell_param m_param;
			public:
				LSTMCell(int vocab_size, int hidden_size, int output_size);
				void forward(tensor* x, tensor* h, tensor* c, tensor* y, tensor* hn, tensor* cn, tensor* U, tensor* W,
					tensor* V, tensor* b1, tensor* b2, int input_offset, int weight_offset, int output_offset);

				void update_weight() override
				{
				};
			};

			class GRUCell : public Base_Layer
			{
			private:
				void computeGroupCount() override;
				RNN_cell_param m_param;
			public:
				GRUCell(int vocab_size, int hidden_size, int output_size);
				void GRUCell::forward(tensor* x, tensor* h, tensor* y, tensor* hn, tensor* U, tensor* W, tensor* V, tensor* b1,
					tensor* b2, int input_offset, int weight_offset, int output_offset);

				void update_weight() override
				{
				};
			};
		}
	}
}

#endif