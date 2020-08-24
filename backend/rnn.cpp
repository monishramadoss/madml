#include "common.h"
#include "utils.h"
#include "rnn.h"

namespace layers
{
	namespace rnn
	{
		RNNCell::RNNCell(int vocab_size, int hidden_size, int output_size) : Base_Layer(9), m_param({
			 0, vocab_size, hidden_size, output_size, 0, 0
			})
		{
			if (output_size == 0)
				m_param.output_size = vocab_size;

			m_type = "RNNCell";
		}

		void RNNCell::computeGroupCount()
		{
			m_group_x = static_cast<int>(alignSize(m_param.hidden_size, local_sz_x_rnn)) / local_sz_x_rnn;
			if (m_group_x > max_compute_work_group_count)
				m_group_x = max_compute_work_group_count;
			m_group_y = static_cast<int>(alignSize(m_param.output_size, local_sz_y_rnn)) / local_sz_y_rnn;
			if (m_group_y > max_compute_work_group_count)
				m_group_y = max_compute_work_group_count;
			m_group_z = 1;
		}

		void RNNCell::operator()(std::shared_ptr<tensor>& x, std::shared_ptr<tensor>& h, std::shared_ptr<tensor>& y,
			std::shared_ptr<tensor>& hn, std::shared_ptr<tensor>& U, std::shared_ptr<tensor>& W,
			std::shared_ptr<tensor>& V, std::shared_ptr<tensor>& b1,
			std::shared_ptr<tensor>& b2, int input_offset, int weight_offset, int output_offset)
		{
			const auto input_shape = x->getShape(); //seq_len, input_size
			const auto hidden_shape = h->getShape(); //num_layers * num_directions, hidden_size

			m_param.input_offset = input_offset;
			m_param.weight_offset = weight_offset;
			m_param.output_offset = output_offset;

			if (m_pipeline == nullptr)
			{
				computeGroupCount();
				createShaderModule(kernel::shaders::rnnCell_spv, sizeof(kernel::shaders::rnnCell_spv));
				createPipeline(sizeof(RNN_cell_param));
			}

			bindTensor(U, 0);
			bindTensor(V, 1);
			bindTensor(W, 2);
			bindTensor(x, 3);
			bindTensor(h, 4);
			bindTensor(b1, 5);
			bindTensor(b2, 6);
			bindTensor(y, 7);
			bindTensor(hn, 8);

			recordCommandBuffer(static_cast<void*>(&m_param), sizeof(RNN_cell_param));
			runCommandBuffer();

			auto m1 = get_input_id(x->getId());
			auto m2 = get_input_id(h->getId());
		}

		LSTMCell::LSTMCell(int vocab_size, int hidden_size, int output_size) : Base_Layer(11), m_param({
																				   0, vocab_size, hidden_size, output_size,
																				   0, 0
			})
		{
			if (output_size == 0)
				m_param.output_size = vocab_size;

			m_type = "LSTMCell";
		}

		void LSTMCell::computeGroupCount()
		{
			m_group_x = static_cast<int>(alignSize(m_param.hidden_size, local_sz_x_rnn)) / local_sz_x_rnn;
			if (m_group_x > max_compute_work_group_count)
				m_group_x = max_compute_work_group_count;
			m_group_y = static_cast<int>(alignSize(m_param.output_size, local_sz_y_rnn)) / local_sz_y_rnn;
			if (m_group_y > max_compute_work_group_count)
				m_group_y = max_compute_work_group_count;
			m_group_z = 1;
		}

		void LSTMCell::operator()(std::shared_ptr<tensor>& x, std::shared_ptr<tensor>& h, std::shared_ptr<tensor>& c,
			std::shared_ptr<tensor>& y, std::shared_ptr<tensor>& hn, std::shared_ptr<tensor>& cn,
			std::shared_ptr<tensor>& U, std::shared_ptr<tensor>& W,
			std::shared_ptr<tensor>& V, std::shared_ptr<tensor>& b1, std::shared_ptr<tensor>& b2,
			int input_offset, int weight_offset, int output_offset)
		{
			const auto input_shape = x->getShape(); //seq_len, input_size
			const auto hidden_shape = h->getShape(); //num_layers * num_directions, hidden_size
			const auto cell_shape = c->getShape();

			m_param.input_offset = input_offset;
			m_param.weight_offset = weight_offset;
			m_param.output_offset = output_offset;

			if (m_pipeline == nullptr)
			{
				computeGroupCount();
				createShaderModule(kernel::shaders::lstmCell_spv, sizeof(kernel::shaders::lstmCell_spv));
				createPipeline(sizeof(RNN_cell_param));
			}

			bindTensor(U, 0);
			bindTensor(V, 1);
			bindTensor(W, 2);
			bindTensor(x, 3);
			bindTensor(h, 4);
			bindTensor(c, 5);
			bindTensor(b1, 6);
			bindTensor(b2, 7);
			bindTensor(y, 8);
			bindTensor(hn, 9);
			bindTensor(cn, 10);

			recordCommandBuffer(static_cast<void*>(&m_param), sizeof(RNN_cell_param));
			runCommandBuffer();

			auto m1 = get_input_id(x->getId());
			auto m2 = get_input_id(h->getId());
			auto m3 = get_input_id(c->getId());
		}

		GRUCell::GRUCell(int vocab_size, int hidden_size, int output_size) : Base_Layer(9), m_param({
																				 0, vocab_size, hidden_size, output_size, 0,
																				 0
			})
		{
			if (output_size == 0)
				m_param.output_size = vocab_size;

			m_type = "GRUCell";
		}

		void GRUCell::computeGroupCount()
		{
			m_group_x = static_cast<int>(alignSize(m_param.hidden_size, local_sz_x_rnn)) / local_sz_x_rnn;
			if (m_group_x > max_compute_work_group_count)
				m_group_x = max_compute_work_group_count;
			m_group_y = static_cast<int>(alignSize(m_param.output_size, local_sz_y_rnn)) / local_sz_y_rnn;
			if (m_group_y > max_compute_work_group_count)
				m_group_y = max_compute_work_group_count;
			m_group_z = 1;
		}

		void GRUCell::operator()(std::shared_ptr<tensor>& x, std::shared_ptr<tensor>& h, std::shared_ptr<tensor>& y,
			std::shared_ptr<tensor>& hn, std::shared_ptr<tensor>& U, std::shared_ptr<tensor>& W,
			std::shared_ptr<tensor>& V, std::shared_ptr<tensor>& b1,
			std::shared_ptr<tensor>& b2, int input_offset, int weight_offset, int output_offset)
		{
			const auto input_shape = x->getShape(); //seq_len, input_size
			const auto hidden_shape = h->getShape(); //num_layers * num_directions, hidden_size

			m_param.input_offset = input_offset;
			m_param.weight_offset = weight_offset;
			m_param.output_offset = output_offset;

			if (m_pipeline == nullptr)
			{
				computeGroupCount();
				createShaderModule(kernel::shaders::gruCell_spv, sizeof(kernel::shaders::gruCell_spv));
				createPipeline(sizeof(RNN_cell_param));
			}

			bindTensor(U, 0);
			bindTensor(V, 1);
			bindTensor(W, 2);
			bindTensor(x, 3);
			bindTensor(h, 4);
			bindTensor(b1, 5);
			bindTensor(b2, 6);
			bindTensor(y, 7);
			bindTensor(hn, 8);

			recordCommandBuffer(static_cast<void*>(&m_param), sizeof(RNN_cell_param));
			runCommandBuffer();

			auto m1 = get_input_id(x->getId());
			auto m2 = get_input_id(h->getId());
		}
	}
}