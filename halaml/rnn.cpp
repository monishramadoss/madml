#include "common.h"
#include "utils.h"
#include "rnn.h"

#define LOCAL_SZ_X 16
#define LOCAL_SZ_Y 64
#define MAX_COMPUTE_WORK_GROUP_COUNT 65535

namespace kernel
{
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
				m_group_x = static_cast<int>(alignSize(m_param.hidden_size, LOCAL_SZ_X)) / LOCAL_SZ_X;
				if (m_group_x > MAX_COMPUTE_WORK_GROUP_COUNT)
					m_group_x = MAX_COMPUTE_WORK_GROUP_COUNT;
				m_group_y = static_cast<int>(alignSize(m_param.output_size, LOCAL_SZ_Y)) / LOCAL_SZ_Y;
				if (m_group_y > MAX_COMPUTE_WORK_GROUP_COUNT)
					m_group_y = MAX_COMPUTE_WORK_GROUP_COUNT;
				m_group_z = 1;
			}

			void RNNCell::forward(std::shared_ptr<tensor>x, std::shared_ptr<tensor>h, std::shared_ptr<tensor>y, std::shared_ptr<tensor>hn, std::shared_ptr<tensor>U, std::shared_ptr<tensor>W, std::shared_ptr<tensor>V, std::shared_ptr<tensor>b1,
				std::shared_ptr<tensor> b2, int input_offset, int weight_offset, int output_offset)
			{
				const auto input_shape = x->getShape(); //seq_len, input_size
				const auto hidden_shape = h->getShape(); //num_layers * num_directions, hidden_size

				m_param.input_offset = input_offset;
				m_param.weight_offset = weight_offset;
				m_param.output_offset = output_offset;

				if (m_pipeline_forward == nullptr)
				{
					computeGroupCount();
					createShaderModuleForward(shaders::rnnCell_spv, sizeof(shaders::rnnCell_spv));
					createPipelineForward(sizeof(RNN_cell_param));
				}

				bindTensor(m_device, *U, 0, m_descriptor_set_forward);
				bindTensor(m_device, *V, 1, m_descriptor_set_forward);
				bindTensor(m_device, *W, 2, m_descriptor_set_forward);
				bindTensor(m_device, *x, 3, m_descriptor_set_forward);
				bindTensor(m_device, *h, 4, m_descriptor_set_forward);
				bindTensor(m_device, *b1, 5, m_descriptor_set_forward);
				bindTensor(m_device, *b2, 6, m_descriptor_set_forward);
				bindTensor(m_device, *y, 7, m_descriptor_set_forward);
				bindTensor(m_device, *hn, 8, m_descriptor_set_forward);

				recordCommandBufferForward(static_cast<void*>(&m_param), sizeof(RNN_cell_param));

				inputs.push_back(x->getId());
				inputs.push_back(h->getId());
				inputs.push_back(U->getId());
				inputs.push_back(W->getId());
				inputs.push_back(V->getId());
				inputs.push_back(b1->getId());
				inputs.push_back(b2->getId());
				outputs.push_back(y->getId());
				outputs.push_back(hn->getId());

				parents.push_back(get_input_id(x->getId()));
				parents.push_back(get_input_id(h->getId()));
			}

			void RNNCell::backward()
			{
			}

			LSTMCell::LSTMCell(int vocab_size, int hidden_size, int output_size) : Base_Layer(11), m_param({
				0, vocab_size, hidden_size, output_size, 0, 0
				})
			{
				if (output_size == 0)
					m_param.output_size = vocab_size;

				m_type = "LSTMCell";
			}

			void LSTMCell::computeGroupCount()
			{
				m_group_x = static_cast<int>(alignSize(m_param.hidden_size, LOCAL_SZ_X)) / LOCAL_SZ_X;
				if (m_group_x > MAX_COMPUTE_WORK_GROUP_COUNT)
					m_group_x = MAX_COMPUTE_WORK_GROUP_COUNT;
				m_group_y = static_cast<int>(alignSize(m_param.output_size, LOCAL_SZ_Y)) / LOCAL_SZ_Y;
				if (m_group_y > MAX_COMPUTE_WORK_GROUP_COUNT)
					m_group_y = MAX_COMPUTE_WORK_GROUP_COUNT;
				m_group_z = 1;
			}

			void LSTMCell::forward(std::shared_ptr<tensor>x, std::shared_ptr<tensor>h, std::shared_ptr<tensor>c, std::shared_ptr<tensor>y, std::shared_ptr<tensor>hn, std::shared_ptr<tensor>cn, std::shared_ptr<tensor>U, std::shared_ptr<tensor>W,
				std::shared_ptr<tensor>V, std::shared_ptr<tensor>b1, std::shared_ptr<tensor>b2, int input_offset, int weight_offset, int output_offset)
			{
				const auto input_shape = x->getShape(); //seq_len, input_size
				const auto hidden_shape = h->getShape(); //num_layers * num_directions, hidden_size
				const auto cell_shape = c->getShape();

				m_param.input_offset = input_offset;
				m_param.weight_offset = weight_offset;
				m_param.output_offset = output_offset;

				if (m_pipeline_forward == nullptr)
				{
					computeGroupCount();
					createShaderModuleForward(shaders::lstmCell_spv, sizeof(shaders::lstmCell_spv));
					createPipelineForward(sizeof(RNN_cell_param));
				}

				bindTensor(m_device, *U, 0, m_descriptor_set_forward);
				bindTensor(m_device, *V, 1, m_descriptor_set_forward);
				bindTensor(m_device, *W, 2, m_descriptor_set_forward);
				bindTensor(m_device, *x, 3, m_descriptor_set_forward);
				bindTensor(m_device, *h, 4, m_descriptor_set_forward);
				bindTensor(m_device, *c, 5, m_descriptor_set_forward);
				bindTensor(m_device, *b1, 6, m_descriptor_set_forward);
				bindTensor(m_device, *b2, 7, m_descriptor_set_forward);
				bindTensor(m_device, *y, 8, m_descriptor_set_forward);
				bindTensor(m_device, *hn, 9, m_descriptor_set_forward);
				bindTensor(m_device, *cn, 10, m_descriptor_set_forward);

				recordCommandBufferForward(static_cast<void*>(&m_param), sizeof(RNN_cell_param));

				inputs.push_back(x->getId());
				inputs.push_back(h->getId());
				inputs.push_back(c->getId());
				inputs.push_back(U->getId());
				inputs.push_back(W->getId());
				inputs.push_back(V->getId());
				inputs.push_back(b1->getId());
				inputs.push_back(b2->getId());
				outputs.push_back(y->getId());
				outputs.push_back(hn->getId());
				outputs.push_back(cn->getId());
				parents.push_back(get_input_id(x->getId()));
				parents.push_back(get_input_id(h->getId()));
			}

			void LSTMCell::backward() {}

			GRUCell::GRUCell(int vocab_size, int hidden_size, int output_size) : Base_Layer(9), m_param({
				0, vocab_size, hidden_size, output_size, 0, 0
				})
			{
				if (output_size == 0)
					m_param.output_size = vocab_size;

				m_type = "GRUCell";
			}

			void GRUCell::computeGroupCount()
			{
				m_group_x = static_cast<int>(alignSize(m_param.hidden_size, LOCAL_SZ_X)) / LOCAL_SZ_X;
				if (m_group_x > MAX_COMPUTE_WORK_GROUP_COUNT)
					m_group_x = MAX_COMPUTE_WORK_GROUP_COUNT;
				m_group_y = static_cast<int>(alignSize(m_param.output_size, LOCAL_SZ_Y)) / LOCAL_SZ_Y;
				if (m_group_y > MAX_COMPUTE_WORK_GROUP_COUNT)
					m_group_y = MAX_COMPUTE_WORK_GROUP_COUNT;
				m_group_z = 1;
			}

			void GRUCell::forward(std::shared_ptr<tensor>x, std::shared_ptr<tensor>h, std::shared_ptr<tensor>y, std::shared_ptr<tensor>hn, std::shared_ptr<tensor>U, std::shared_ptr<tensor>W, std::shared_ptr<tensor>V, std::shared_ptr<tensor>b1,
				std::shared_ptr<tensor>b2, int input_offset, int weight_offset, int output_offset)
			{
				const auto input_shape = x->getShape(); //seq_len, input_size
				const auto hidden_shape = h->getShape(); //num_layers * num_directions, hidden_size

				m_param.input_offset = input_offset;
				m_param.weight_offset = weight_offset;
				m_param.output_offset = output_offset;

				if (m_pipeline_forward == nullptr)
				{
					computeGroupCount();
					createShaderModuleForward(shaders::gruCell_spv, sizeof(shaders::gruCell_spv));
					createPipelineForward(sizeof(RNN_cell_param));
				}

				bindTensor(m_device, *U, 0, m_descriptor_set_forward);
				bindTensor(m_device, *V, 1, m_descriptor_set_forward);
				bindTensor(m_device, *W, 2, m_descriptor_set_forward);
				bindTensor(m_device, *x, 3, m_descriptor_set_forward);
				bindTensor(m_device, *h, 4, m_descriptor_set_forward);
				bindTensor(m_device, *b1, 5, m_descriptor_set_forward);
				bindTensor(m_device, *b2, 6, m_descriptor_set_forward);
				bindTensor(m_device, *y, 7, m_descriptor_set_forward);
				bindTensor(m_device, *hn, 8, m_descriptor_set_forward);

				recordCommandBufferForward(static_cast<void*>(&m_param), sizeof(RNN_cell_param));

				inputs.push_back(x->getId());
				inputs.push_back(h->getId());
				inputs.push_back(U->getId());
				inputs.push_back(W->getId());
				inputs.push_back(V->getId());
				inputs.push_back(b1->getId());
				inputs.push_back(b2->getId());
				outputs.push_back(y->getId());
				parents.push_back(get_input_id(x->getId()));
				parents.push_back(get_input_id(h->getId()));
			}

			void GRUCell::backward() {}
		}
	}
}