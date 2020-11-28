#include "common.h"
#include "utils.h"
#include "rnn.h"

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

    bindtensor(U, 0);
    bindtensor(V, 1);
    bindtensor(W, 2);
    bindtensor(x, 3);
    bindtensor(h, 4);
    bindtensor(b1, 5);
    bindtensor(b2, 6);
    bindtensor(y, 7);
    bindtensor(hn, 8);

    recordCommandBuffer(static_cast<void*>(&m_param), sizeof(RNN_cell_param));
    runCommandBuffer();
}

LSTMCell::LSTMCell(int vocab_size, int hidden_size, int output_size) : Base_Layer(11),
m_param({ 0, vocab_size, hidden_size, output_size, 0, 0 })
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

    bindtensor(U, 0);
    bindtensor(V, 1);
    bindtensor(W, 2);
    bindtensor(x, 3);
    bindtensor(h, 4);
    bindtensor(c, 5);
    bindtensor(b1, 6);
    bindtensor(b2, 7);
    bindtensor(y, 8);
    bindtensor(hn, 9);
    bindtensor(cn, 10);

    recordCommandBuffer(static_cast<void*>(&m_param), sizeof(RNN_cell_param));
    runCommandBuffer();
}

GRUCell::GRUCell(int vocab_size, int hidden_size, int output_size) : Base_Layer(9),
m_param({ 0, vocab_size, hidden_size, output_size, 0, 0 })
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

    bindtensor(U, 0);
    bindtensor(V, 1);
    bindtensor(W, 2);
    bindtensor(x, 3);
    bindtensor(h, 4);
    bindtensor(b1, 5);
    bindtensor(b2, 6);
    bindtensor(y, 7);
    bindtensor(hn, 8);

    recordCommandBuffer(static_cast<void*>(&m_param), sizeof(RNN_cell_param));
    runCommandBuffer();
}

namespace nn
{
    //TODO rnn needs dynamic graph

    RNN::RNN(int vocab_size, int hidden_size, int num_layers, int seq_length, bool bidirectional, int output_size,
        float dropout, bool bias, std::string nonlinearity) :
        m_vocab_size(vocab_size), m_hidden_size(hidden_size), m_num_layers(num_layers), m_directions(1),
        m_output_size(output_size), m_seq_length(seq_length), USE_BIAS(bias)
    {
        
        if (bidirectional)
            m_directions = 2;
        if (output_size == 0)
            m_output_size = vocab_size;

        for (int dir = 0; dir < m_directions; ++dir)
        {
            for (int l = 0; l < m_num_layers; ++l)
            {
                const int input = l == 0 ? m_vocab_size : m_hidden_size;
                const int output = l == m_num_layers - 1 ? m_output_size : m_hidden_size;

                weights_biases.push_back(std::make_shared<tensor>(tensor(1.0, std::vector<int>{m_hidden_size, input})));
                weights_biases.push_back(
                    std::make_shared<tensor>(tensor(1.0, std::vector<int>{m_hidden_size, m_hidden_size})));
                weights_biases.push_back(std::make_shared<tensor>(tensor(1.0, std::vector<int>{output, m_hidden_size})));
                if (USE_BIAS)
                {
                    weights_biases.push_back(std::make_shared<tensor>(tensor(1.0, std::vector<int>{m_hidden_size})));
                    weights_biases.push_back(std::make_shared<tensor>(tensor(1.0, std::vector<int>{output})));
                }
                else
                {
                    weights_biases.push_back(std::make_shared<tensor>(tensor(0.0, std::vector<int>{m_hidden_size})));
                    weights_biases.push_back(std::make_shared<tensor>(tensor(0.0, std::vector<int>{output})));
                }

                cells.push_back(new RNNCell(input, m_hidden_size, output));
            }
        }
    }

    std::tuple<std::shared_ptr<tensor>&, std::shared_ptr<tensor>&> RNN::operator()(const std::shared_ptr<tensor>& x)
    {
        h = std::make_shared<tensor>(tensor(0.0, std::vector<int>{m_seq_length, m_directions, m_hidden_size}));
        return operator()(x, h);
    }

    std::tuple<std::shared_ptr<tensor>&, std::shared_ptr<tensor>&> RNN::operator()(
        const std::shared_ptr<tensor>& x, const std::shared_ptr<tensor>& h)
    {
        this->x = x;
        this->h = h;
        const auto input_shape = x->getShape();
        if (x->getShape().size() == 2)
            x->reShape(std::vector<int>{input_shape[0], 1, m_vocab_size});
        cache.push_back(x);
        cache.push_back(h);

        for (int l = 0; l < m_num_layers; ++l)
        {
            const int output = l == m_num_layers - 1 ? m_output_size : m_hidden_size;
            cache.push_back(std::make_shared<tensor>(tensor(0.0, std::vector<int>{m_seq_length, m_directions, output})));
            cache.push_back(
                std::make_shared<tensor>(tensor(0.0, std::vector<int>{m_seq_length, m_directions, m_hidden_size})));
        }

        for (int dir = 0; dir < m_directions; ++dir)
        {
            for (int i = 0; i < input_shape[0]; ++i)
            {
                for (int l = 0; l < m_num_layers; ++l)
                {
                    const uint64_t weight_bias_idx = static_cast<uint64_t>(m_num_layers) * dir * 5 + static_cast<
                        uint64_t>(l) * 5;
                    const uint64_t cache_idx = static_cast<uint64_t>(l) * 2;
                    const uint64_t direction = dir == 1 ? input_shape[0] - i - 1 : i;

                    uint64_t input_offset = cache[cache_idx]->getShape()[2] * direction;
                    uint64_t weight_offset = direction * m_hidden_size;
                    uint64_t output_offset = direction * cache[2 + cache_idx]->getShape()[2];

                    if (dir == 1)
                    {
                        if (cache_idx != 0)
                            input_offset += static_cast<uint64_t>(m_seq_length) * cache[cache_idx]->getShape()[2];
                        weight_offset += static_cast<uint64_t>(m_hidden_size) * m_seq_length;
                        output_offset += static_cast<uint64_t>(m_seq_length) * cache[2 + cache_idx]->getShape()[2];
                    }
                    const uint64_t cell_idx = static_cast<uint64_t>(m_num_layers) * dir * m_seq_length + static_cast<
                        uint64_t>(m_seq_length) * l + i;
                    cells[cell_idx]->operator()(
                        cache[0 + cache_idx],
                        cache[1 + cache_idx],
                        cache[2 + cache_idx],
                        cache[3 + cache_idx],
                        weights_biases[0 + weight_bias_idx],
                        weights_biases[1 + weight_bias_idx],
                        weights_biases[2 + weight_bias_idx],
                        weights_biases[3 + weight_bias_idx],
                        weights_biases[4 + weight_bias_idx],
                        static_cast<int>(input_offset), static_cast<int>(weight_offset), static_cast<int>(output_offset)
                        );
                }
            }
        }

        return std::forward_as_tuple(cache[cache.size() - 2], cache[cache.size() - 1]);
    }

    int RNN::set_backward()
    {
        if (USE_BIAS)
        {
        }
        else
        {
        }

        return 1;
    }

    void RNN::update_weight()
    {
    }

    LSTM::LSTM(int vocab_size, int hidden_size, int num_layers, int seq_length, bool bidirectional, int output_size,
        float dropout, bool bias, std::string nonlinearity) :
        m_vocab_size(vocab_size), m_hidden_size(hidden_size), m_num_layers(num_layers), m_directions(1),
        m_output_size(output_size), m_seq_length(seq_length), USE_BIAS(bias), nonlinearity_(std::move(nonlinearity))
    {
        
        if (bidirectional)
            m_directions = 2;
        if (output_size == 0)
            m_output_size = vocab_size;

        for (int dir = 0; dir < m_directions; ++dir)
        {
            for (int l = 0; l < m_num_layers; ++l)
            {
                const int input = l == 0 ? m_vocab_size : m_hidden_size;
                const int output = l == m_num_layers - 1 ? m_output_size : m_hidden_size;

                weights_biases.push_back(std::make_shared<tensor>(tensor(1.0, std::vector<int>{m_hidden_size, input, 4})));
                weights_biases.push_back(
                    std::make_shared<tensor>(tensor(1.0, std::vector<int>{m_hidden_size, m_hidden_size, 4})));
                weights_biases.push_back(std::make_shared<tensor>(tensor(1.0, std::vector<int>{output, m_hidden_size, 4})));

                if (USE_BIAS)
                {
                    weights_biases.push_back(std::make_shared<tensor>(tensor(1.0, std::vector<int>{m_hidden_size, 4})));
                    weights_biases.push_back(std::make_shared<tensor>(tensor(1.0, std::vector<int>{output})));
                }
                else
                {
                    weights_biases.push_back(std::make_shared<tensor>(tensor(0.0, std::vector<int>{m_hidden_size, 4})));
                    weights_biases.push_back(std::make_shared<tensor>(tensor(0.0, std::vector<int>{output})));
                }

                cells.push_back(new LSTMCell(input, m_hidden_size, output));
            }
        }
    }

    std::tuple<std::shared_ptr<tensor>&, std::shared_ptr<tensor>&, std::shared_ptr<tensor>&> LSTM::operator()(
        const std::shared_ptr<tensor>& x)
    {
        h = std::make_shared<tensor>(tensor(0.0, std::vector<int>{m_seq_length, m_directions, m_hidden_size}));
        c = std::make_shared<tensor>(tensor(0.0, std::vector<int>{m_seq_length, m_directions, m_hidden_size}));
        return operator()(x, h, c);
    }

    std::tuple<std::shared_ptr<tensor>&, std::shared_ptr<tensor>&, std::shared_ptr<tensor>&> LSTM::operator()(
        const std::shared_ptr<tensor>& x, const std::shared_ptr<tensor>& h, const std::shared_ptr<tensor>& c)
    {
        this->x = x;
        this->h = h;
        this->c = c;
        const auto input_shape = x->getShape();
        if (x->getShape().size() == 2)
            x->reShape(std::vector<int>{input_shape[0], 1, m_vocab_size});
        cache.push_back(x);
        cache.push_back(h);
        cache.push_back(c);

        for (int l = 0; l < m_num_layers; ++l)
        {
            const int output = l == m_num_layers - 1 ? m_output_size : m_hidden_size;
            cache.push_back(std::make_shared<tensor>(tensor(0.0, std::vector<int>{m_seq_length, m_directions, output})));
            cache.push_back(
                std::make_shared<tensor>(tensor(0.0, std::vector<int>{m_seq_length, m_directions, m_hidden_size})));
            cache.push_back(
                std::make_shared<tensor>(tensor(0.0, std::vector<int>{m_seq_length, m_directions, m_hidden_size})));
        }

        for (int dir = 0; dir < m_directions; ++dir)
        {
            for (int i = 0; i < input_shape[0]; ++i)
            {
                for (int l = 0; l < m_num_layers; ++l)
                {
                    const uint64_t weight_bias_idx = static_cast<uint64_t>(m_num_layers) * dir * 5 + static_cast<
                        uint64_t>(l) * 5;
                    const uint64_t cache_idx = static_cast<uint64_t>(l) * 2;
                    const uint64_t direction = dir == 1 ? input_shape[0] - i - 1 : i;

                    uint64_t input_offset = direction * cache[cache_idx]->getShape()[2];
                    uint64_t weight_offset = direction * m_hidden_size;
                    uint64_t output_offset = direction * cache[3 + cache_idx]->getShape()[2];

                    if (dir == 1)
                    {
                        if (cache_idx != 0)
                            input_offset += static_cast<uint64_t>(m_seq_length) * cache[cache_idx]->getShape()[2];
                        weight_offset += static_cast<uint64_t>(m_hidden_size) * m_seq_length;
                        output_offset += static_cast<uint64_t>(m_seq_length) * cache[2 + cache_idx]->getShape()[2];
                    }
                    const uint64_t cell_idx = static_cast<uint64_t>(m_num_layers) * dir * m_seq_length + static_cast<uint64_t>(
                        m_seq_length) * l + i;
                    cells[cell_idx]->operator()(
                        cache[0 + cache_idx],
                        cache[1 + cache_idx],
                        cache[2 + cache_idx],
                        cache[3 + cache_idx],
                        cache[4 + cache_idx],
                        cache[5 + cache_idx],
                        weights_biases[0 + weight_bias_idx],
                        weights_biases[1 + weight_bias_idx],
                        weights_biases[2 + weight_bias_idx],
                        weights_biases[3 + weight_bias_idx],
                        weights_biases[4 + weight_bias_idx],
                        static_cast<int>(input_offset), static_cast<int>(weight_offset), static_cast<int>(output_offset)
                        );
                }
            }
        }

        return std::forward_as_tuple(cache[cache.size() - 3], cache[cache.size() - 2], cache[cache.size() - 1]);
    }

    int LSTM::set_backward()
    {
        if (USE_BIAS)
        {
        }
        else
        {
        }

        return 1;
    }

    void LSTM::update_weight()
    {
    }

    GRU::GRU(int vocab_size, int hidden_size, int num_layers, int seq_length, bool bidirectional, int output_size,
        float dropout, bool bias, std::string nonlinearity) :
        m_vocab_size(vocab_size), m_hidden_size(hidden_size), m_num_layers(num_layers), m_directions(1),
        m_output_size(output_size), m_seq_length(seq_length), USE_BIAS(bias), nonlinearity_(std::move(nonlinearity))
    {
        
        if (bidirectional)
            m_directions = 2;
        if (output_size == 0)
            m_output_size = vocab_size;

        for (int dir = 0; dir < m_directions; ++dir)
        {
            for (int l = 0; l < m_num_layers; ++l)
            {
                const int input = l == 0 ? m_vocab_size : m_hidden_size;
                const int output = l == m_num_layers - 1 ? m_output_size : m_hidden_size;

                weights_biases.push_back(std::make_shared<tensor>(tensor(1.0, std::vector<int>{m_hidden_size, input, 3})));
                weights_biases.push_back(
                    std::make_shared<tensor>(tensor(1.0, std::vector<int>{m_hidden_size, m_hidden_size, 3})));
                weights_biases.push_back(std::make_shared<tensor>(tensor(1.0, std::vector<int>{output, m_hidden_size, 3})));

                if (USE_BIAS)
                {
                    weights_biases.push_back(std::make_shared<tensor>(tensor(1.0, std::vector<int>{m_hidden_size, 3})));
                    weights_biases.push_back(std::make_shared<tensor>(tensor(1.0, std::vector<int>{output})));
                }
                else
                {
                    weights_biases.push_back(std::make_shared<tensor>(tensor(0.0, std::vector<int>{m_hidden_size, 3})));
                    weights_biases.push_back(std::make_shared<tensor>(tensor(0.0, std::vector<int>{output})));
                }

                cells.push_back(new GRUCell(input, m_hidden_size, output));
            }
        }
    }

    std::tuple<std::shared_ptr<tensor>&, std::shared_ptr<tensor>&> GRU::operator()(const std::shared_ptr<tensor>& x)
    {
        h = std::make_shared<tensor>(tensor(0.0, std::vector<int>{m_seq_length, m_directions, m_hidden_size}));
        return operator()(x, h);
    }

    std::tuple<std::shared_ptr<tensor>&, std::shared_ptr<tensor>&> GRU::operator()(
        const std::shared_ptr<tensor>& x, const std::shared_ptr<tensor>& h)
    {
        this->x = x;
        this->h = h;
        const auto input_shape = x->getShape();
        if (x->getShape().size() == 2)
            x->reShape(std::vector<int>{input_shape[0], 1, m_vocab_size});
        cache.push_back(x);
        cache.push_back(h);

        for (int l = 0; l < m_num_layers; ++l)
        {
            const int output = l == m_num_layers - 1 ? m_output_size : m_hidden_size;
            cache.push_back(std::make_shared<tensor>(tensor(0.0, std::vector<int>{m_seq_length, m_directions, output})));
            cache.push_back(
                std::make_shared<tensor>(tensor(0.0, std::vector<int>{m_seq_length, m_directions, m_hidden_size})));
        }

        for (int dir = 0; dir < m_directions; ++dir)
        {
            for (int i = 0; i < input_shape[0]; ++i)
            {
                for (int l = 0; l < m_num_layers; ++l)
                {
                    const uint64_t weight_bias_idx = static_cast<uint64_t>(m_num_layers) * dir * 5 + static_cast<
                        uint64_t>(l) * 5;
                    const uint64_t cache_idx = static_cast<uint64_t>(l) * 2;
                    const uint64_t direction = dir == 1 ? input_shape[0] - i - 1 : i;

                    uint64_t input_offset = direction * cache[cache_idx]->getShape()[2];
                    uint64_t weight_offset = direction * m_hidden_size;
                    uint64_t output_offset = direction * cache[3 + cache_idx]->getShape()[2];

                    if (dir == 1)
                    {
                        if (cache_idx != 0)
                            input_offset += static_cast<uint64_t>(m_seq_length) * cache[cache_idx]->getShape()[2];
                        weight_offset += static_cast<uint64_t>(m_hidden_size) * m_seq_length;
                        output_offset += static_cast<uint64_t>(m_seq_length) * cache[cache_idx + 2]->getShape()[2];
                    }

                    const uint64_t cell_idx = static_cast<uint64_t>(m_num_layers) * dir * m_seq_length + static_cast<
                        uint64_t>(m_seq_length) * l + i;
                    cells[cell_idx]->operator()(
                        cache[0 + cache_idx],
                        cache[1 + cache_idx],
                        cache[2 + cache_idx],
                        cache[3 + cache_idx],
                        weights_biases[0 + weight_bias_idx],
                        weights_biases[1 + weight_bias_idx],
                        weights_biases[2 + weight_bias_idx],
                        weights_biases[3 + weight_bias_idx],
                        weights_biases[4 + weight_bias_idx],
                        static_cast<int>(input_offset), static_cast<int>(weight_offset), static_cast<int>(output_offset)
                        );
                }
            }
        }

        return std::forward_as_tuple(cache[cache.size() - 2], cache[cache.size() - 1]);
    }

    int GRU::set_backward()
    {
        if (USE_BIAS)
        {
        }
        else
        {
        }

        return 1;
    }

    void GRU::update_weight()
    {
    }
}