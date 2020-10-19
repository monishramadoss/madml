#ifndef RNN_H
#define RNN_H
#include <vector>
#include "backend.h"
#include "layer.h"

constexpr int local_sz_x_rnn = 32;
constexpr int local_sz_y_rnn = 32;

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

namespace nn
{
    //TODO RNN needs a dynamc seq

    class RNN : public Module
    {
    public:
        RNN(int vocab_size, int hidden_size, int num_layers = 1, int seq_length = 16, bool bidirectional = false,
            int output_size = 0, float dropout = 0.9, bool bias = false, std::string nonlinearity = "tanh");
        std::tuple<std::shared_ptr<tensor>&, std::shared_ptr<tensor>&> operator()(const std::shared_ptr<tensor>& x);
        std::tuple<std::shared_ptr<tensor>&, std::shared_ptr<tensor>&> operator()(
            const std::shared_ptr<tensor>& x, const std::shared_ptr<tensor>& h);
        int set_backward() override;
        void update_weight() override;

    private:
        std::shared_ptr<tensor> x, h;
        int m_vocab_size, m_hidden_size, m_num_layers, m_directions;
        int m_output_size, m_seq_length;
        bool USE_BIAS, bidirectional{};
        std::vector<RNNCell*> cells;
        std::vector<std::shared_ptr<tensor>> weights_biases;
        std::vector<std::shared_ptr<tensor>> cache;
    };

    class LSTM : public Module
    {
    public:
        LSTM(int vocab_size, int hidden_size, int num_layers = 1, int seq_length = 16, bool bidirectional = false,
             int output_size = 0, float dropout = 0.9, bool bias = false, std::string nonlinearity = "tanh");
        std::tuple<std::shared_ptr<tensor>&, std::shared_ptr<tensor>&, std::shared_ptr<tensor>&> operator()(
            const std::shared_ptr<tensor>& x);
        std::tuple<std::shared_ptr<tensor>&, std::shared_ptr<tensor>&, std::shared_ptr<tensor>&> operator()(
            const std::shared_ptr<tensor>& x, const std::shared_ptr<tensor>& h, const std::shared_ptr<tensor>& c);
        int set_backward() override;
        void update_weight() override;

    private:
        std::shared_ptr<tensor> x, h, c;
        int m_vocab_size, m_hidden_size, m_num_layers, m_directions;
        int m_output_size, m_seq_length;
        bool USE_BIAS, bidirectional{};
        std::vector<LSTMCell*> cells;
        std::vector<std::shared_ptr<tensor>> weights_biases;
        std::vector<std::shared_ptr<tensor>> cache;
        std::string nonlinearity_;
    };

    class GRU : public Module
    {
    public:
        GRU(int vocab_size, int hidden_size, int num_layers = 1, int seq_length = 16, bool bidirectional = false,
            int output_size = 0, float dropout = 0.9, bool bias = false, std::string nonlinearity = "tanh");
        std::tuple<std::shared_ptr<tensor>&, std::shared_ptr<tensor>&> operator()(const std::shared_ptr<tensor>& x);
        std::tuple<std::shared_ptr<tensor>&, std::shared_ptr<tensor>&> operator()(
            const std::shared_ptr<tensor>& x, const std::shared_ptr<tensor>& h);
        int set_backward() override;
        void update_weight() override;

    protected:

    private:
        std::shared_ptr<tensor> x, h;
        int m_vocab_size, m_hidden_size, m_num_layers, m_directions;
        int m_output_size, m_seq_length;
        bool USE_BIAS, bidirectional{};
        std::vector<GRUCell*> cells;
        std::vector<std::shared_ptr<tensor>> weights_biases;
        std::vector<std::shared_ptr<tensor>> cache;
        std::string nonlinearity_;
    };
}
#endif
