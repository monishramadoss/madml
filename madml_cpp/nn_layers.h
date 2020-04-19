#ifndef NN
#define NN

#include <vector>
#include "madml.h"
#include "layer.h"

namespace kernel {
	namespace nn {

		class Module {
		protected:
			std::vector<layer*> layers;
			std::vector<tensor*> input;
			std::vector<tensor*> output;
		public:
			virtual std::vector<tensor*> forward(std::vector<tensor*> x) = 0;
			virtual void backward() = 0;
		};
		std::vector<Module*> layers;


		class conv : public Module {
			int kernel_size, num_filters, stride, padding, dialation, padding_type;
		public:
			conv(int kernel_size, int num_filters, bool bias, int stride, int padding, int dialation, int padding_type);

			std::vector<tensor*> forward(std::vector<tensor*> x);
			void backward();
		};

		class dense : public Module {

		public:
			dense(int size, bool bias);
			std::vector<tensor*> forward(std::vector<tensor*> x);
			void backward();
		};

		class RNN : public Module {

		public:
			RNN(int hidden_size, int num_layers, float dropout, bool bidirectional, bool bias);
			std::vector<tensor*> forward(std::vector<tensor*> x);
			void backward();
		};

		class LSTM : public Module {

		public:
			LSTM(int hidden_size, int num_layers, float dropout, bool bidirectional, bool bias);
			std::vector<tensor*> forward(std::vector<tensor*> x);
			void backward();
		};

		class GRU : public Module {

		public:
			GRU(int hidden_size, int num_layers, float dropout, bool bidirectional, bool bias);
			std::vector<tensor*> forward(std::vector<tensor*> x);
			void backward();
		};
	}

}





#endif //!NN