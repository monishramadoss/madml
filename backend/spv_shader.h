#include <cstdlib>

namespace kernel { 
	namespace shaders {
		extern const unsigned int d_bias_spv[724];
		extern const unsigned int d_gemm_spv[917];
		extern const unsigned int gradient_spv[457];
		extern const unsigned int d_celu_spv[690];
		extern const unsigned int d_elu_spv[497];
		extern const unsigned int d_relu_spv[420];
		extern const unsigned int d_relu6_spv[467];
		extern const unsigned int d_sigmoid_spv[430];
		extern const unsigned int binary_operator_spv[478];
		extern const unsigned int d_batch_normalization_spv[244];
		extern const unsigned int d_instance_normalization_spv[244];
		extern const unsigned int d_layer_normalization_spv[244];
		extern const unsigned int d_abs_spv[426];
		extern const unsigned int d_exp_spv[407];
		extern const unsigned int d_ln_spv[410];
		extern const unsigned int d_sqrt_spv[416];
		extern const unsigned int unary_operator_spv[401];
		extern const unsigned int d_acos_spv[444];
		extern const unsigned int d_acosh_spv[440];
		extern const unsigned int d_asin_spv[440];
		extern const unsigned int d_asinh_spv[440];
		extern const unsigned int d_atan_spv[434];
		extern const unsigned int d_atanh_spv[434];
		extern const unsigned int d_cos_spv[411];
		extern const unsigned int d_cosh_spv[407];
		extern const unsigned int d_sin_spv[441];
		extern const unsigned int d_sinh_spv[407];
		extern const unsigned int d_tan_spv[427];
		extern const unsigned int d_tanh_spv[441];
		extern const unsigned int gemm_1_spv[1246];
		extern const unsigned int gemm_2_spv[3022];
		extern const unsigned int celu_spv[583];
		extern const unsigned int elu_spv[516];
		extern const unsigned int gelu_spv[519];
		extern const unsigned int hardshrink_spv[538];
		extern const unsigned int hardtanh_spv[559];
		extern const unsigned int leakyrelu_spv[473];
		extern const unsigned int logsigmoid_spv[442];
		extern const unsigned int prelu_spv[471];
		extern const unsigned int relu_spv[423];
		extern const unsigned int relu6_spv[434];
		extern const unsigned int selu_spv[516];
		extern const unsigned int sigmoid_spv[425];
		extern const unsigned int softplus_spv[474];
		extern const unsigned int softshrink_spv[586];
		extern const unsigned int softsign_spv[446];
		extern const unsigned int tanhshrink_spv[437];
		extern const unsigned int add_spv[470];
		extern const unsigned int div_spv[470];
		extern const unsigned int equal_spv[480];
		extern const unsigned int greater_eq_spv[480];
		extern const unsigned int greater_than_spv[480];
		extern const unsigned int less_eq_spv[480];
		extern const unsigned int less_than_spv[480];
		extern const unsigned int max_spv[472];
		extern const unsigned int min_spv[472];
		extern const unsigned int mod_spv[470];
		extern const unsigned int mul_spv[470];
		extern const unsigned int nequal_spv[480];
		extern const unsigned int pow_spv[472];
		extern const unsigned int sub_spv[470];
		extern const unsigned int xor_spv[483];
		extern const unsigned int batch_normalization_spv[1701];
		extern const unsigned int instance_normalization_spv[396];
		extern const unsigned int layer_normalization_spv[244];
		extern const unsigned int gruCell_spv[2320];
		extern const unsigned int lstmCell_spv[2671];
		extern const unsigned int rnnCell_spv[1588];
		extern const unsigned int col2vol_spv[2063];
		extern const unsigned int transpose_spv[756];
		extern const unsigned int vol2col_spv[2074];
		extern const unsigned int abs_spv[407];
		extern const unsigned int ceil_spv[407];
		extern const unsigned int clip_spv[459];
		extern const unsigned int exp_spv[407];
		extern const unsigned int floor_spv[407];
		extern const unsigned int ln_spv[407];
		extern const unsigned int round_spv[407];
		extern const unsigned int sqrt_spv[475];
		extern const unsigned int acos_spv[407];
		extern const unsigned int acosh_spv[407];
		extern const unsigned int asin_spv[407];
		extern const unsigned int asinh_spv[407];
		extern const unsigned int atan_spv[407];
		extern const unsigned int atanh_spv[407];
		extern const unsigned int cos_spv[407];
		extern const unsigned int cosh_spv[407];
		extern const unsigned int sin_spv[407];
		extern const unsigned int sinh_spv[407];
		extern const unsigned int tan_spv[407];
		extern const unsigned int tanh_spv[407];
		extern const unsigned int d_MSE_spv[498];
		extern const unsigned int MSE_spv[611];
	}
}