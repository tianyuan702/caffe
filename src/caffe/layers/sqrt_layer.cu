#include "caffe/layers/sqrt_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void SqrtLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
	for (int i = 0; i < bottom[0]->count(0); ++i) {
		Dtype sign_i = caffe_sign<Dtype>(bottom[0]->cpu_data()[i]);
		bottom_sgn_.mutable_cpu_data()[i] = sign_i;
	}
	caffe_gpu_abs<Dtype>(bottom[0]->count(0), bottom[0]->gpu_data(),
				top[0]->mutable_gpu_data());
	caffe_gpu_powx<Dtype>(top[0]->count(0), top[0]->gpu_data(),
				Dtype(0.5), top[0]->mutable_gpu_data());
	caffe_gpu_mul<Dtype>(top[0]->count(0), top[0]->gpu_data(),
				bottom_sgn_.gpu_data(),
				top[0]->mutable_gpu_data());
} 

template <typename Dtype>
void SqrtLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
	for (int i = 0; i < bottom[0]->count(); ++i) {
		if (static_cast<int>(bottom_sgn_.cpu_data()[i]) != 0) {
			Dtype value_i = Dtype(0.5) / top[0]->cpu_data()[i];
			bottom[0]->mutable_cpu_diff()[i] = value_i;
		} else {
			bottom[0]->mutable_cpu_diff()[i] = Dtype(0.);
		}
	}
	caffe_gpu_mul<Dtype>(bottom[0]->count(), bottom[0]->gpu_diff(),
			top[0]->gpu_diff(), bottom[0]->mutable_gpu_diff());
}

INSTANTIATE_LAYER_GPU_FUNCS(SqrtLayer);

}
