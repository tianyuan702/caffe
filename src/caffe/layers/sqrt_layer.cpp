#include "caffe/layers/sqrt_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void SqrtLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
	top[0]->Reshape(bottom[0]->shape());
}

template <typename Dtype>
void SqrtLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
	bottom_sgn_.Reshape(bottom[0]->shape());
}

template <typename Dtype>
void SqrtLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
	for (int i = 0; i < bottom[0]->count(0); ++i) {
		Dtype sign_i = caffe_sign<Dtype>(bottom[0]->cpu_data()[i]);
		bottom_sgn_.mutable_cpu_data()[i] = sign_i;
	}
	caffe_abs<Dtype>(bottom[0]->count(0), bottom[0]->cpu_data(),
				top[0]->mutable_cpu_data());
	caffe_powx<Dtype>(top[0]->count(0), top[0]->cpu_data(),
				Dtype(0.5), top[0]->mutable_cpu_data());
	caffe_mul<Dtype>(top[0]->count(0), top[0]->cpu_data(),
				bottom_sgn_.cpu_data(),
				top[0]->mutable_cpu_data());
} 

template <typename Dtype>
void SqrtLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
	for (int i = 0; i < bottom[0]->count(); ++i) {
		if (static_cast<int>(bottom_sgn_.cpu_data()[i]) != 0) {
			Dtype value_i = Dtype(0.5) / top[0]->cpu_data()[i];
			bottom[0]->mutable_cpu_diff()[i] = value_i;
		} else {
			bottom[0]->mutable_cpu_diff()[i] = Dtype(0.);
		}
	}
	caffe_mul<Dtype>(bottom[0]->count(), bottom[0]->cpu_diff(),
			top[0]->cpu_diff(), bottom[0]->mutable_cpu_diff());
}

#ifdef CPU_ONLY
STUB_GPU(SqrtLayer);
#endif

INSTANTIATE_CLASS(SqrtLayer);
REGISTER_LAYER_CLASS(Sqrt);

}
