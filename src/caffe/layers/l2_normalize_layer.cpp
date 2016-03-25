#include "caffe/layers/l2_normalize_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "math.h"

namespace caffe {

template <typename Dtype>
void L2NormalizeLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
	top[0]->Reshape(bottom[0]->shape());
}

template <typename Dtype>
void L2NormalizeLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
	top[0]->Reshape(bottom[0]->shape());
	bottom_norm_.Reshape(bottom[0]->num(), 1, 1, 1);
	num_ = bottom[0]->num();
	feature_dim_ = bottom[0]->count(0) / num_;
}

template <typename Dtype>
void L2NormalizeLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
	for (int i = 0; i < num_; ++i) {
		int start_idx = i * feature_dim_;
		Dtype norm_i = sqrt(caffe_cpu_dot<Dtype>(feature_dim_,
				bottom[0]->cpu_data() + start_idx,
				bottom[0]->cpu_data() + start_idx));
		bottom_norm_.mutable_cpu_data()[i] = norm_i;
		caffe_cpu_axpby<Dtype>(feature_dim_, Dtype(1.) / norm_i,
				bottom[0]->cpu_data() + start_idx, Dtype(0.),
				top[0]->mutable_cpu_data() + start_idx);
	}
} 

template <typename Dtype>
void L2NormalizeLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
	for (int i = 0; i < num_; ++i) {
		int start_idx = i * feature_dim_;
		Dtype norm_i = bottom_norm_.cpu_data()[i];
		Dtype norm_i_inv = Dtype(1.) / norm_i;
		Dtype norm_i_tri = pow(norm_i, 3);
		Dtype scal = -caffe_cpu_dot<Dtype>(feature_dim_,
				bottom[0]->cpu_data() + start_idx,
				top[0]->cpu_diff() + start_idx) / norm_i_tri;
		caffe_cpu_axpby<Dtype>(feature_dim_, norm_i_inv,
				top[0]->cpu_diff() + start_idx, Dtype(0.),
				bottom[0]->mutable_cpu_diff() + start_idx);
		caffe_cpu_axpby<Dtype>(feature_dim_, scal,
				bottom[0]->cpu_data() + start_idx, Dtype(1.),
				bottom[0]->mutable_cpu_diff() + start_idx);
	}
}

#ifdef CPU_ONLY
STUB_GPU(L2NormalizeLayer);
#endif

INSTANTIATE_CLASS(L2NormalizeLayer);
REGISTER_LAYER_CLASS(L2Normalize);

}
