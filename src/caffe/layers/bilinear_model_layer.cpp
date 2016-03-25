#include "caffe/layers/bilinear_model_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void BilinearModelLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
	top[0]->Reshape(num_, channels_* channels_, 1, 1);
}
template <typename Dtype>
void BilinearModelLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
	num_ = bottom[0]->num();
	channels_ = bottom[0]->channels();
	feature_dim_ = bottom[0]->height() * bottom[0]->width();
}

template <typename Dtype>
void BilinearModelLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
	for (int i = 0; i < num_; ++i) {
		int start_bottom_idx = i * channels_ * feature_dim_;
		int start_top_idx = i * channels_ * channels_;
		caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, channels_,
			channels_, feature_dim_, Dtype(1.),
			bottom[0]->cpu_data() + start_bottom_idx,
			bottom[0]->cpu_data() + start_bottom_idx, Dtype(0.),
			top[0]->mutable_cpu_data() + start_top_idx);
	}
} 

template <typename Dtype>
void BilinearModelLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
	for (int i = 0; i < num_; ++i) {
		int start_bottom_idx = i * channels_ * feature_dim_;
		int start_top_idx = i * channels_ * channels_;
		caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channels_,
			feature_dim_, channels_, Dtype(1.),
			top[0]->cpu_diff() + start_top_idx,
			bottom[0]->cpu_data() + start_bottom_idx, Dtype(0.),
			bottom[0]->mutable_cpu_diff() + start_bottom_idx);
		caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, channels_,
			feature_dim_, channels_, Dtype(1.),
			top[0]->cpu_diff() + start_top_idx,
			bottom[0]->cpu_data() + start_bottom_idx, Dtype(1.),
			bottom[0]->mutable_cpu_diff() + start_bottom_idx);
	}
}

#ifdef CPU_ONLY
STUB_GPU(BilinearModelLayer);
#endif

INSTANTIATE_CLASS(BilinearModelLayer);
REGISTER_LAYER_CLASS(BilinearModel);

}
