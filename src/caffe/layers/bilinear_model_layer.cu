#include "caffe/layers/bilinear_model_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void BilinearModelLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
	for (int i = 0; i < num_; ++i) {
		int start_bottom_idx = i * channels_ * feature_dim_;
		int start_top_idx = i * channels_ * channels_;
		caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, channels_,
			channels_, feature_dim_, Dtype(1.),
			bottom[0]->gpu_data() + start_bottom_idx,
			bottom[0]->gpu_data() + start_bottom_idx, Dtype(0.),
			top[0]->mutable_gpu_data() + start_top_idx);
	}
} 

template <typename Dtype>
void BilinearModelLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
	for (int i = 0; i < num_; ++i) {
		int start_bottom_idx = i * channels_ * feature_dim_;
		int start_top_idx = i * channels_ * channels_;
		caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channels_,
			feature_dim_, channels_, Dtype(1.),
			top[0]->gpu_diff() + start_top_idx,
			bottom[0]->gpu_data() + start_bottom_idx, Dtype(0.),
			bottom[0]->mutable_gpu_diff() + start_bottom_idx);
		caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, channels_,
			feature_dim_, channels_, Dtype(1.),
			top[0]->gpu_diff() + start_top_idx,
			bottom[0]->gpu_data() + start_bottom_idx, Dtype(1.),
			bottom[0]->mutable_gpu_diff() + start_bottom_idx);
	}
}

INSTANTIATE_LAYER_GPU_FUNCS(BilinearModelLayer);

}
