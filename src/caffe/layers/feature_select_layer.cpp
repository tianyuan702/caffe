#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/feature_select_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void FeatureSelectLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
	num_ = bottom[0]->num();
	bottom_channels_ = bottom[0]->channels();
	feature_dim_ = bottom[0]->count(2);
	bottom_data_t_.Reshape(feature_dim_ * bottom_channels_, num_, 1, 1);
	bottom_diff_t_.Reshape(feature_dim_ * bottom_channels_, num_, 1, 1);
	diag_mat_.Reshape(num_, num_, 1, 1);
	for (int i = 0; i < num_; ++i) {
		diag_mat_.mutable_cpu_data()[i * num_ + i] = Dtype(1.);
	}
	tmp_diffs_.Reshape(feature_dim_, num_, 1, 1);
}

template <typename Dtype>
void FeatureSelectLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
	if (aug_features_.size() > 0) {
		int height = bottom[0]->height();
		int width = bottom[0]->width();
		top_channels_ = aug_features_.size();
		top[0]->Reshape(num_, top_channels_, width, height);
		top_data_t_.Reshape(feature_dim_ * top_channels_, num_, 1, 1);
		top_diff_t_.Reshape(feature_dim_ * top_channels_, num_, 1, 1);
	}
}

template <typename Dtype>
void FeatureSelectLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
	/*caffe_set(top[0]->count(), Dtype(1.), top[0]->mutable_cpu_data());
	for (int i = 0; i < num_; ++i) {
		for (int j = 0; j < aug_features_size_; ++j) {
			int top_idx = i * feature_dim_ * aug_features_size_ + j * feature_dim_;
			for (int k = 0; k < aug_features_[j].size(); ++k) {
				int bottom_idx = i * feature_dim_ * channels_ + aug_features_[j][k] * feature_dim_;
				caffe_mul(feature_dim_, top[0]->cpu_data() + top_idx,
						bottom[0]->cpu_data() + bottom_idx,
						top[0]->mutable_cpu_data() + top_idx);
			}
		}		
	}*/
	if (aug_features_.size() > 0) {
		caffe_cpu_gemm(CblasTrans, CblasNoTrans, feature_dim_ * bottom_channels_,
				num_, num_, Dtype(1.), bottom[0]->cpu_data(),
				diag_mat_.cpu_data(), Dtype(0.),
				bottom_data_t_.mutable_cpu_data());

		caffe_set(top_data_t_.count(), Dtype(1.), top_data_t_.mutable_cpu_data());
		for (int i = 0; i < top_channels_; ++i) {
			int top_t_idx = i * feature_dim_ * num_;
			for (int j = 0; j < aug_features_[i].size(); ++j) {
				int bottom_idx = aug_features_[i][j] * feature_dim_ * num_;
				caffe_mul(feature_dim_ * num_, top_data_t_.cpu_data() + top_t_idx,
					bottom_data_t_.cpu_data() + bottom_idx,
					top_data_t_.mutable_cpu_data() + top_t_idx);
			}
		}

		caffe_cpu_gemm(CblasNoTrans, CblasTrans, num_,
					feature_dim_ * top_channels_, num_, Dtype(1.),
					diag_mat_.cpu_data(), top_data_t_.cpu_data(),
					Dtype(0.), top[0]->mutable_cpu_data());
	}
}

template <typename Dtype>
void FeatureSelectLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
	/*caffe_set(bottom[0]->count(), Dtype(0.), bottom[0]->mutable_cpu_diff());
	for (int i = 0; i < num_; ++i) {
		for (int j = 0; j < aug_features_size_; ++j) {
			int top_idx = i * feature_dim_ * aug_features_size_ + j * feature_dim_;
			for (int k = 0; k < aug_features_[j].size(); ++k) {
				int bottom_idx = i * feature_dim_ * channels_ + aug_features_[j][k] * feature_dim_;
				caffe_div(feature_dim_, top[0]->cpu_data() + top_idx,
						bottom[0]->cpu_data() + bottom_idx,
						tmp_diff_.mutable_cpu_data());
				caffe_mul(feature_dim_, top[0]->cpu_diff() + top_idx,
					tmp_diff_.cpu_data(),
					tmp_diff_.mutable_cpu_data());
				caffe_add(feature_dim_, tmp_diff_.cpu_data(),
					bottom[0]->cpu_diff() + bottom_idx,
					bottom[0]->mutable_cpu_diff() + bottom_idx);
			}
		}
	}*/
	if (aug_features_.size() > 0) {
		caffe_set(bottom_diff_t_.count(), Dtype(0.),
			bottom_diff_t_.mutable_cpu_diff());
	
		caffe_cpu_gemm(CblasTrans, CblasNoTrans, feature_dim_ * top_channels_,
					num_, num_,	Dtype(1.), top[0]->cpu_diff(),
					diag_mat_.cpu_data(), Dtype(0.),
					top_diff_t_.mutable_cpu_diff());

		for (int i = 0; i < top_channels_; ++i) {
			for (int j = 0; j < aug_features_[i].size(); ++j) {
				caffe_copy(tmp_diffs_.count(),
						top_diff_t_.cpu_diff() + i * feature_dim_ * num_,
						tmp_diffs_.mutable_cpu_diff());
				for (int k = 0; k < aug_features_[i].size(); ++k) {
					if (k != j) {
						int map_idx = aug_features_[i][k] * feature_dim_ * num_;
						caffe_mul(feature_dim_ * num_,
							bottom_data_t_.cpu_data() + map_idx,
							tmp_diffs_.cpu_diff(), tmp_diffs_.mutable_cpu_diff());
					}
				}
				caffe_cpu_axpby(feature_dim_ * num_, Dtype(1.),
							tmp_diffs_.cpu_diff(), Dtype(1.),
							bottom_diff_t_.mutable_cpu_diff() +
								aug_features_[i][j] * feature_dim_ * num_);
			}
		}
		
		caffe_cpu_gemm(CblasNoTrans, CblasTrans, num_,
					feature_dim_ * bottom_channels_, num_, Dtype(1.),
					diag_mat_.cpu_data(), bottom_diff_t_.cpu_diff(),
					Dtype(0.), bottom[0]->mutable_cpu_diff());
	}
}

#ifdef CPU_ONLY
STUB_GPU(FeatureSelectLayer);
#endif

INSTANTIATE_CLASS(FeatureSelectLayer);
REGISTER_LAYER_CLASS(FeatureSelect);

}  // namespace caffe
