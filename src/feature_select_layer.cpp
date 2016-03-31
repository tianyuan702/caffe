#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/feature_select_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void FeatureSelectLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
	num_ = bottom[0]->num();
	channels_ = bottom[0]->channels();
	height_ = bottom[0]->height();
	width_ = bottom[0]->width();
	feature_dim_ = bottom[0]->count(2);
	bottom_data_t_.Reshape(feature_dim_ * channels_, num_, 1, 1);
	bottom_diff_t_.Reshape(feature_dim_ * channels_, num_, 1, 1);
	diag_mat_.Reshape(num_, num_, 1, 1);
	for (int i = 0; i < num_; ++i) {
		diag_mat_.mutable_cpu_data()[i * num_ + i] = Dtype(1.);
	}
	tmp_diffs_.Reshape(feature_dim_, num_, 1, 1);
	tmp_diff_.Reshape(feature_dim_, 1, 1, 1);
	tmp_maps_.Reshape(feature_dim_, num_, 1, 1);
}

template <typename Dtype>
void FeatureSelectLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
	aug_features_size_ = aug_features_.size();
	top[0]->Reshape(feature_dim_ * aug_features_size_, num_, 1, 1);
	//top[0]->Reshape(num_, aug_features_size_, width_, height_);
	//top_t_.Reshape(feature_dim_ * aug_features_size_, num_, 1, 1);
	//caffe_set(top_t_.count(), Dtype(0.), top_data_t_.mutable_cpu_data());
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

	caffe_cpu_gemm(CblasTrans, CblasNoTrans, feature_dim_ * channels_, num_, num_,
				Dtype(1.), bottom[0]->cpu_data(), diag_mat_.cpu_data(), Dtype(0.),
				bottom_data_t_.mutable_cpu_data());

	//caffe_set(top_t_.count(), Dtype(1.), top_t_.mutable_cpu_data());
	for (int i = 0; i < aug_features_size_; ++i) {
		int top_t_idx = i * feature_dim_ * num_;
		for (int j = 0; j < aug_features_[i].size(); ++j) {
			int bottom_idx = aug_features_[i][j] * feature_dim_ * num_;
			caffe_cpu_gemm(CblasNoTrans, CblasNoTrans, feature_dim_, num_, num_,
						Dtype(1.), bottom_data_t_.cpu_data() + bottom_idx,
						diag_mat_.cpu_data(), Dtype(0.), tmp_maps_.mutable_cpu_data());
			/*caffe_mul(feature_dim_ * num_, top_t_.cpu_data() + top_t_idx,
					tmp_maps_.cpu_data(), top_t_.mutable_cpu_data() + top_t_idx);*/
			caffe_mul(feature_dim_ * num_, top[0]->cpu_data() + top_t_idx,
					tmp_maps_.cpu_data(), top[0]->mutable_cpu_data() + top_t_idx);
		}
	}

	/*caffe_cpu_gemm(CblasNoTrans, CblasTrans, num_, feature_dim_ * aug_features_size_,
				num_, Dtype(1.), diag_mat_.cpu_data(), top_t_.cpu_data(),
				Dtype(0.), top[0]->mutable_cpu_data());*/
	caffe_cpu_gemm(CblasNoTrans, CblasTrans, num_, feature_dim_ * aug_features_size_,
				num_, Dtype(1.), diag_mat_.cpu_data(), top[0]->cpu_data(),
				Dtype(0.), top[0]->mutable_cpu_data());
	top[0]->Reshape(num_, aug_features_size_, height_, width_);
}

template <typename Dtype>
void FeatureSelectLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
	caffe_set(bottom[0]->count(), Dtype(0.), bottom[0]->mutable_cpu_diff());
	for (int i = 0; i < top[0]->count(); ++i) {
		std::cout << top[0]->cpu_data()[i] << " ";
	}
	std::cout << std::endl;
	for (int i = 0; i < num_; ++i) {
		for (int j = 0; j < aug_features_size_; ++j) {
			int top_idx = i * feature_dim_ * aug_features_size_ + j * feature_dim_;
			for (int k = 0; k < aug_features_[j].size(); ++k) {
				int bottom_idx = i * feature_dim_ * channels_ + aug_features_[j][k] * feature_dim_;
				std::cout << top[0]->cpu_data()[top_idx] << std::endl;
				caffe_div(feature_dim_, top[0]->cpu_data() + top_idx,
						bottom[0]->cpu_data() + bottom_idx,
						tmp_diff_.mutable_cpu_data());

				/*std::cout << tmp_diff_.shape()[0] << " " << tmp_diff_.shape()[1] << " "
						  << tmp_diff_.shape()[2] << " " << tmp_diff_.shape()[3] << " "  << feature_dim_ << std::endl;
				for (int w = 0; w < tmp_diff_.count(); ++w) {
					std::cout << tmp_diff_.count() << "_" << tmp_diff_.cpu_data()[w] << " ";
				}
				std::cout << std::endl;*/
				caffe_mul(feature_dim_, top[0]->cpu_diff() + top_idx,
					tmp_diff_.cpu_data(),
					tmp_diff_.mutable_cpu_data());
				caffe_add(feature_dim_, tmp_diff_.cpu_data(),
					bottom[0]->cpu_diff() + bottom_idx,
					bottom[0]->mutable_cpu_diff() + bottom_idx);
			}
		}
	}
	/*top[0]->Reshape(num_, feature_dim_ * aug_features_size_, 1, 1);
	caffe_cpu_gemm(CblasTrans, CblasNoTrans, feature_dim_ * aug_features_size_, num_, num_,
				Dtype(1.), top[0]->cpu_diff(), diag_mat_.cpu_data(),
				Dtype(1.0), top[0]->mutable_cpu_diff());
	top[0]->Reshape(feature_dim_ * aug_features_size_, num_, 1, 1);
	caffe_set(bottom_diff_t_.count(), Dtype(0.), bottom_diff_t_.mutable_cpu_data());
	for (int i = 0; i < aug_features_size_; ++i) {
		for (int j = 0; j < aug_features_[i].size(); ++j) {
			caffe_copy(tmp_diffs_.count(), top[0]->cpu_diff() + i * feature_dim_ * num_, tmp_diffs_.mutable_cpu_diff());
			for (int k = 0; k < aug_features_[i].size(); ++k) {
				if (k != j) {
					int map_idx = aug_features_[i][k] * feature_dim_ * num_;
					caffe_mul(feature_dim_ * num_, bottom_data_t_.cpu_data() + map_idx,
						tmp_diffs_.cpu_data(), tmp_diffs_.mutable_cpu_data());
				}
			}
			caffe_cpu_axpby(feature_dim_ * num_, Dtype(1.), tmp_diffs_.cpu_data(),
						Dtype(1.), bottom_diff_t_.mutable_cpu_data() + aug_features_[i][j] * feature_dim_ * num_);
		}
	}
	caffe_cpu_gemm(CblasNoTrans, CblasTrans, num_, feature_dim_ * channels_, num_,
			Dtype(1.), diag_mat_.cpu_data(), bottom_diff_t_.cpu_data(), Dtype(0.), bottom[0]->mutable_cpu_diff());*/
	/*for (int i = 0; i < top[0]->count(); ++i) {
    	std::cout << top[0]->cpu_diff()[i] << " ";
  	}*/
	for (int i = 0; i < bottom[0]->count(); ++i) {
    	std::cout << bottom[0]->cpu_diff()[i] << " ";
  	}
	std::cout << std::endl;
}

#ifdef CPU_ONLY
STUB_GPU(FeatureSelectLayer);
#endif

INSTANTIATE_CLASS(FeatureSelectLayer);
REGISTER_LAYER_CLASS(FeatureSelect);

}  // namespace caffe
