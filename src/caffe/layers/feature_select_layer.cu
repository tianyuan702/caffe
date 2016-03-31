#include "caffe/layers/feature_select_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void FeatureSelectLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
    if (aug_features_.size() > 0) {
        caffe_gpu_gemm(CblasTrans, CblasNoTrans, feature_dim_ * bottom_channels_,
                num_, num_, Dtype(1.), bottom[0]->gpu_data(),
                diag_mat_.gpu_data(), Dtype(0.),
                bottom_data_t_.mutable_gpu_data());

        caffe_gpu_set(top_data_t_.count(), Dtype(1.),
                    top_data_t_.mutable_gpu_data());
        for (int i = 0; i < top_channels_; ++i) {
            int top_t_idx = i * feature_dim_ * num_;
            for (int j = 0; j < aug_features_[i].size(); ++j) {
                int bottom_idx = aug_features_[i][j] * feature_dim_ * num_;
                caffe_gpu_mul(feature_dim_ * num_,
                    top_data_t_.gpu_data() + top_t_idx,
                    bottom_data_t_.gpu_data() + bottom_idx,
                    top_data_t_.mutable_gpu_data() + top_t_idx);
            }
        }

        caffe_gpu_gemm(CblasNoTrans, CblasTrans, num_,
                    feature_dim_ * top_channels_, num_, Dtype(1.),
                    diag_mat_.gpu_data(), top_data_t_.gpu_data(),
                    Dtype(0.), top[0]->mutable_gpu_data());
    }
}

template <typename Dtype>
void FeatureSelectLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    if (aug_features_.size() > 0) {
        caffe_gpu_set(bottom_diff_t_.count(), Dtype(0.),
                bottom_diff_t_.mutable_cpu_diff());
    
        caffe_gpu_gemm(CblasTrans, CblasNoTrans, feature_dim_ * top_channels_,
                    num_, num_, Dtype(1.), top[0]->gpu_diff(),
                    diag_mat_.gpu_data(), Dtype(0.),
                    top_diff_t_.mutable_gpu_diff());

        for (int i = 0; i < top_channels_; ++i) {
            for (int j = 0; j < aug_features_[i].size(); ++j) {
                caffe_gpu_axpby(tmp_diffs_.count(), Dtype(1.),
                            top_diff_t_.gpu_diff() + i * feature_dim_ * num_,
                            Dtype(0.), tmp_diffs_.mutable_gpu_diff());
                for (int k = 0; k < aug_features_[i].size(); ++k) {
                    if (k != j) {
                        int map_idx = aug_features_[i][k] * feature_dim_ * num_;
                        caffe_gpu_mul(feature_dim_ * num_,
                                bottom_data_t_.gpu_data() + map_idx,
                                tmp_diffs_.gpu_diff(),
                                tmp_diffs_.mutable_gpu_diff());
                    }
                }
                caffe_gpu_axpby(feature_dim_ * num_, Dtype(1.),
                            tmp_diffs_.gpu_diff(), Dtype(1.),
                            bottom_diff_t_.mutable_gpu_diff() +
                            aug_features_[i][j] * feature_dim_ * num_);
            }
        }
        
        caffe_gpu_gemm(CblasNoTrans, CblasTrans, num_,
                    feature_dim_ * bottom_channels_, num_, Dtype(1.),
                    diag_mat_.gpu_data(), bottom_diff_t_.gpu_diff(),
                    Dtype(0.), bottom[0]->mutable_gpu_diff());
    }
}

INSTANTIATE_LAYER_GPU_FUNCS(FeatureSelectLayer);

}  // namespace caffe