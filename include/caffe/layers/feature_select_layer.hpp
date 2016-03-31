#ifndef CAFFE_FEATURE_SELECT_LAYER_HPP_
#define CAFFE_FEATURE_SELECT_LAYER_HPP_

#include <vector>
#include <bitset>
#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

template <typename Dtype>
class FeatureSelectLayer : public Layer<Dtype> {
 public:
  explicit FeatureSelectLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual inline const char* type() const { return "FeatureSelectLayer"; }
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
 
 private:
  int num_;
  int top_channels_;
  int bottom_channels_;
  int feature_dim_;
  Blob<Dtype> top_data_t_;
  Blob<Dtype> top_diff_t_;
  Blob<Dtype> bottom_data_t_;
  Blob<Dtype> bottom_diff_t_;
  Blob<Dtype> diag_mat_;
  Blob<Dtype> tmp_diffs_;
 public:
  std::vector<std::vector<int> > aug_features_;
};

}

#endif  // CAFFE_FEATURE_SELECT_LAYER_HPP_
