#include <iostream>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/feature_select_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

#ifndef CPU_ONLY
extern cudaDeviceProp CAFFE_TEST_CUDA_PROP;
#endif

template <typename TypeParam>
class FeatureSelectLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;
 protected:
  FeatureSelectLayerTest()
      : blob_bottom_(new Blob<Dtype>(2, 2, 2, 2)),
        blob_top_(new Blob<Dtype>()) {
    // fill the values
    FillerParameter filler_param;
    UniformFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
  }
  virtual ~FeatureSelectLayerTest() {
    delete blob_bottom_;
    delete blob_top_;
  }
  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(FeatureSelectLayerTest, TestDtypesAndDevices);

/*TYPED_TEST(FeatureSelectLayerTest, TestForward) {
  typedef typename TypeParam::Dtype Dtype;
  this->blob_bottom_vec_.push_back(this->blob_bottom_);
  this->blob_top_vec_.push_back(this->blob_top_);
  LayerParameter layer_param;
  shared_ptr<FeatureSelectLayer<Dtype> > layer(
    new FeatureSelectLayer<Dtype>(layer_param));
  for (int i = 0; i < this->blob_bottom_vec_[0]->channels(); ++i) {
    for (int j = i + 1; j < this->blob_bottom_vec_[0]->channels(); ++j) {
      std::vector<int> p;
      p.push_back(i);
      p.push_back(j);
      layer->aug_features_.push_back(p);
    }
    std::vector<int> p;
    p.push_back(i);
    p.push_back(i);
    layer->aug_features_.push_back(p);
  }
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  std::vector<bool> propagate_down;
  for (int i = 0; i < this->blob_top_vec_[0]->count(); ++i) {
    this->blob_top_vec_[0]->mutable_cpu_diff()[i] = Dtype(1.);
  }
  layer->Backward(this->blob_top_vec_, propagate_down, this->blob_bottom_vec_);
}*/

TYPED_TEST(FeatureSelectLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  this->blob_bottom_vec_.push_back(this->blob_bottom_);
  this->blob_top_vec_.push_back(this->blob_top_);
  LayerParameter layer_param;
  FeatureSelectLayer<Dtype> layer(layer_param);
  for (int i = 0; i < this->blob_bottom_vec_[0]->channels(); ++i) {
    for (int j = i + 1; j < this->blob_bottom_vec_[0]->channels(); ++j) {
      std::vector<int> p;
      p.push_back(i);
      p.push_back(j);
      layer.aug_features_.push_back(p);
    }
    std::vector<int> p;
    p.push_back(i);
    p.push_back(i);
    p.push_back(i);
    layer.aug_features_.push_back(p);
  }
  GradientChecker<Dtype> checker(1e-2, 1e-3);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
                                  this->blob_top_vec_);
}

}
