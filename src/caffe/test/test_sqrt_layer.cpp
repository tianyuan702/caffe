#include <iostream>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/sqrt_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

#ifndef CPU_ONLY
extern cudaDeviceProp CAFFE_TEST_CUDA_PROP;
#endif

template <typename TypeParam>
class SqrtLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;
 protected:
  SqrtLayerTest()
      : blob_bottom_(new Blob<Dtype>(2, 4, 1, 1)),
        blob_top_(new Blob<Dtype>()) {
    // fill the values
    blob_bottom_->mutable_cpu_data()[0] = 0;
    blob_bottom_->mutable_cpu_data()[1] = -4;
    blob_bottom_->mutable_cpu_data()[2] = 9;
    blob_bottom_->mutable_cpu_data()[3] = -16;
    blob_bottom_->mutable_cpu_data()[4] = -25;
    blob_bottom_->mutable_cpu_data()[5] = -36;
    blob_bottom_->mutable_cpu_data()[6] = 49;
    blob_bottom_->mutable_cpu_data()[7] = -64;
  }
  virtual ~SqrtLayerTest() {
    delete blob_bottom_;
    delete blob_top_;
  }
  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(SqrtLayerTest, TestDtypesAndDevices);

/*TYPED_TEST(SqrtLayerTest, Forward) {
  typedef typename TypeParam::Dtype Dtype;
  this->blob_bottom_vec_.push_back(this->blob_bottom_);
  this->blob_top_vec_.push_back(this->blob_top_);
  LayerParameter layer_param;
  shared_ptr<SqrtLayer<Dtype> > layer(
    new SqrtLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  this->blob_top_vec_[0]->mutable_cpu_diff()[0] = 1;
  this->blob_top_vec_[0]->mutable_cpu_diff()[1] = 1;
  this->blob_top_vec_[0]->mutable_cpu_diff()[2] = 1;
  this->blob_top_vec_[0]->mutable_cpu_diff()[3] = 1;
  this->blob_top_vec_[0]->mutable_cpu_diff()[4] = 1;
  this->blob_top_vec_[0]->mutable_cpu_diff()[5] = 1;
  this->blob_top_vec_[0]->mutable_cpu_diff()[6] = 1;
  this->blob_top_vec_[0]->mutable_cpu_diff()[7] = 1;
  std::vector<bool> propagation_down;
  layer->Backward(this->blob_top_vec_, propagation_down,
                  this->blob_bottom_vec_);
  for (int i = 0; i < 8; ++i) {
    std::cout << this->blob_bottom_vec_[0]->cpu_diff()[i] << std::endl;
  }
}*/

}
