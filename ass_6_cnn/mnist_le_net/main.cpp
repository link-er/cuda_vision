#include <cstring>
#include <cstdlib>
#include <vector>

#include <string>
#include <iostream>
#include <stdio.h>

#include "caffe/caffe.hpp"
#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "mnist.h"

using namespace caffe;
using namespace std;
typedef double Dtype;

int main(int argc, char** argv) {
  Caffe::set_mode(Caffe::GPU);

  // ==== 1 LAYER ====
  // data layer, read MNIST data from lmdb
  vector<Blob<Dtype>*> blob_bottom_data_vec_;
  vector<Blob<Dtype>*> blob_top_data_vec_;

  Blob<Dtype>* const blob_data = new Blob<Dtype>();
  Blob<Dtype>* const blob_label = new Blob<Dtype>();

  blob_top_data_vec_.push_back(blob_data);
  blob_top_data_vec_.push_back(blob_label);

  LayerParameter layer_data_param;
  DataParameter* data_param = layer_data_param.mutable_data_param();
  data_param->set_batch_size(64);
  data_param->set_source("/home/stud/adilova/caffe/caffe-rc2/examples/mnist/mnist_train_lmdb");
  data_param->set_backend(caffe::DataParameter_DB_LMDB);
  // transforming input data - normilizing it from 0~255 to 0~1
  TransformationParameter* transform_param = layer_data_param.mutable_transform_param();
  transform_param->set_scale(1./255.);
  DataLayer<Dtype> layer_data(layer_data_param);
  layer_data.SetUp(blob_bottom_data_vec_, blob_top_data_vec_);
  // ==== 1 LAYER ====

  // ==== 2-3 LAYER ====
  // first pair - convolution\pooling
  vector<Blob<Dtype>*> blob_top_conv1_vec_;
  vector<Blob<Dtype>*> blob_bottom_conv1_vec_;

  Blob<Dtype>* const blob_top_conv1_ = new Blob<Dtype>();

  blob_bottom_conv1_vec_.push_back(blob_data);
  blob_top_conv1_vec_.push_back(blob_top_conv1_);

  LayerParameter layer_conv1_param;
  ConvolutionParameter* conv1_param = layer_conv1_param.mutable_convolution_param();
  conv1_param->set_num_output(20);
  conv1_param->add_kernel_size(5);
  conv1_param->mutable_weight_filler()->set_type("xavier");
  ConvolutionLayer<Dtype> conv1_layer(layer_conv1_param);
  conv1_layer.SetUp(blob_bottom_conv1_vec_, blob_top_conv1_vec_)

  vector<Blob<Dtype>*> blob_top_pool1_vec_;
  vector<Blob<Dtype>*> blob_bottom_pool1_vec_;

  Blob<Dtype>* const blob_top_pool1_ = new Blob<Dtype>();

  blob_bottom_pool1_vec_.push_back(blob_top_conv1_);
  blob_top_pool1_vec_.push_back(blob_top_pool1_);

  LayerParameter layer_pool1_param;
  PoolingParameter* pool1_param = layer_pool1_param.mutable_pooling_param();
  pool1_param->set_pool(caffe::PoolingParameter::MAX);
  pool1_param->set_kernel_size(2);
  pool1_param->set_stride(2);
  PoolingLayer<Dtype> pool1_layer(layer_pool1_param);
  pool1_layer.SetUp(blob_bottom_pool1_vec_, blob_top_pool1_vec_)
  // ==== 2-3 LAYER ====

  // ==== 4-5 LAYER ====
  // second pair - convolution\pooling
  vector<Blob<Dtype>*> blob_top_conv2_vec_;
  vector<Blob<Dtype>*> blob_bottom_conv2_vec_;

  Blob<Dtype>* const blob_top_conv2_ = new Blob<Dtype>();

  blob_bottom_conv2_vec_.push_back(blob_top_pool1_);
  blob_top_conv2_vec_.push_back(blob_top_conv2_);

  LayerParameter layer_conv2_param;
  ConvolutionParameter* conv2_param = layer_conv2_param.mutable_convolution_param();
  conv2_param->set_num_output(50);
  conv2_param->add_kernel_size(5);
  conv2_param->mutable_weight_filler()->set_type("xavier");
  ConvolutionLayer<Dtype> conv2_layer(layer_conv2_param);
  conv2_layer.SetUp(blob_bottom_conv2_vec_, blob_top_conv2_vec_)

  vector<Blob<Dtype>*> blob_top_pool2_vec_;
  vector<Blob<Dtype>*> blob_bottom_pool2_vec_;

  Blob<Dtype>* const blob_top_pool2_ = new Blob<Dtype>();

  blob_bottom_pool2_vec_.push_back(blob_top_conv2_);
  blob_top_pool2_vec_.push_back(blob_top_pool2_);

  LayerParameter layer_pool2_param;
  PoolingParameter* pool2_param = layer_pool2_param.mutable_pooling_param();
  pool2_param->set_pool(caffe::PoolingParameter::MAX);
  pool2_param->set_kernel_size(2);
  pool2_param->set_stride(2);
  PoolingLayer<Dtype> pool2_layer(layer_pool2_param);
  pool2_layer.SetUp(blob_bottom_pool2_vec_, blob_top_pool2_vec_)
  // ==== 4-5 LAYER ====

  // ==== 6 LAYER ====
  // first inner product layer
  vector<Blob<Dtype>*> blob_top_ip1_vec_;
  vector<Blob<Dtype>*> blob_bottom_ip1_vec_;

  Blob<Dtype>* const blob_top_ip1_ = new Blob<Dtype>();

  blob_bottom_ip1_vec_.push_back(blob_top_pool2_);
  blob_top_ip1_vec_.push_back(blob_top_ip1_);

  LayerParameter layer_ip1_param;
  InnerProductParameter* ip1_param = layer_ip1_param.mutable_inner_product_param();
  ip1_param->set_num_output(500);
  ip1_param->mutable_weight_filler()->set_type("xavier");
  InnerProductLayer<Dtype> ip1_layer(layer_ip1_param);
  ip1_layer.SetUp(blob_bottom_ip1_vec_, blob_top_ip1_vec_);
  // ==== 6 LAYER ====

  // ==== 7 LAYER ====
  // ReLu layer
  vector<Blob<Dtype>*> blob_top_relu_vec_;
  vector<Blob<Dtype>*> blob_bottom_relu_vec_;

  blob_bottom_relu_vec_.push_back(blob_top_ip1_);
  blob_top_relu_vec_.push_back(blob_top_ip1_);

  ReLULayer<Dtype> relu_layer();
  relu_layer.SetUp(blob_bottom_relu_vec_, blob_top_relu_vec_);
  // ==== 7 LAYER ====

  // ==== 8 LAYER ====
  // second inner product layer
  vector<Blob<Dtype>*> blob_top_ip2_vec_;
  vector<Blob<Dtype>*> blob_bottom_ip2_vec_;

  Blob<Dtype>* const blob_top_ip2_ = new Blob<Dtype>();

  blob_bottom_ip1_vec_.push_back(blob_top_ip1_);
  blob_top_ip2_vec_.push_back(blob_top_ip2_);

  LayerParameter layer_ip2_param;
  InnerProductParameter* ip2_param = layer_ip2_param.mutable_inner_product_param();
  ip1_param->set_num_output(10);
  ip1_param->mutable_weight_filler()->set_type("xavier");
  InnerProductLayer<Dtype> ip2_layer(layer_ip2_param);
  ip2_layer.SetUp(blob_bottom_ip2_vec_, blob_top_ip2_vec_);
  // ==== 8 LAYER ====

  // ==== 9 LAYER ====
  // softmax loss layer
  vector<Blob<Dtype>*> blob_bottom_loss_vec_;
  vector<Blob<Dtype>*> blob_top_loss_vec_;

  Blob<Dtype>* const blob_top_loss_ = new Blob<Dtype>();

  blob_bottom_loss_vec_.push_back(blob_top_ip2_);
  blob_bottom_loss_vec_.push_back(blob_label);
  blob_top_loss_vec_.push_back(blob_top_loss_);

  LayerParameter layer_loss_param;
  SoftmaxWithLossLayer<Dtype> layer_loss(layer_loss_param);
  layer_loss.SetUp(blob_bottom_loss_vec_, blob_top_loss_vec_);
  // ==== 9 LAYER ====


  return 0;
}

