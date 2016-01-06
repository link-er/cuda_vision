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

using namespace caffe;
using namespace std;
typedef double Dtype;

int nIter = 1000;
int nTest = 10000;

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
  conv1_param->set_kernel_size(5);
  conv1_param->mutable_weight_filler()->set_type("xavier");
  ConvolutionLayer<Dtype> conv1_layer(layer_conv1_param);
  conv1_layer.SetUp(blob_bottom_conv1_vec_, blob_top_conv1_vec_);

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
  pool1_layer.SetUp(blob_bottom_pool1_vec_, blob_top_pool1_vec_);
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
  conv2_param->set_kernel_size(5);
  conv2_param->mutable_weight_filler()->set_type("xavier");
  ConvolutionLayer<Dtype> conv2_layer(layer_conv2_param);
  conv2_layer.SetUp(blob_bottom_conv2_vec_, blob_top_conv2_vec_);

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
  pool2_layer.SetUp(blob_bottom_pool2_vec_, blob_top_pool2_vec_);
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

  LayerParameter layer_relu_param;
  ReLULayer<Dtype> relu_layer(layer_relu_param);
  relu_layer.SetUp(blob_bottom_relu_vec_, blob_top_relu_vec_);
  // ==== 7 LAYER ====

  // ==== 8 LAYER ====
  // second inner product layer
  /*vector<Blob<Dtype>*> blob_top_ip2_vec_;
  vector<Blob<Dtype>*> blob_bottom_ip2_vec_;

  Blob<Dtype>* const blob_top_ip2_ = new Blob<Dtype>();

  blob_bottom_ip2_vec_.push_back(blob_top_ip1_);
  blob_top_ip2_vec_.push_back(blob_top_ip2_);

  LayerParameter layer_ip2_param;
  InnerProductParameter* ip2_param = layer_ip2_param.mutable_inner_product_param();
  ip1_param->set_num_output(10);
  ip1_param->mutable_weight_filler()->set_type("xavier");
  InnerProductLayer<Dtype> ip2_layer(layer_ip2_param);
  ip2_layer.SetUp(blob_bottom_ip2_vec_, blob_top_ip2_vec_);
  */// ==== 8 LAYER ====

  // ==== 9 LAYER ====
  // softmax loss layer
  vector<Blob<Dtype>*> blob_bottom_loss_vec_;
  vector<Blob<Dtype>*> blob_top_loss_vec_;

  Blob<Dtype>* const blob_top_loss_ = new Blob<Dtype>();

  blob_bottom_loss_vec_.push_back(blob_top_ip1_);
  blob_bottom_loss_vec_.push_back(blob_label);
  blob_top_loss_vec_.push_back(blob_top_loss_);

  LayerParameter layer_loss_param;
  SoftmaxWithLossLayer<Dtype> layer_loss(layer_loss_param);
  layer_loss.SetUp(blob_bottom_loss_vec_, blob_top_loss_vec_);
  // ==== 9 LAYER ====

  // ==== TRAINING ====
  ofstream learning_curve;
  learning_curve.open("/home/stud/adilova/cuda_vision/ass_6_cnn/mnist_le_net/errors.txt");
  // forward and backward iteration
  int n = 0;
  Dtype loss = 1000;
  while(n < nIter && loss > 0.1){
    // forward
    layer_data.Forward(blob_bottom_data_vec_, blob_top_data_vec_);
    conv1_layer.Forward(blob_bottom_conv1_vec_, blob_top_conv1_vec_);
    pool1_layer.Forward(blob_bottom_pool1_vec_, blob_top_pool1_vec_);
    conv2_layer.Forward(blob_bottom_conv2_vec_, blob_top_conv2_vec_);
    pool2_layer.Forward(blob_bottom_pool2_vec_, blob_top_pool2_vec_);
    ip1_layer.Forward(blob_bottom_ip1_vec_, blob_top_ip1_vec_);
    relu_layer.Forward(blob_bottom_relu_vec_, blob_top_relu_vec_);
    // ip2_layer.Forward(blob_bottom_ip2_vec_, blob_top_ip2_vec_);
    loss = layer_loss.Forward(blob_bottom_loss_vec_, blob_top_loss_vec_);

    cout<<"Iter "<<n<<" loss "<<loss<<endl;
    learning_curve << n << " " << loss << endl;

    // backward
    vector<bool> backpro_vec;
    backpro_vec.push_back(1);
    backpro_vec.push_back(0);
    layer_loss.Backward(blob_top_loss_vec_, backpro_vec, blob_bottom_loss_vec_);
    // ip2_layer.Backward(blob_top_ip2_vec_, backpro_vec, blob_bottom_ip2_vec_);
    relu_layer.Backward(blob_top_relu_vec_, backpro_vec, blob_bottom_relu_vec_);
    ip1_layer.Backward(blob_top_ip1_vec_, backpro_vec, blob_bottom_ip1_vec_);
    pool2_layer.Backward(blob_top_pool2_vec_, backpro_vec, blob_bottom_pool2_vec_);
    conv2_layer.Backward(blob_top_conv2_vec_, backpro_vec, blob_bottom_conv2_vec_);
    pool1_layer.Backward(blob_top_pool1_vec_, backpro_vec, blob_bottom_pool1_vec_);
    conv1_layer.Backward(blob_top_conv1_vec_, backpro_vec, blob_bottom_conv1_vec_);
    layer_data.Backward(blob_top_data_vec_, backpro_vec, blob_bottom_data_vec_);

    // update weights of layer_ip
    Dtype rate = 0.1;
    // vector<shared_ptr<Blob<Dtype> > > param2 = ip2_layer.blobs();
    vector<shared_ptr<Blob<Dtype> > > param1 = ip1_layer.blobs();
    // caffe_scal(param2[0]->count(), rate, param2[0]->mutable_cpu_diff());
    // param2[0]->Update();
    caffe_scal(param1[0]->count(), rate, param1[0]->mutable_cpu_diff());
    param1[0]->Update();

    n++;
  }
  // ==== TRAINING ====

  // ==== PREDICTING ====
  vector<Blob<Dtype>*> blob_bottom_test_data_vec_;
  vector<Blob<Dtype>*> blob_top_test_data_vec_;

  Blob<Dtype>* const blob_test_data = new Blob<Dtype>();
  Blob<Dtype>* const blob_test_label = new Blob<Dtype>();

  blob_top_test_data_vec_.push_back(blob_test_data);
  blob_top_test_data_vec_.push_back(blob_test_label);

  LayerParameter layer_test_data_param;
  DataParameter* test_data_param = layer_test_data_param.mutable_data_param();
  data_param->set_batch_size(nTest);
  data_param->set_source("/home/stud/adilova/caffe/caffe-rc2/examples/mnist/mnist_test_lmdb");
  data_param->set_backend(caffe::DataParameter_DB_LMDB);
  // transforming input data - normilizing it from 0~255 to 0~1
  TransformationParameter* transform_test_param = layer_test_data_param.mutable_transform_param();
  transform_test_param->set_scale(1./255.);
  DataLayer<Dtype> layer_test_data(layer_test_data_param);
  layer_test_data.SetUp(blob_bottom_test_data_vec_, blob_top_test_data_vec_);

  // conv1
  vector<Blob<Dtype>*> blob_bottom_conv1_test_vec_;
  vector<Blob<Dtype>*> blob_top_conv1_test_vec_;
  Blob<Dtype>* const blob_top_conv1_test_ = new Blob<Dtype>();

  blob_bottom_conv1_test_vec_.push_back(blob_test_data);
  blob_top_conv1_vec_.push_back(blob_top_conv1_test_);

  conv1_layer.Reshape(blob_bottom_conv1_test_vec_, blob_top_conv1_test_vec_);

  // pool1
  vector<Blob<Dtype>*> blob_bottom_pool1_test_vec_;
  vector<Blob<Dtype>*> blob_top_pool1_test_vec_;
  Blob<Dtype>* const blob_top_pool1_test_ = new Blob<Dtype>();

  blob_bottom_pool1_test_vec_.push_back(blob_top_conv1_test_);
  blob_top_pool1_test_vec_.push_back(blob_top_pool1_test_);
  pool1_layer.Reshape(blob_bottom_pool1_test_vec_, blob_top_pool1_test_vec_);

  // conv2
  vector<Blob<Dtype>*> blob_bottom_conv2_test_vec_;
  vector<Blob<Dtype>*> blob_top_conv2_test_vec_;
  Blob<Dtype>* const blob_top_conv2_test_ = new Blob<Dtype>();

  blob_bottom_conv2_test_vec_.push_back(blob_top_pool1_test_);
  blob_top_conv2_test_vec_.push_back(blob_top_conv2_test_);
  conv2_layer.Reshape(blob_bottom_conv2_test_vec_, blob_top_conv2_test_vec_);

  // pool2
  vector<Blob<Dtype>*> blob_bottom_pool2_test_vec_;
  vector<Blob<Dtype>*> blob_top_pool2_test_vec_;
  Blob<Dtype>* const blob_top_pool2_test_ = new Blob<Dtype>();

  blob_bottom_pool2_test_vec_.push_back(blob_top_conv2_test_);
  blob_top_pool2_test_vec_.push_back(blob_top_pool2_test_);
  pool2_layer.Reshape(blob_bottom_pool2_test_vec_, blob_top_pool2_test_vec_);

  // ip1
  vector<Blob<Dtype>*> blob_bottom_ip1_test_vec_;
  vector<Blob<Dtype>*> blob_top_ip1_test_vec_;
  Blob<Dtype>* const blob_top_ip1_test_ = new Blob<Dtype>();

  blob_bottom_ip1_test_vec_.push_back(blob_top_pool2_test_);
  blob_top_ip1_test_vec_.push_back(blob_top_ip1_test_);
  ip1_layer.Reshape(blob_bottom_ip1_test_vec_, blob_top_ip1_test_vec_);

  // relu
  vector<Blob<Dtype>*> blob_bottom_relu_test_vec_;
  vector<Blob<Dtype>*> blob_top_relu_test_vec_;

  blob_bottom_relu_test_vec_.push_back(blob_top_ip1_test_);
  blob_top_relu_test_vec_.push_back(blob_top_ip1_test_);
  relu_layer.Reshape(blob_bottom_relu_test_vec_, blob_top_relu_test_vec_);

  // ip2
  /*vector<Blob<Dtype>*> blob_bottom_ip2_test_vec_;
  vector<Blob<Dtype>*> blob_top_ip2_test_vec_;
  Blob<Dtype>* const blob_top_ip2_test_ = new Blob<Dtype>();

  blob_bottom_ip2_test_vec_.push_back(blob_top_ip1_test_);
  blob_top_ip2_test_vec_.push_back(blob_top_ip2_test_);
  ip2_layer.Reshape(blob_bottom_ip2_test_vec_, blob_top_ip2_test_vec_);
  */

  // loss
  vector<Blob<Dtype>*> blob_bottom_loss_test_vec_;
  vector<Blob<Dtype>*> blob_top_loss_test_vec_;
  Blob<Dtype>* const blob_top_loss_test_ = new Blob<Dtype>();

  blob_bottom_loss_test_vec_.push_back(blob_top_ip1_test_);
  blob_top_loss_test_vec_.push_back(blob_top_loss_test_);
  layer_loss.Reshape(blob_bottom_loss_test_vec_, blob_top_loss_test_vec_);

  // argmax
  vector<Blob<Dtype>*> blob_bottom_argmax_vec_;
  vector<Blob<Dtype>*> blob_top_argmax_vec_;

  Blob<Dtype>* blob_top_argmax_ = new Blob<Dtype>();

  blob_bottom_argmax_vec_.push_back(blob_top_loss_test_);
  blob_top_argmax_vec_.push_back(blob_top_argmax_);

  LayerParameter layer_argmax_param;
  ArgMaxParameter* argmax_param = layer_argmax_param.mutable_argmax_param();
  argmax_param->set_out_max_val(false);
  ArgMaxLayer<Dtype> argmax_layer(layer_argmax_param);
  argmax_layer.SetUp(blob_bottom_argmax_vec_, blob_top_argmax_vec_);

  // predict
  layer_data.Forward(blob_bottom_test_data_vec_, blob_top_test_data_vec_);
  conv1_layer.Forward(blob_bottom_conv1_test_vec_, blob_top_conv1_test_vec_);
  pool1_layer.Forward(blob_bottom_pool1_test_vec_, blob_top_pool1_test_vec_);
  conv2_layer.Forward(blob_bottom_conv2_test_vec_, blob_top_conv2_test_vec_);
  pool2_layer.Forward(blob_bottom_pool2_test_vec_, blob_top_pool2_test_vec_);
  ip1_layer.Forward(blob_bottom_ip1_test_vec_, blob_top_ip1_test_vec_);
  relu_layer.Forward(blob_bottom_relu_test_vec_, blob_top_relu_test_vec_);
  // ip2_layer.Forward(blob_bottom_ip2_test_vec_, blob_top_ip2_test_vec_);
  layer_loss.Forward(blob_bottom_loss_test_vec_, blob_top_loss_test_vec_);
  argmax_layer.Forward(blob_bottom_argmax_vec_, blob_top_argmax_vec_);

  int label, max_index, score = 0;
  const Dtype* predictions = blob_top_argmax_->cpu_data();
  for(int c = 0; c < nTest ; c++){
    max_index = predictions[blob_top_argmax_->offset(c,0,0,0)];
    // label = data.blob_train_labels->cpu_data()[data.blob_train_labels->offset(max_index,0,0,0)];
    // if(label == data.blob_test_labels->cpu_data()[data.blob_test_labels->offset(c,0,0,0)])
    //   score++;
  }
  cout<<"Test score: "<<score<<" out of "<<nTest<<endl;
  // ==== PREDICTING ====

  return 0;
}

