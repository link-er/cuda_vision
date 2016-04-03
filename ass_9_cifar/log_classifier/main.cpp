#include <cstring>
#include <cstdlib>
#include <vector>

#include <string>
#include <iostream>
#include <stdio.h>

#include <time.h>

#include "caffe/caffe.hpp"
#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/inner_product_layer.hpp"
#include "caffe/layers/softmax_loss_layer.hpp"
#include "mnist.h"

using namespace caffe;
using namespace std;
typedef double Dtype;

// parameters
int nTrainData = 50000;
int nTestData = 10000;
int dim = 32*32*3;
int clas = 10;
int nIter = 1000;

int main(int argc, char** argv) {
    Caffe::set_mode(Caffe::GPU);

    vector<Blob<Dtype>*> blob_bottom_data_vec;
    vector<Blob<Dtype>*> blob_top_data_vec;
    Blob<Dtype>* const blob_data = new Blob<Dtype>();
    Blob<Dtype>* const blob_label = new Blob<Dtype>();

    blob_top_data_vec.push_back(blob_data);
    blob_top_data_vec.push_back(blob_label);

    LayerParameter layer_data_param;
    DataParameter* data_param = layer_data_param.mutable_data_param();
    // data_param->set_batch_size(64);
    data_param->set_source("/home/VI/stud/adilova/caffe-master/examples/cifar10/cifar10_train_lmdb");
    data_param->set_backend(caffe::DataParameter_DB_LMDB);

    // TransformationParameter* transform_param = layer_data_param.mutable_transform_param();
    // transform_param->set_scale(1./255.);

    DataLayer<Dtype> layer_data(layer_data_param);
    layer_data.SetUp(blob_bottom_data_vec, blob_top_data_vec);


    // set inner product layer
    vector<Blob<Dtype>*> blob_bottom_ip_vec_;
    vector<Blob<Dtype>*> blob_top_ip_vec_;
    Blob<Dtype>* const blob_top_ip_ = new Blob<Dtype>();

    blob_bottom_ip_vec_.push_back(data.blob_train_images);
    blob_top_ip_vec_.push_back(blob_top_ip_);

    LayerParameter layer_ip_param;
    layer_ip_param.mutable_inner_product_param()->set_num_output(clas);
    layer_ip_param.mutable_inner_product_param()->mutable_weight_filler()->set_type("xavier");
    layer_ip_param.mutable_inner_product_param()->mutable_bias_filler()->set_type("constant");

    InnerProductLayer<Dtype> layer_ip(layer_ip_param);
    layer_ip.SetUp(blob_bottom_ip_vec_, blob_top_ip_vec_);

    // set softmax loss layer
    vector<Blob<Dtype>*> blob_bottom_loss_vec_;
    vector<Blob<Dtype>*> blob_top_loss_vec_;
    Blob<Dtype>* const blob_top_loss_ = new Blob<Dtype>();

    blob_bottom_loss_vec_.push_back(blob_top_ip_);
    blob_bottom_loss_vec_.push_back(data.blob_train_labels);
    blob_top_loss_vec_.push_back(blob_top_loss_);

    LayerParameter layer_loss_param;
    SoftmaxWithLossLayer<Dtype> layer_loss(layer_loss_param);
    layer_loss.SetUp(blob_bottom_loss_vec_, blob_top_loss_vec_);

    clock_t tStart = clock();
    ofstream learning_curve;
    learning_curve.open("/home/stud/adilova/cuda_vision/ass_5_layers/mnist_layers/errors.txt");
    // forward and backward iteration
    for(int n=0; n<nIter; n++){
        // forward
        layer_ip.Forward(blob_bottom_ip_vec_, blob_top_ip_vec_);
        Dtype loss = layer_loss.Forward(blob_bottom_loss_vec_, blob_top_loss_vec_);
        cout<<"Iter "<<n<<" loss "<<loss<<endl;
        learning_curve << n << " " << loss << endl;

        // backward
        vector<bool> backpro_vec;
        backpro_vec.push_back(1);
        backpro_vec.push_back(0);
        layer_loss.Backward(blob_top_loss_vec_, backpro_vec, blob_bottom_loss_vec_);
        layer_ip.Backward(blob_top_ip_vec_, backpro_vec, blob_bottom_ip_vec_);

        // update weights of layer_ip
        Dtype rate = 0.1;
        vector<shared_ptr<Blob<Dtype> > > param = layer_ip.blobs();
        caffe_scal(param[0]->count(), rate, param[0]->mutable_cpu_diff());
        param[0]->Update();
    }
    learning_curve.close();
    double time_taken = (double)(clock() - tStart)/CLOCKS_PER_SEC;
    cout<<"Learning time: "<<time_taken<<endl;

    // prediction
    vector<Blob<Dtype>*> blob_bottom_ip_test_vec_;
    vector<Blob<Dtype>*> blob_top_ip_test_vec_;
    Blob<Dtype>* const blob_top_ip_test_ = new Blob<Dtype>();

    blob_bottom_ip_test_vec_.push_back(data.blob_test_images);
    blob_top_ip_test_vec_.push_back(blob_top_ip_test_);

    layer_ip.Reshape(blob_bottom_ip_test_vec_, blob_top_ip_test_vec_);
    layer_ip.Forward(blob_bottom_ip_test_vec_, blob_top_ip_test_vec_);

    // evaluation
    int score = 0, label, max_label;
    Dtype max_score;
    Dtype* scores = new Dtype[10];
    for (int n=0; n<nTestData; n++){
        label = data.blob_test_labels->mutable_cpu_data()[data.blob_test_labels->offset(n,0,0,0)];
        // argmax evaluate
        max_score = 0;
        max_label = 0;
        for(int i=0; i<clas; i++) {
            scores[i] = blob_top_ip_test_->mutable_cpu_data()[blob_top_ip_test_->offset(n,i,0,0)];
            if(scores[i] > max_score) {
                max_score = scores[i];
                max_label = i;
            }
        }

        if(max_label == label)
            score++;
    }
    cout<<"Test score: "<<score<<" out of "<<nTestData<<endl;

    return 0;
}


