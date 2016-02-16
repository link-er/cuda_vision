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
#include "caffe/net.hpp"
#include "google/protobuf/text_format.h"

using namespace caffe;
using namespace std;
typedef double Dtype;

int main(int argc, char** argv) {

    string proto_test =
        "name: 'mnist_test' "
        "layer { "
        "  name: 'data' "
        "  type: 'Data' "
        "  top: 'data' "
        "  top: 'label' "
        "  data_param { "
        "    source: '/home/stud/adilova/caffe/caffe-rc2/examples/mnist/mnist_test_lmdb' "
        "    backend: LMDB "
        "    batch_size: 10000 "
        "  } "
        "} "
        "layer { "
        "  name: 'conv1' "
        "  type: 'Convolution' "
        "  bottom: 'data' "
        "  top: 'conv1' "
        "  convolution_param { "
        "    num_output: 20 "
        "    kernel_size: 5 "
        "    weight_filler { "
        "      type: xavier "
        "    } "
        "  } "
        "} "
        "layer { "
        "  name: 'pool1' "
        "  type: 'Pooling' "
        "  bottom: 'conv1' "
        "  top: 'pool1' "
        "  pooling_param { "
        "    pool: MAX "
        "    kernel_size: 2 "
        "    stride: 2 "
        "  } "
        "} "
        "layer { "
        "  name: 'conv2' "
        "  type: 'Convolution' "
        "  bottom: 'pool1' "
        "  top: 'conv2' "
        "  convolution_param { "
        "    num_output: 50 "
        "    kernel_size: 5 "
        "    weight_filler { "
        "      type: xavier "
        "    } "
        "  } "
        "} "
        "layer { "
        "  name: 'pool2' "
        "  type: 'Pooling' "
        "  bottom: 'conv2' "
        "  top: 'pool2' "
        "  pooling_param { "
        "    pool: MAX "
        "    kernel_size: 2 "
        "    stride: 2 "
        "  } "
        "} "
        "layer { "
        "  name: 'ip1' "
        "  type: 'InnerProduct' "
        "  bottom: 'pool2' "
        "  top: 'ip1' "
        "  inner_product_param { "
        "    num_output: 500 "
        "    weight_filler { "
        "      type: xavier "
        "    } "
        "  } "
        "} "
        "layer { "
        "  name: 'relu' "
        "  type: 'ReLU' "
        "  bottom: 'ip1' "
        "  top: 'ip1' "
        "} "
        "layer { "
        "  name: 'ip2' "
        "  type: 'InnerProduct' "
        "  bottom: 'ip1' "
        "  top: 'ip2' "
        "  inner_product_param { "
        "    num_output: 10 "
        "    weight_filler { "
        "      type: xavier "
        "    } "
        "  } "
        "} "
        "layer { "
        "  name: 'loss' "
        "  type: 'SoftmaxWithLoss' "
        "  bottom: 'ip2' "
        "  bottom: 'label' "
        "  top: 'loss' "
        "} ";
    NetParameter param_test;
    google::protobuf::TextFormat::ParseFromString(proto_test, &param_test);

    Net<Dtype> net_test(param_test);
    net_test.CopyTrainedLayersFrom(argv[1]);

    double loss = 0;
    const vector<Blob<Dtype>*>& result = net_test.ForwardPrefilled();
    loss = result[0]->cpu_data()[0];
    LOG(ERROR) << "Loss: " << loss;

    // blobs from the forwarded net
    shared_ptr<Blob<Dtype> > blob_label = net_test.blob_by_name("label");
    shared_ptr<Blob<Dtype> > blob_ip = net_test.blob_by_name("ip2");

    // evaluation
    int score = 0, label, max_label;
    Dtype max_score;
    Dtype* scores = new Dtype[10];;
    for (int n=0;n<10000;n++){
        label = blob_label->mutable_cpu_data()[blob_label->offset(n,0,0,0)];
        // argmax evaluate
        max_score = 0;
        max_label = 0;
        for(int i=0; i<10; i++) {
            scores[i] = blob_ip->mutable_cpu_data()[blob_ip->offset(n,i,0,0)];
            if(scores[i] > max_score) {
                max_score = scores[i];
                max_label = i;
            }
        }

        if(max_label == label)
            score++;
    }
    cout<<"Test score: "<<score<<" out of "<<10000<<endl;

}
