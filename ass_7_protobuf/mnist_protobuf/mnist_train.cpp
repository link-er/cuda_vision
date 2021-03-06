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
#include "caffe/sgd_solvers.hpp"
#include "google/protobuf/text_format.h"

using namespace caffe;
using namespace std;
typedef double Dtype;

int main(int argc, char** argv) {
    // set net
    string proto =
        "name: 'mnist_train' "
        "layer { "
        "  name: 'data' "
        "  type: 'Data' "
        "  top: 'data' "
        "  top: 'label' "
        "  data_param { "
        "    source: '/home/stud/adilova/caffe/caffe-rc3/examples/mnist/mnist_train_lmdb' "
        "    backend: LMDB "
        "    batch_size: 64 "
        "  } "
        "  transform_param { "
        "    scale: 0.00390625 "
        "  } "
        "  include { "
        "    phase: TRAIN "
        "  } "
        "} "
        "layer { "
        "  name: 'conv1' "
        "  type: 'Convolution' "
        "  bottom: 'data' "
        "  top: 'conv1' "
        "  param { "
        "    lr_mult: 1 "
        "  } "
        "  param { "
        "    lr_mult: 2 "
        "  } "
        "  convolution_param { "
        "    num_output: 20 "
        "    kernel_size: 5 "
        "    stride: 1 "
        "    weight_filler { "
        "      type: 'xavier' "
        "    } "
        "   bias_filler { "
        "      type: 'constant' "
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
        "  param { "
        "    lr_mult: 1 "
        "  } "
        "  param { "
        "    lr_mult: 2 "
        "  } "
        "  convolution_param { "
        "    num_output: 50 "
        "    kernel_size: 5 "
        "    stride: 1 "
        "    weight_filler { "
        "      type: 'xavier' "
        "    } "
        "   bias_filler { "
        "      type: 'constant' "
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
        "  param { "
        "    lr_mult: 1 "
        "  } "
        "  param { "
        "    lr_mult: 2 "
        "  } "
        "  inner_product_param { "
        "    num_output: 500 "
        "    weight_filler { "
        "      type: 'xavier' "
        "    } "
        "   bias_filler { "
        "      type: 'constant' "
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
        "  param { "
        "    lr_mult: 1 "
        "  } "
        "  param { "
        "    lr_mult: 2 "
        "  } "
        "  inner_product_param { "
        "    num_output: 10 "
        "    weight_filler { "
        "      type: 'xavier' "
        "    } "
        "   bias_filler { "
        "      type: 'constant' "
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

    NetParameter param_net;
    google::protobuf::TextFormat::ParseFromString(proto, &param_net);

    SolverParameter param_solver;
    param_solver.set_allocated_net_param(&param_net);
    param_solver.set_base_lr(0.01);
    param_solver.set_max_iter(1000);
    param_solver.set_lr_policy("inv");
    param_solver.set_momentum(0.9);
    param_solver.set_gamma(0.0001);
    param_solver.set_snapshot(1000);
    param_solver.set_snapshot_prefix("mnist_train_");
    param_solver.set_display(100);
    param_solver.set_solver_mode(SolverParameter_SolverMode_GPU);

    // training
    AdaDeltaSolver<Dtype> solver(param_solver);
    solver.Solve();
}
