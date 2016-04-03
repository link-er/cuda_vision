#include <cstring>
#include <cstdlib>
#include <vector>
#include <iomanip>

#include "caffe/caffe.hpp"
#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"

typedef double Dtype;
using namespace caffe;
using namespace std;

int main(int argc, char** argv) {
    Caffe::set_mode(Caffe::GPU);

    // reading train CIFAR data into blobs
    vector<Blob<Dtype>*> blob_bottom_data_vec;
    vector<Blob<Dtype>*> blob_top_data_vec;
    Blob<Dtype>* const blob_data = new Blob<Dtype>();
    Blob<Dtype>* const blob_label = new Blob<Dtype>();

    blob_top_data_vec.push_back(blob_data);
    blob_top_data_vec.push_back(blob_label);

    LayerParameter layer_data_param;
    DataParameter* data_param = layer_data_param.mutable_data_param();
    data_param->set_batch_size(50000);
    data_param->set_source("/home/VI/stud/adilova/caffe-rc2/examples/cifar10/cifar10_train_lmdb");
    data_param->set_backend(caffe::DataParameter_DB_LMDB);

    TransformationParameter* transform_param = layer_data_param.mutable_transform_param();
    transform_param->set_mean_file("/home/VI/stud/adilova/caffe-rc2/examples/cifar10/mean.binaryproto");

    DataLayer<Dtype> layer_data(layer_data_param);
    layer_data.SetUp(blob_bottom_data_vec, blob_top_data_vec);

    layer_data.Forward(blob_bottom_data_vec, blob_top_data_vec);

    /*int n = 0;
    for(int i=0;i<3;i++){
        for(int j=0;j<32;j++){
            for(int k=0;k<32;k++){
            	int px = blob_data->cpu_data()[blob_data->offset(n,i,j,k)];
            	cout << setfill(' ') << setw(4) << px;
	    }
	    cout<<endl;
        }
        cout<<endl;
    }
    int label = blob_label->cpu_data()[blob_label->offset(n,0,0,0)];
    cout<<"label: "<<label<<" "<<endl;*/

    // reading test CIFAR data into blobs
    vector<Blob<Dtype>*> blob_test_bottom_data_vec;
    vector<Blob<Dtype>*> blob_test_top_data_vec;
    Blob<Dtype>* const blob_test_data = new Blob<Dtype>();
    Blob<Dtype>* const blob_test_label = new Blob<Dtype>();

    blob_test_bottom_data_vec.push_back(blob_test_data);
    blob_test_top_data_vec.push_back(blob_test_label);

    data_param->set_batch_size(10000);
    data_param->set_source("/home/VI/stud/adilova/caffe-rc2/examples/cifar10/cifar10_test_lmdb");
    data_param->set_backend(caffe::DataParameter_DB_LMDB);

    layer_data.SetUp(blob_test_bottom_data_vec, blob_test_top_data_vec);

    layer_test_data.Forward(blob_test_bottom_data_vec, blob_test_top_data_vec);

    // ===================
    cout<<"Data read succefully."<<endl;

    int n = 50000;
    int m = 10000;
    int d = 32*32*3;

    blob_data->Reshape(1, 1, n, d);
    blob_test_data->Reshape(1, 1, m, d);

    cout << "Reshaped\n";

    Blob<Dtype>* blob_train_out = new Blob<Dtype>(1, 1, n, d);
    caffe_gpu_powx<Dtype>(blob_data->count(), blob_data->gpu_data(), 2.0, blob_train_out->mutable_gpu_data());

    Blob<Dtype>* blob_test_out = new Blob<Dtype>(1, 1, m, d);
    caffe_gpu_powx<Dtype>(blob_test_data->count(), blob_test_data->gpu_data(), 2.0, blob_test_out->mutable_gpu_data());

    cout << "Squared\n";

    Blob<Dtype>* unary_blob1 = new Blob<Dtype>(1, 1, d, m);
    FillerParameter filler_param;
    filler_param.set_min(1);
    filler_param.set_max(1);
    UniformFiller<Dtype> filler(filler_param);
    filler.Fill(unary_blob1);

    Blob<Dtype>* result_blob = new Blob<Dtype>(1, 1, m, n);
    caffe_gpu_gemm<Dtype>(CblasTrans, CblasTrans, m, n, d, 1., unary_blob1->gpu_data(), blob_train_out->gpu_data(),
                          0., result_blob->mutable_gpu_data());
    delete unary_blob1;
    delete blob_train_out;

    Blob<Dtype>* unary_blob2 = new Blob<Dtype>(1, 1, n, d);
    filler.Fill(unary_blob2);

    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, m, n, d, 1, blob_test_out->gpu_data(), unary_blob2->gpu_data(),
                          1, result_blob->mutable_gpu_data());
    delete unary_blob2;
    delete blob_test_out;
    cout << "Multiplied by unary and added the first one multiplied\n";

    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, m, n, d, -2., blob_test_data->gpu_data(),
                          blob_data->gpu_data(), 1., result_blob->mutable_gpu_data());
    cout << "Got final distances\n";
    delete blob_data;
    delete blob_test_data;

    // // ============= ARGMAX LAYER ================
    // cout << "Preparing layer\n";
    // Blob<Dtype>* blob_top = new Blob<Dtype>();
    // vector<Blob<Dtype>*> blob_bottom_vec;
    // vector<Blob<Dtype>*> blob_top_vec;

    // caffe_gpu_scale<Dtype>(m*n, -1, result_blob->gpu_data(), result_blob->mutable_gpu_data());
    // result_blob->Reshape(m, n, 1, 1);
    // blob_bottom_vec.push_back(result_blob);
    // blob_top_vec.push_back(blob_top);

    // LayerParameter layer_param;
    // ArgMaxParameter* argmax_param = layer_param.mutable_argmax_param();
    // argmax_param->set_out_max_val(false);
    // ArgMaxLayer<Dtype> layer(layer_param);
    // layer.SetUp(blob_bottom_vec, blob_top_vec);
    // cout << "Getting forward step results\n";
    // layer.Forward(blob_bottom_vec, blob_top_vec);

    // cout << "Counting error\n";
    // int label, max_index, errors = 0;
    // const Dtype* top_data = blob_top->cpu_data();
    // for(int c = 0; c < m ; c++){
    //   max_index = top_data[blob_top->offset(c,0,0,0)];
    //   label = data.blob_train_labels->cpu_data()[data.blob_train_labels->offset(max_index,0,0,0)];
    //   if(label != data.blob_test_labels->cpu_data()[data.blob_test_labels->offset(c,0,0,0)])
    //     errors++;
    // }
    // cout << "Error rate is " << errors*1.0/m << "\n";
    // // ============= ARGMAX LAYER ================

    //============= MANUAL RESULTS COUNTiNG ==============
    Dtype minimal_dist, current_dist;
    int minimal_index, label, errors = 0;

    cout << "Processing the results\n";
    for(int c = 0; c < m ; c++){
      minimal_dist = result_blob->mutable_cpu_data()[result_blob->offset(0, 0, c, 0)];
      minimal_index = 0;
      for(int r = 0; r < n ; r++){
        current_dist = result_blob->mutable_cpu_data()[result_blob->offset(0, 0, c, r)];
        if(current_dist < minimal_dist) {
          minimal_dist = current_dist;
          minimal_index = r;
        }
      }
      label = blob_label->cpu_data()[blob_label->offset(minimal_index,0,0,0)];
      if(c%100 == 0)
        cout << ".";

      if(label != blob_test_label->cpu_data()[blob_test_label->offset(c,0,0,0)])
          errors++;
    }
    cout << "\nError rate is " << errors*1.0/m << "\n";
    //============= MANUAL RESULTS COUNTiNG ==============

    delete result_blob;
    delete blob_label;
    delete blob_test_label;

    return 0;
}


