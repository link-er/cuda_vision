#include <cstring>
#include <cstdlib>
#include <vector>
#include <iomanip>

#include "caffe/caffe.hpp"
#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "mnist.h"

using namespace caffe;
using namespace std;

int main(int argc, char** argv) {
    MNIST data("/home/stud/adilova/caffe/caffe-rc2/data/mnist/");

//    int n = 0;
//    for(int i=0;i<28;i++){
//        for(int j=0;j<28;j++){
//            int px = data.blob_train_images->cpu_data()[data.blob_train_images->offset(n,0,i,j)];
//            cout << setfill(' ') << setw(4) << px;
//        }
//        cout<<endl;
//    }
//    int label = data.blob_train_labels->cpu_data()[data.blob_train_labels->offset(n,0,0,0)];
//    cout<<"label: "<<label<<" "<<endl;

    int n = 60000;
    int m = 10000;

    Blob<Dtype>* blob_train_out = new Blob<Dtype>(1, 1, n, d);
    caffe_gpu_powx<Dtype>(blob_train_out->count(), train_data->gpu_data(), 2.0, blob_train_out->mutable_gpu_data());

    Blob<Dtype>* blob_test_out = new Blob<Dtype>(1, 1, m, d);
    caffe_gpu_powx<Dtype>(test_data->count(), test_data->gpu_data(), 2.0, blob_test_out->mutable_gpu_data());

    Blob<Dtype>* unary_blob1 = new Blob<Dtype>(1, 1, d, m);
    FillerParameter filler_param;
    filler_param.set_min(1);
    filler_param.set_max(1);
    UniformFiller<Dtype> filler(filler_param);
    filler.Fill(unary_blob1);

    Blob<Dtype>* result_blob = new Blob<Dtype>(1, 1, n, m);
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, n, m, d, 1., blob_train_out->gpu_data(),
                          unary_blob1->gpu_data(), 0., result_blob->mutable_gpu_data());

    Blob<Dtype>* unary_blob2 = new Blob<Dtype>(1, 1, n, d);
    filler.Fill(unary_blob2);

    Blob<Dtype>* temp_blob = new Blob<Dtype>(1, 1, n, m);
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, n, m, d, 1, unary_blob2->gpu_data(),
                           blob_test_out->gpu_data(), 0, temp_blob->mutable_gpu_data());

    caffe_gpu_add<Dtype>(result_blob->count(), result_blob->gpu_data(), temp_blob->gpu_data(), result_blob->mutable_gpu_data());

    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, n, m, d, 2., train_data->gpu_data(),
                          test_data->gpu_data(), 0., temp_blob->mutable_gpu_data());

    caffe_gpu_sub<Dtype>(result_blob->count(), result_blob->gpu_data(), temp_blob->gpu_data(), result_blob->mutable_gpu_data());

    Dtype minimal_dist;
    int minimal_index;
    Dtype current_dist;
    int errors = 0;
    for(int c = 0; c < m ; c++){
      minimal_dist = result_blob->mutable_cpu_data()[result_blob->offset(0, 0, 0, c)];
      minimal_index = 0;
      for(int r = 0; r < n ; r++){
        current_dist = result_blob->mutable_cpu_data()[result_blob->offset(0, 0, r, c)];
        if(current_dist < minimal_dist) {
          minimal_dist = current_dist;
          minimal_index = r;
        }
      }
//      classes[minimal_index];

//      if(current_class != classes[minimal_index])
//          errors++;
    }
    cout << "Error rate is " << errors*1.0/m << "\n";

    return 0;
}


