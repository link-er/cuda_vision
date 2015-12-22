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
    int d = 28*28;

    data.blob_train_images->Reshape(1, 1, n, d);
    data.blob_test_images->Reshape(1, 1, m, d);

    cout << "Reshaped\n";

    Blob<Dtype>* blob_train_out = new Blob<Dtype>(1, 1, n, d);
    caffe_gpu_powx<Dtype>(data.blob_train_images->count(), data.blob_train_images->gpu_data(), 2.0, blob_train_out->mutable_gpu_data());

    Blob<Dtype>* blob_test_out = new Blob<Dtype>(1, 1, m, d);
    caffe_gpu_powx<Dtype>(data.blob_test_images->count(), data.blob_test_images->gpu_data(), 2.0, blob_test_out->mutable_gpu_data());

    cout << "Squared\n";

    Blob<Dtype>* unary_blob1 = new Blob<Dtype>(1, 1, d, m);
    FillerParameter filler_param;
    filler_param.set_min(1);
    filler_param.set_max(1);
    UniformFiller<Dtype> filler(filler_param);
    filler.Fill(unary_blob1);

    Blob<Dtype>* result_blob = new Blob<Dtype>(1, 1, n, m);
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, n, m, d, 1., blob_train_out->gpu_data(),
                          unary_blob1->gpu_data(), 0., result_blob->mutable_gpu_data());
    delete unary_blob1;
    delete blob_train_out;

    Blob<Dtype>* unary_blob2 = new Blob<Dtype>(1, 1, n, d);
    filler.Fill(unary_blob2);

    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, n, m, d, 1, unary_blob2->gpu_data(),
                           blob_test_out->gpu_data(), 1, result_blob->mutable_gpu_data());
    delete unary_blob2;
    delete blob_test_out;
    cout << "Multiplied by unary and added the first one multiplied\n";

    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, n, m, d, -2., data.blob_train_images->gpu_data(),
                          data.blob_test_images->gpu_data(), 1., result_blob->mutable_gpu_data());
    cout << "Got final distances\n";

    Dtype minimal_dist, current_dist;
    int minimal_index, errors = 0;

    cout << "Processing the results\n";
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
      int label = data.blob_train_labels->cpu_data()[data.blob_train_labels->offset(minimal_index,0,0,0)];
      cout << ".";

      if(label != data.blob_test_labels->cpu_data()[data.blob_test_labels->offset(c,0,0,0)])
          errors++;
    }
    cout << "\nError rate is " << errors*1.0/m << "\n";

    delete result_blob;
    delete data.blob_train_images;
    delete data.blob_test_images;
    delete data.blob_train_labels;
    delete data.blob_test_labels;

    return 0;
}


