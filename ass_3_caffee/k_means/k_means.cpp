#include <iostream>
#include <fstream>
#include <string>
#include <cstring>
#include <cstdlib>
#include <vector>

#include <stdio.h>
#include <time.h>

#include <opencv2/highgui/highgui.hpp>

#include "caffe/caffe.hpp"
#include "caffe/util/io.hpp"
#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"

using namespace caffe;
using namespace std;
using namespace cv;

typedef double Dtype;

void print_blob(Blob<Dtype>* blob, int rows, int cols) {
    for(int r = 0; r < rows ; r++){
      for(int c = 0; c < cols ; c++){
        cout << blob->mutable_cpu_data()[blob->offset(0, 0, r, c)] << " ";
      }
      cout << "\n";
    }
}

int main (int argc, char** argv)
{
    ifstream trainfile ("/home/stud/adilova/cuda_vision/ass_3_caffee/dataset1.txt");
    int n, d;
    float number;
    trainfile >> n >> d;
    cout << n << " " << d << "\n";
    float** train = new float*[n];
    for(int i=0; i<n; i++) {
        train[i] = new float[d];
        for(int j=0; j<d; j++) {
            trainfile >> number;
            train[i][j] = number;
        }
    }
    trainfile.close();

    Blob<Dtype>* const train_data = new Blob<Dtype>(1, 1, n, d);
    for(int r = 0; r < n ; r++){
       for(int c = 0; c < d ; c++){
          train_data->mutable_cpu_data()[train_data->offset(0, 0, r, c)] = train[r][c];
       }
    }

    ifstream classesfile ("/home/stud/adilova/cuda_vision/ass_3_caffee/classes1.txt");
    int temp, klass;
    classesfile >> n >> temp;
    cout << n << " " << temp << "\n";
    int* classes = new int[n];
    for(int i=0; i<n; i++) {
        classesfile >> klass;
        classes[i] = klass;
    }
    classesfile.close();

     Blob<Dtype>* const classes_data = new Blob<Dtype>(1, 1, 1, n);
     for(int r = 0; r < n ; r++){
        classes_data->mutable_cpu_data()[classes_data->offset(0, 0, 0, r)] = classes[r];
     }

    ifstream testfile ("/home/stud/adilova/cuda_vision/ass_3_caffee/test_dataset1.txt");
    int m;
    testfile >> m >> d;
    cout << m << " " << d << "\n";
    float** test = new float*[m];
    for(int i=0; i<m; i++) {
        test[i] = new float[d];
        for(int j=0; j<d; j++) {
            testfile >> number;
            test[i][j] = number;
        }
    }
    testfile.close();

    Blob<Dtype>* const test_data = new Blob<Dtype>(1, 1, m, d);
    for(int r = 0; r < m ; r++){
       for(int c = 0; c < d ; c++){
          test_data->mutable_cpu_data()[test_data->offset(0, 0, r, c)] = test[r][c];
       }
    }

    cout << "train data \n";
    print_blob(train_data, n, d);
    // square the train vectors
    Blob<Dtype>* blob_train_out = new Blob<Dtype>(1, 1, n, d);
    caffe_gpu_powx<Dtype>(blob_train_out->count(), train_data->gpu_data(), 2.0, blob_train_out->mutable_gpu_data());
    cout << "train data squared \n";
    print_blob(blob_train_out, n, d);

    cout << "test data \n";
    print_blob(test_data, m, d);
    // square the test vectors
    Blob<Dtype>* blob_test_out = new Blob<Dtype>(1, 1, m, d);
    caffe_gpu_powx<Dtype>(test_data->count(), test_data->gpu_data(), 2.0, blob_test_out->mutable_gpu_data());
    cout << "test data squared \n";
    print_blob(blob_test_out, m, d);

    // make 1 matrix, dxm
    Blob<Dtype>* unary_blob1 = new Blob<Dtype>(1, 1, d, m);
    FillerParameter filler_param;
    filler_param.set_min(1);
    filler_param.set_max(1);
    UniformFiller<Dtype> filler(filler_param);
    filler.Fill(unary_blob1);
    cout << "unary matrix \n";
    print_blob(unary_blob1, d, m);

    // gather results
    Blob<Dtype>* result_blob = new Blob<Dtype>(1, 1, n, m);
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, n, m, d, 1., blob_train_out->gpu_data(),
                          unary_blob1->gpu_data(), 0., result_blob->mutable_gpu_data());
    cout << "train squared multiplied by unary \n";
    print_blob(result_blob, n, m);

    // make 1 matrix, dxn
    Blob<Dtype>* unary_blob2 = new Blob<Dtype>(1, 1, n, d);
    filler.Fill(unary_blob2);
    cout << "unary matrix 2 \n";
    print_blob(unary_blob2, n, d);

    Blob<Dtype>* temp_blob = new Blob<Dtype>(1, 1, n, m);
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, n, m, d, 1, unary_blob2->gpu_data(),
                           blob_test_out->gpu_data(), 0, temp_blob->mutable_gpu_data());
    cout << "test squared multiplied by unary \n";
    print_blob(temp_blob, n, m);

    // keep the sum of them in the first blob
    caffe_gpu_add<Dtype>(result_blob->count(), result_blob->gpu_data(), temp_blob->gpu_data(), result_blob->mutable_gpu_data());
    cout << "sum of squared members \n";
    print_blob(result_blob, n, m);

    // find scaled multiplication of X and Z
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, n, m, d, 2, train_data->gpu_data(),
                          test_data->gpu_data(), 0, temp_blob->mutable_gpu_data());
    cout << "multiplication of train on test scaled by 2 \n";
    print_blob(temp_blob, n, m);

    // find final D
    caffe_gpu_sub<Dtype>(result_blob->count(), result_blob->gpu_data(), temp_blob->gpu_data(), result_blob->mutable_gpu_data());
    cout << "final result \n";
    print_blob(result_blob, n, m);

    // determine the closest for each of the test data
    Dtype minimal_dist;
    int minimal_index;
    Dtype current_dist;
    ofstream test_classes;
    test_classes.open("/home/stud/adilova/cuda_vision/ass_3_caffee/test_classes1.txt");
    for(int c = 0; c < m ; c++){
      cout << "Test vector number " << c << "--------------\n";
      minimal_dist = result_blob->mutable_cpu_data()[result_blob->offset(0, 0, 0, c)];
      minimal_index = 0;
      for(int r = 0; r < n ; r++){
        current_dist = result_blob->mutable_cpu_data()[result_blob->offset(0, 0, r, c)];
        if(current_dist < minimal_dist) {
          minimal_dist = current_dist;
          minimal_index = r;
        }
      }
      cout << "minimal distance is " << minimal_dist << " to train vector number " << minimal_index << "\n";
      cout << "class of the vectors is " << classes[minimal_index] << "\n";
      test_classes << classes[minimal_index] << "\n";
    }
    test_classes.close();

    for (int i = 0; i < n; ++i)
        delete [] train[i];
    delete [] train;
    delete [] classes;
    for (int i = 0; i < m; ++i)
        delete [] test[i];
    delete [] test;
    delete train_data;
    delete test_data;
    delete blob_train_out;
    delete blob_test_out;
    delete unary_blob1;
    delete unary_blob2;
    delete result_blob;
    delete temp_blob;

    return 0;
}

