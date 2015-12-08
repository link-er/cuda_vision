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

int main (int argc, char** argv)
{
    ifstream trainfile ("/home/stud/adilova/cuda_vision/ass_3_caffee/dataset1.txt");
    int rows, cols;
    float number;
    trainfile >> rows >> cols;
    cout << rows << " " << cols << "\n";
    float** train = new float*[rows];
    for(int i=0; i<rows; i++) {
        train[i] = new float[cols];
        for(int j=0; j<cols; j++) {
            trainfile >> number;
            train[i][j] = number;
        }
    }
    trainfile.close();

    Blob<Dtype>* const train_data = new Blob<Dtype>(1, 1, rows, cols);
    for(int r = 0; r < rows ; r++){
       for(int c = 0; c < cols ; c++){
          train_data->mutable_cpu_data()[train_data->offset(0, 0, r, c)] = train[r][c];
       }
    }

    ifstream classesfile ("/home/stud/adilova/cuda_vision/ass_3_caffee/classes1.txt");
    int rows1;
    int klass;
    classesfile >> rows1 >> cols;
    cout << rows1 << " " << cols << "\n";
    int* classes = new int[rows1];
    for(int i=0; i<rows1; i++) {
        classesfile >> klass;
        classes[i] = klass;
    }
    classesfile.close();

    // Blob<Dtype>* const classes_data = new Blob<Dtype>(1, 1, 1, rows1);
    // for(int r = 0; r < rows1 ; r++){
    //    classes_data->mutable_cpu_data()[classes_data->offset(0, 0, 0, r)] = classes[r];
    // }

    ifstream testfile ("/home/stud/adilova/cuda_vision/ass_3_caffee/test_dataset1.txt");
    int rows2, cols2;
    testfile >> rows2 >> cols2;
    cout << rows2 << " " << cols2 << "\n";
    float** test = new float*[rows2];
    for(int i=0; i<rows2; i++) {
        test[i] = new float[cols2];
        for(int j=0; j<cols2; j++) {
            testfile >> number;
            test[i][j] = number;
        }
    }
    testfile.close();

    Blob<Dtype>* const test_data = new Blob<Dtype>(1, 1, rows2, cols2);
    for(int r = 0; r < rows2 ; r++){
       for(int c = 0; c < cols2 ; c++){
          test_data->mutable_cpu_data()[test_data->offset(0, 0, r, c)] = test[r][c];
       }
    }

    // square the train vectors
    Blob<Dtype>* blob_train_out = new Blob<Dtype>(1, 1, rows, cols);
    int train_count = train_data->count();
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, rows, rows, cols, 1, train_data->cpu_data(), train_data->cpu_data(), 0,
                          blob_train_out->mutable_cpu_data());


   for(int r = 0; r < rows ; r++){
     for(int c = 0; c < cols ; c++){
        cout << "from " << train_data->cpu_data()[train_data->offset(0, 0, r, c)];
        cout << " got " << blob_train_out->cpu_data()[blob_train_out->offset(0, 0, r, c)] << "\n";
      }
    }

//    // square the test vectors
//    Blob<Dtype>* blob_test_out = new Blob<Dtype>(1, 1, rows2, cols2);
//    int test_count = test_data->count();
//    caffe_gpu_powx<Dtype>(test_count, test_data->gpu_data(), 2, blob_test_out->mutable_gpu_data());
//    // make 1 matrix, mxn
//    Blob<Dtype>* unary_blob = new Blob<Dtype>(1, 1, rows, rows2);
//    FillerParameter filler_param;
//    filler_param.set_min(1);
//    filler_param.set_max(1);
//    UniformFiller<Dtype> filler(filler_param);
//    filler.Fill(unary_blob);
//    // gather results
//    Blob<Dtype>* result_blob = new Blob<Dtype>(1, 1, rows, rows2);
//    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, rows, rows2, rows, 1., blob_train_out->gpu_data(), unary_blob->gpu_data(), 0., result_blob->mutable_gpu_data());
//    Blob<Dtype>* temp_blob = new Blob<Dtype>(1, 1, rows, rows2);
//    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, rows, rows2, rows2, 1, unary_blob->gpu_data(), blob_test_out->gpu_data(), 0, temp_blob->mutable_gpu_data());
//    // keep the sum of them in the first blob
//    int result_count = unary_blob->count();
//    caffe_gpu_add<Dtype>(result_count, result_blob->gpu_data(), temp_blob->gpu_data(), result_blob->mutable_gpu_data());
//    // find scaled multiplication of X and Z
//    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, rows, rows2, cols, 2, train_data->gpu_data(), test_data->gpu_data(), 0, temp_blob->mutable_gpu_data());
//    // find final D
//    caffe_gpu_sub<Dtype>(result_count, result_blob->gpu_data(), temp_blob->gpu_data(), result_blob->mutable_gpu_data());

//    // determine the closest for each of the test data
//    Dtype minimal_dist;
//    int minimal_index;
//    Dtype current_dist;
//    ofstream test_classes;
//    test_classes.open("/home/stud/adilova/cuda_vision/ass_3_datasets/test_classes1.txt");
//    for(int c = 0; c < rows2 ; c++){
//      cout << "Test vector number " << c << "--------------\n";
//      minimal_dist = result_blob->mutable_cpu_data()[result_blob->offset(0, 0, 0, c)];
//      minimal_index = 0;
//      for(int r = 0; r < rows ; r++){
//        current_dist = result_blob->mutable_cpu_data()[result_blob->offset(0, 0, r, c)];
//        if(current_dist < minimal_dist) {
//          minimal_dist = current_dist;
//          minimal_index = r;
//        }
//      }
//      cout << "minimal distance is " << minimal_dist << " to train vector number " << minimal_index << "\n";
//      cout << "class of the vectors is " << classes[minimal_index] << "\n";
//      test_classes << classes[minimal_index] << "\n";
//    }
//    test_classes.close();

//    for (int i = 0; i < rows; ++i)
//        delete [] train[i];
//    delete [] train;
//    delete [] classes;
//    for (int i = 0; i < rows2; ++i)
//        delete [] test[i];
//    delete [] test;
//    delete train_data;
//    delete test_data;
//    delete blob_train_out;
//    delete blob_test_out;
//    delete unary_blob;
//    delete result_blob;
//    delete temp_blob;

    return 0;
}

