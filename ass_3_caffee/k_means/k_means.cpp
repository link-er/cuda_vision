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

    Blob<Dtype>* const classes_data = new Blob<Dtype>(1, 1, 1, rows1);
    for(int r = 0; r < rows1 ; r++){
       classes_data->mutable_cpu_data()[classes_data->offset(0, 0, 0, r)] = classes[r];
    }

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

    Blob<Dtype>* blob_out = new Blob<Dtype>(1, 1, 1, rows1);
    Dtype asum;
    caffe_gpu_asum<Dtype>(rows1, classes_data->gpu_data(), &asum);
    cout<<"asum: "<<asum<<endl;
    caffe_gpu_scale<Dtype>(rows1, 10, classes_data->gpu_data(), blob_out->mutable_gpu_data());
    const Dtype* x = blob_out->cpu_data();
    for (int i = 0; i < rows1; ++i) cout<<x[i]<<endl;
    FillerParameter filler_param;
    filler_param.set_min(-3);
    filler_param.set_max(3);
    UniformFiller<Dtype> filler(filler_param);
    filler.Fill(blob);



//        ofstream multiplied;
//        multiplied.open("/home/stud/adilova/cuda_vision/ass_2_datasets/multiplied_test.txt");
//        for(int i=0; i<rows; i++) {
//            for(int j=0; j<cols; j++) {
//                multiplied << matrix[i][j] << " ";
//            }
//            multiplied << "\n";
//        }
//        multiplied.close();

    for (int i = 0; i < rows; ++i)
        delete [] train[i];
    delete [] train;
    delete [] classes;
    for (int i = 0; i < rows2; ++i)
        delete [] test[i];
    delete [] test;

    return 0;
}

