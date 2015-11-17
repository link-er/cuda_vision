#include <iostream>
#include "multiplier.h"
#include <fstream>
#include <string>
using namespace std;


int main (int argc, char** argv)
{
    string line;
    ifstream datafile ("/home/stud/adilova/cuda_vision/ass_2_datasets/test.txt");
    int rows, cols;
    float number;
    if (datafile.is_open())
    {
        datafile >> rows >> cols;
        cout << rows << " " << cols << "\n";
        float** matrix = new float*[rows];
        for(int i=0; i<rows; i++) {
            matrix[i] = new float[cols];
            for(int j=0; j<cols; j++) {
                datafile >> number;
                matrix[i][j] = number;
            }
        }
        datafile.close();

        cout << "multiply=============\n";
        MULTIPLIER multpl(rows, cols);
        multpl.compute(matrix);

        ofstream multiplied;
        multiplied.open("/home/stud/adilova/cuda_vision/ass_2_datasets/multiplied_test.txt");
        for(int i=0; i<rows; i++) {
            for(int j=0; j<cols; j++) {
                multiplied << matrix[i][j] << " ";
            }
            multiplied << "\n";
        }
        multiplied.close();

        for (int i = 0; i < rows; ++i)
            delete [] matrix[i];
        delete [] matrix;
    }

    else cout << "Unable to open file";

    return 0;
}

