#include "mnist.h"

MNIST::MNIST(string path)
{
    //Read training data

    string filename_train_image = path + "train-images-idx3-ubyte";
    string filename_train_label = path + "train-labels-idx1-ubyte";
    string filename_test_image = path + "t10k-images-idx3-ubyte";
    string filename_test_label = path + "t10k-labels-idx1-ubyte";

    // Open files
    ifstream file_train_image(filename_train_image.c_str(), ios::in | ios::binary);
    ifstream file_train_label(filename_train_label.c_str(), ios::in | ios::binary);

    ifstream file_test_image(filename_test_image.c_str(), ios::in | ios::binary);
    ifstream file_test_label(filename_test_label.c_str(), ios::in | ios::binary);

    CHECK(file_train_image) << "Unable to open file " << filename_train_image;
    CHECK(file_train_label) << "Unable to open file " << filename_train_label;
    CHECK(file_test_image) << "Unable to open file " << filename_test_image;
    CHECK(file_test_label) << "Unable to open file " << filename_test_label;

    // Read the magic and the meta data
    uint32_t magic;

    file_train_image.read(reinterpret_cast<char*>(&magic), 4);
    magic = swap_endian(magic);
    CHECK_EQ(magic, 2051) << "Incorrect image file magic.";
    file_train_label.read(reinterpret_cast<char*>(&magic), 4);
    magic = swap_endian(magic);
    CHECK_EQ(magic, 2049) << "Incorrect label file magic.";

    file_test_image.read(reinterpret_cast<char*>(&magic), 4);
    magic = swap_endian(magic);
    CHECK_EQ(magic, 2051) << "Incorrect image file magic.";
    file_test_label.read(reinterpret_cast<char*>(&magic), 4);
    magic = swap_endian(magic);
    CHECK_EQ(magic, 2049) << "Incorrect label file magic.";


    file_train_image.read(reinterpret_cast<char*>(&num_train_images), 4);
    num_train_images = swap_endian(num_train_images);
    file_train_label.read(reinterpret_cast<char*>(&num_train_labels), 4);
    num_train_labels = swap_endian(num_train_labels);

    file_test_image.read(reinterpret_cast<char*>(&num_test_images), 4);
    num_test_images = swap_endian(num_test_images);
    file_test_label.read(reinterpret_cast<char*>(&num_test_labels), 4);
    num_test_labels = swap_endian(num_test_labels);


    int dim;

    CHECK_EQ(num_train_images, num_train_labels);
    file_train_image.read(reinterpret_cast<char*>(&rows), 4);
    rows = swap_endian(rows);
    file_train_image.read(reinterpret_cast<char*>(&cols), 4);
    cols = swap_endian(cols);
    dim = cols*rows;

    cout << "A total of " << num_train_images << " images." << endl;
    cout << "Rows: " << rows << " Cols: " << cols << endl;
    char label;
    char* pixels = new char[rows * cols];

    blob_train_images = new Blob<Dtype>(num_train_images, 1, rows, cols);
    blob_test_images = new Blob<Dtype>(num_test_images, 1, rows, cols);
    blob_train_labels = new Blob<Dtype>(num_train_labels, 1, 1, 1);
    blob_test_labels = new Blob<Dtype>(num_test_labels, 1, 1, 1);

    cout << "Training samples "<< num_train_images<< endl;
    cout << "Test samples "<< num_test_images<< endl;

    for (int item_id = 0; item_id < num_train_images; ++item_id) {
        file_train_image.read(pixels, rows * cols);
        file_train_label.read(&label, 1);
        // use pixels
        // use label
        for (int h = 0; h < rows; ++h) {
            for (int w = 0; w < cols; ++w) {
                unsigned char pixel = pixels[h*cols + w];
                blob_train_images->mutable_cpu_data()[blob_train_images->offset(item_id, 0, h, w)] = pixel;
            }
        }
        blob_train_labels->mutable_cpu_data()[item_id] = label;
    }

    for (int item_id = 0; item_id < num_test_images; ++item_id) {
        file_test_image.read(pixels, rows * cols);
        file_test_label.read(&label, 1);
        // use pixels
        // use label
        for (int h = 0; h < rows; ++h) {
            for (int w = 0; w < cols; ++w) {
                unsigned char pixel = pixels[h*cols + w];
                blob_test_images->mutable_cpu_data()[blob_test_images->offset(item_id, 0, h, w)] = pixel;
            }
        }
        blob_test_labels->mutable_cpu_data()[item_id] = label;
    }

    delete pixels;

    cout << "done reading data" << endl;

}

