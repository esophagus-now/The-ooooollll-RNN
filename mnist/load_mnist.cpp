#include <cstdio>
#include <cstring>
#include <cstdint>
#include <cassert>
#include <vector>
#include <utility> //pair 

#include "load_mnist.h"

using namespace std;

typedef union {
	char c[4];
	uint32_t val;
} fix_endian;

static uint32_t get_val(fix_endian &x) {
	char tmp = x.c[3];
	x.c[3] = x.c[0];
	x.c[0] = tmp;
	tmp = x.c[1];
	x.c[1] = x.c[2];
	x.c[2] = tmp;
	return x.val;
}

/*

FILE FORMATS FOR THE MNIST DATABASE
The data is stored in a very simple file format designed for storing vectors and multidimensional matrices. General info on this format is given at the end of this page, but you don't need to read that to use the data files.
All the integers in the files are stored in the MSB first (high endian) format used by most non-Intel processors. Users of Intel processors and other low-endian machines must flip the bytes of the header.

There are 4 files:

train-images-idx3-ubyte: training set images
train-labels-idx1-ubyte: training set labels
t10k-images-idx3-ubyte:  test set images
t10k-labels-idx1-ubyte:  test set labels

The training set contains 60000 examples, and the test set 10000 examples.

The first 5000 examples of the test set are taken from the original NIST training set. The last 5000 are taken from the original NIST test set. The first 5000 are cleaner and easier than the last 5000.

TRAINING SET LABEL FILE (train-labels-idx1-ubyte):
[offset] [type]          [value]          [description]
0000     32 bit integer  0x00000801(2049) magic number (MSB first)
0004     32 bit integer  60000            number of items
0008     unsigned byte   ??               label
0009     unsigned byte   ??               label
........
xxxx     unsigned byte   ??               label
The labels values are 0 to 9.

TRAINING SET IMAGE FILE (train-images-idx3-ubyte):
[offset] [type]          [value]          [description]
0000     32 bit integer  0x00000803(2051) magic number
0004     32 bit integer  60000            number of images
0008     32 bit integer  28               number of rows
0012     32 bit integer  28               number of columns
0016     unsigned byte   ??               pixel
0017     unsigned byte   ??               pixel
........
xxxx     unsigned byte   ??               pixel
Pixels are organized row-wise. Pixel values are 0 to 255. 0 means background (white), 255 means foreground (black).

TEST SET LABEL FILE (t10k-labels-idx1-ubyte):
[offset] [type]          [value]          [description]
0000     32 bit integer  0x00000801(2049) magic number (MSB first)
0004     32 bit integer  10000            number of items
0008     unsigned byte   ??               label
0009     unsigned byte   ??               label
........
xxxx     unsigned byte   ??               label
The labels values are 0 to 9.

TEST SET IMAGE FILE (t10k-images-idx3-ubyte):
[offset] [type]          [value]          [description]
0000     32 bit integer  0x00000803(2051) magic number
0004     32 bit integer  10000            number of images
0008     32 bit integer  28               number of rows
0012     32 bit integer  28               number of columns
0016     unsigned byte   ??               pixel
0017     unsigned byte   ??               pixel
........
xxxx     unsigned byte   ??               pixel
Pixels are organized row-wise. Pixel values are 0 to 255. 0 means background (white), 255 means foreground (black).

*/

vector<tpair> load_mnist_training() {
	vector<tpair> ret;

	FILE *train_imgs = fopen("train-images-idx3-ubyte", "rb");
	if (!train_imgs) {
		perror("Could not open training images");
		throw "whatever";
	}
	
	FILE *train_lbls = fopen("train-labels-idx1-ubyte", "rb");
	if (!train_lbls) {
		perror("Could not open training labels");
		throw "whatever";
	}
	
	fix_endian x;

	fread(&x, 4, 1, train_imgs);
	uint32_t img_magic = get_val(x);
	if (img_magic != 0x803) {
		fprintf(stderr, "Marco got the endianness wrong.\n");
		throw "stupid endianness";
	}

	fread(&x, 4, 1, train_lbls);
	uint32_t lbl_magic = get_val(x);
	if (lbl_magic != 0x801) {
		fprintf(stderr, "Marco got the endianness wrong.\n");
		throw "stupid endianness";
	}

	fread(&x, 4, 1, train_imgs);
	uint32_t num_imgs = get_val(x);
	fread(&x, 4, 1, train_lbls);
	uint32_t num_lbls = get_val(x);

	assert(num_imgs == num_lbls);

	fread(&x, 4, 1, train_imgs);
	uint32_t img_rows = get_val(x);
	fread(&x, 4, 1, train_imgs);
	uint32_t img_cols = get_val(x);
	assert(img_rows == 28);
	assert(img_cols == 28);

	//Okay. We validated all our inputs, so now we can write some 
	//simpler code 

	int i;
	for (i = 0; i < num_imgs; i++) {
		unsigned char img[784];
		unsigned char lbl;

		fread(img, 1, 784, train_imgs);
		fread(&lbl, 1, 1, train_lbls);

		vector<float> fimg;
		fimg.reserve(784);

		vector<float> flbl(10, 0.0);

		int j;
		for (j = 0; j < 784; j++) {
			fimg.push_back(float(img[j])/255.0);
		}

		flbl[lbl] = 1.0;

		ret.push_back({fimg, flbl});
	}

	fclose(train_imgs);
	fclose(train_lbls);

	return ret;
}