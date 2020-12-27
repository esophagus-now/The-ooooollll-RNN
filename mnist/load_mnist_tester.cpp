#include <iostream>
#include <vector>
#include <algorithm>
#include "load_mnist.h"

using namespace std;

template <typename T>
ostream& operator<<(ostream &o, vector<T> const& v) {
	o << "[";
	auto delim = "";
	for (auto const& i : v) {
		cout << delim << i;
		delim = ",";
	}

	return o << "]";
}

int main() {
	vector<tpair> thing = load_mnist_training();

	int i;
	for (i = 0; i < 5; i++) {
		cout << "Image is " << thing[i].first << endl;
		cout << "Label = " << thing[i].second << endl;
	}
}