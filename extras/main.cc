#include <iostream>
#include <sstream>
#include <iomanip>
#include <armadillo>
#include "H5Cpp.h"
#include <openssl/md5.h>
using namespace arma;
using namespace H5;

std::vector<float> logrange(const float start, const float stop, const float step, const float base) {
    std::vector<float> values;
    for(float v = start; v < stop+step; v += step) {
        values.push_back(pow(base, v));
    }
    return values;
}

vec rbf_kernel_triu(const fmat& X, const fmat& Y, const float gamma) {
    fmat XX = sum(square(X), 0);
    fmat YY = sum(square(Y), 0).t();
    fmat XY = (2 * X.t())* Y;
    XX = ones<fmat>(X.n_cols, 1) * XX;
    YY = YY * ones<fmat>(1, X.n_cols);
    //XX = repmat(XX, X.n_cols, 1);
    //YY = repmat(YY, 1, X.n_cols);
    fmat D = XX + YY - XY;
    fmat K = exp(D * -gamma);
    auto upper_indices = trimatl_ind(size(K));
    return conv_to<vec>::from(K(upper_indices));
}

std::string compute_hash(const fmat& X) {
    unsigned char digest[16];
    char buf[sizeof(digest) * 2 + 1];
    MD5((const unsigned char*)X.memptr(), X.n_rows * X.n_cols * sizeof(float), digest);
    for(auto i = 0, j = 0; i < 16; i++, j += 2) {
        sprintf(buf+j, "%02x", digest[i]);
    }
    buf[sizeof(digest) * 2] = 0;
    return std::string(buf);
}

std::string compute_key(const std::string& data_hash, const float gamma) {
    char buffer[64];
    snprintf(buffer, sizeof(buffer), "%.16g", gamma);
    return "/" + data_hash + "/KernelType.RBF_" + buffer + "_0_0";
}

int main(int argc, char** argv) {
    if(argc < 3) {
        std::cout << "usage:" << std::endl;
        std::cout << "  ./main <file.h5> <step>" << std::endl;
        return 0;
    }
    fmat X;
    X.load(hdf5_name(argv[1], "data"));

    float step = std::stof(argv[2]);
    auto data_hash = compute_hash(X);
    auto gammas = logrange(-25, 5, step, 2);
    H5File file("gram.h5", H5F_ACC_TRUNC);
    FloatType datatype(PredType::NATIVE_FLOAT);
    file.createGroup("/" + data_hash);

    for(auto i = 0; i < gammas.size(); i++) {
        auto key = compute_key(data_hash, gammas[i]);
        std::cout << "key: " << key << std::endl;
        auto triu = rbf_kernel_triu(X, X, gammas[i]);

        hsize_t dim = triu.n_rows;
        DataSpace dataspace(1, &dim);

        DataSet dataset = file.createDataSet(key, datatype, dataspace);
        dataset.write(triu.memptr(), PredType::NATIVE_FLOAT);
    }

    return 0;
}
