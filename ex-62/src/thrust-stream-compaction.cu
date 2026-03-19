#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/copy.h>
#include <thrust/remove.h>
#include <iostream>
#include <cstdlib>
#include <ctime>

struct IsNonZero {
    __host__ __device__
    bool operator()(const int x) const {
        return x != 0;
    }
};

int main() {
    int N = 1000000; 

    std::srand(static_cast<unsigned int>(std::time(0)));

    thrust::host_vector<int> h_vec(N);
    for (int i = 0; i < N; i++) {
        h_vec[i] = (std::rand() % 10 < 3) ? 0 : (std::rand() % 1000 + 1);
    }

    thrust::device_vector<int> d_vec = h_vec;

    thrust::device_vector<int> d_compact(N);
    auto new_end = thrust::copy_if(d_vec.begin(), d_vec.end(), d_compact.begin(), IsNonZero());

    int new_size = new_end - d_compact.begin();
    std::cout << "Original size: " << N << ", Compacted size: " << new_size << std::endl;

    thrust::host_vector<int> h_compact = d_compact;
    std::cout << "First 10 elements of compacted array:" << std::endl;
    for (int i = 0; i < 10 && i < new_size; i++) {
        std::cout << h_compact[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}
