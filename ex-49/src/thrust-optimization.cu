#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/transform.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <thrust/copy.h>
#include <iostream>

struct multiply_by_factor {
    float factor;
    multiply_by_factor(float f) : factor(f) {}
    __host__ __device__
    float operator()(const float &x) const {
        return x * factor;
    }
};

int main(){
    int N = 10;
    thrust::host_vector<float> h_vec(N);
    for(int i=0; i<N; i++){
        h_vec[i]=(float)(rand()%100);
    }

    thrust::device_vector<float> d_vec = h_vec;

    thrust::transform(d_vec.begin(), d_vec.end(),
                      d_vec.begin(), // in place
                      multiply_by_factor(2.5f));

    thrust::sort(d_vec.begin(), d_vec.end());

    float sum = thrust::reduce(d_vec.begin(), d_vec.end(), 0.0f, thrust::plus<float>());
    std::cout << "Sum after transform & sort= " << sum << std::endl;

    thrust::copy(d_vec.begin(), d_vec.end(), h_vec.begin());
    std::cout << "Sorted *2.5 data:\n";
    for(int i=0; i<N; i++){
        std::cout << h_vec[i] << " ";
    }
    std::cout<<"\n";

    return 0;
}
