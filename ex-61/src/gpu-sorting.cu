#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <iostream>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <algorithm> 

int main() {
    int N = 10000000;
    
    std::srand(static_cast<unsigned int>(std::time(0)));
    
    thrust::host_vector<int> h_vec(N);
    for (int i = 0; i < N; i++) {
        h_vec[i] = std::rand() % 100000;
    }
    
    thrust::device_vector<int> d_vec = h_vec;
    
    auto gpu_start = std::chrono::high_resolution_clock::now();
    
    thrust::sort(d_vec.begin(), d_vec.end());
    
    auto gpu_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> gpu_sort_time = gpu_end - gpu_start;
    
    thrust::host_vector<int> sorted_vec = d_vec;
    
    std::cout << "GPU Sort Sample: First element = " << sorted_vec[0] << std::endl;
    std::cout << "GPU Sort Time: " << gpu_sort_time.count() << " ms" << std::endl;
    
    auto cpu_start = std::chrono::high_resolution_clock::now();
    std::sort(h_vec.begin(), h_vec.end());
    auto cpu_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpu_sort_time = cpu_end - cpu_start;
    std::cout << "CPU Sort Time: " << cpu_sort_time.count() << " ms" << std::endl;
    
    return 0;
}
