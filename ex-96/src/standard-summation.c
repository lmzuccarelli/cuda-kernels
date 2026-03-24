#include <stdio.h>
#include <stdlib.h>

float standardSum(const float* data, int N) {
    float sum = 0.0f;
    for (int i = 0; i < N; i++) {
        sum += data[i];
    }
    return sum;
}

int main() {
    const int N = 1000000;
    float* data = (float*)malloc(N * sizeof(float));
    // Initialize data (for simplicity, all elements are 0.1f)
    for (int i = 0; i < N; i++) {
        data[i] = 0.1f;
    }
    
    float sum = standardSum(data, N);
    printf("Standard Sum: %f\n", sum);
    
    free(data);
    return 0;
}
