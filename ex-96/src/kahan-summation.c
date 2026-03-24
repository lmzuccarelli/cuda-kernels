#include <stdio.h>
#include <stdlib.h>

float kahanSum(const float* data, int N) {
    float sum = 0.0f;
    float c = 0.0f; // A running compensation for lost low-order bits.
    for (int i = 0; i < N; i++) {
        float y = data[i] - c;
        float t = sum + y;
        c = (t - sum) - y;
        sum = t;
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
    
    float sum = kahanSum(data, N);
    printf("Kahan Sum: %f\n", sum);
    
    free(data);
    return 0;
}
