#include <stdio.h>
#include <stdlib.h>
#include <math.h>

float neumaierSum(const float* data, int N) {
    float sum = data[0];
    float c = 0.0f;
    for (int i = 1; i < N; i++) {
        float t = sum + data[i];
        if (fabs(sum) >= fabs(data[i]))
            c += (sum - t) + data[i];
        else
            c += (data[i] - t) + sum;
        sum = t;
    }
    return sum + c;
}

int main() {
    const int N = 1000000;
    float* data = (float*)malloc(N * sizeof(float));
    // Initialize data (e.g., all elements are 0.1f)
    for (int i = 0; i < N; i++)
        data[i] = 0.1f;
    float sum = neumaierSum(data, N);
    printf("Neumaier Sum: %f\n", sum);
    free(data);
    return 0;
}
