#include <stdio.h>
#include <stdlib.h>

float pairwiseSum(const float* data, int start, int end) {
    if (end - start == 1)
        return data[start];
    int mid = start + (end - start) / 2;
    return pairwiseSum(data, start, mid) + pairwiseSum(data, mid, end);
}

int main() {
    const int N = 1000000;
    float* data = (float*)malloc(N * sizeof(float));
    // Initialize data (e.g., all elements are 0.1f)
    for (int i = 0; i < N; i++)
        data[i] = 0.1f;
    float sum = pairwiseSum(data, 0, N);
    printf("Pairwise Sum: %f\n", sum);
    free(data);
    return 0;
}
