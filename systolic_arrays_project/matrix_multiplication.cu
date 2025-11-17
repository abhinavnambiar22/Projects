#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>

#define TILE_SIZE 16  

using namespace std;

__global__ void matrixMulShared(float *A, float *B, float *C, int N) {
    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;
    for (int i = 0; i < N / TILE_SIZE; i++) {
        tileA[threadIdx.y][threadIdx.x] = A[row * N + (i * TILE_SIZE + threadIdx.x)];
        tileB[threadIdx.y][threadIdx.x] = B[(i * TILE_SIZE + threadIdx.y) * N + col];
        __syncthreads();

        for (int j = 0; j < TILE_SIZE; j++)
            sum += tileA[threadIdx.y][j] * tileB[j][threadIdx.x];

        __syncthreads();
    }

    if (row < N && col < N)
        C[row * N + col] = sum;
}

void matrixMultiply(int N) {
    size_t bytes = N * N * sizeof(float);

    float *h_A = (float *)malloc(bytes);
    float *h_B = (float *)malloc(bytes);
    float *h_C = (float *)malloc(bytes);

    for (int i = 0; i < N * N; i++) {
        h_A[i] = rand() % 100;
        h_B[i] = rand() % 100;
    }

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid(N / TILE_SIZE, N / TILE_SIZE);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    
    matrixMulShared<<<grid, block>>>(d_A, d_B, d_C, N);
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    cout << "Matrix Multiplication Time: " << milliseconds << " ms" << endl;

    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);
}

int main() {
    int N = 1024;  // (1024x1024)
    matrixMultiply(N);
    return 0;
}
