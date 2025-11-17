#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <iostream>

#define TILE_WIDTH 16
#define MASK_WIDTH 3
#define MASK_RADIUS (MASK_WIDTH / 2)

using namespace cv;
using namespace std;

// applying sobel operator
__global__ void convolveSystolic(unsigned char *input, unsigned char *output, float *mask, int width, int height) {
    __shared__ float tile[TILE_WIDTH + MASK_WIDTH - 1][TILE_WIDTH + MASK_WIDTH - 1];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * TILE_WIDTH + ty;
    int col = blockIdx.x * TILE_WIDTH + tx;
    
    int sharedRow = ty + MASK_RADIUS;
    int sharedCol = tx + MASK_RADIUS;

    if (row < height && col < width) {
        tile[sharedRow][sharedCol] = input[row * width + col];
    } else {
        tile[sharedRow][sharedCol] = 0.0f;
    }

    if (tx < MASK_RADIUS) {
        tile[sharedRow][tx] = (col >= MASK_RADIUS) ? input[row * width + col - MASK_RADIUS] : 0.0f;
    }
    if (tx >= TILE_WIDTH - MASK_RADIUS) {
        tile[sharedRow][sharedCol + MASK_RADIUS] = (col + MASK_RADIUS < width) ? input[row * width + col + MASK_RADIUS] : 0.0f;
    }
    if (ty < MASK_RADIUS) {
        tile[ty][sharedCol] = (row >= MASK_RADIUS) ? input[(row - MASK_RADIUS) * width + col] : 0.0f;
    }
    if (ty >= TILE_WIDTH - MASK_RADIUS) {
        tile[sharedRow + MASK_RADIUS][sharedCol] = (row + MASK_RADIUS < height) ? input[(row + MASK_RADIUS) * width + col] : 0.0f;
    }

    __syncthreads();

    float result = 0.0f;
    if (row < height && col < width) {
        for (int i = 0; i < MASK_WIDTH; i++) {
            for (int j = 0; j < MASK_WIDTH; j++) {
                result += mask[i * MASK_WIDTH + j] * tile[sharedRow - MASK_RADIUS + i][sharedCol - MASK_RADIUS + j];
            }
        }

        result = fabs(result);  
        result = min(max(result, 0.0f), 255.0f); 

        output[row * width + col] = static_cast<unsigned char>(result);
    }
}

int main() {
    Mat inputImage = imread("input.jpg", IMREAD_GRAYSCALE);
    if (inputImage.empty()) {
        cerr << "Error: Could not open or find the image!" << endl;
        return -1;
    }

    int width = inputImage.cols;
    int height = inputImage.rows;

    Mat outputImage(height, width, CV_8UC1);

    float h_mask[MASK_WIDTH * MASK_WIDTH] = {-1, 0, 1,
                                             -2, 0, 2,
                                             -1, 0, 1};

    unsigned char *d_input, *d_output;
    float *d_mask;

    cudaMalloc(&d_input, width * height * sizeof(unsigned char));
    cudaMalloc(&d_output, width * height * sizeof(unsigned char));
    cudaMalloc(&d_mask, MASK_WIDTH * MASK_WIDTH * sizeof(float));

    cudaMemcpy(d_input, inputImage.data, width * height * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mask, h_mask, MASK_WIDTH * MASK_WIDTH * sizeof(float), cudaMemcpyHostToDevice);

    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 dimGrid((width + TILE_WIDTH - 1) / TILE_WIDTH, (height + TILE_WIDTH - 1) / TILE_WIDTH);

    cudaEvent_t start, stop;
    float milliseconds = 0;

    cudaEventCreate(&start); // start timer
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    convolveSystolic<<<dimGrid, dimBlock>>>(d_input, d_output, d_mask, width, height);
    cudaEventRecord(stop);

    cudaDeviceSynchronize(); 

    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        cerr << "CUDA error: " << cudaGetErrorString(err) << endl;
        return -1;
    }

    cudaMemcpy(outputImage.data, d_output, width * height * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    imwrite("output.jpg", outputImage);

    cout << "Kernel execution time: " << milliseconds << " ms" << endl;

    imshow("Input Image", inputImage);
    imshow("Output Image", outputImage);
    waitKey(0);

    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_mask);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    destroyAllWindows();
    return 0;
}
