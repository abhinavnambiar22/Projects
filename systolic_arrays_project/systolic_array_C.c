#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <opencv2/opencv.hpp>

#define MASK_WIDTH 3
#define MASK_RADIUS (MASK_WIDTH / 2)

// using a sobel filter
void convolveCPU(unsigned char *input, unsigned char *output, float *mask, int width, int height) {
    for (int row = 0; row < height; row++) {
        for (int col = 0; col < width; col++) {
            float result = 0.0f;

            for (int i = -MASK_RADIUS; i <= MASK_RADIUS; i++) {
                for (int j = -MASK_RADIUS; j <= MASK_RADIUS; j++) {
                    int r = row + i;
                    int c = col + j;
                    
                    if (r >= 0 && r < height && c >= 0 && c < width) {
                        result += mask[(i + MASK_RADIUS) * MASK_WIDTH + (j + MASK_RADIUS)] * input[r * width + c];
                    }
                }
            }

            result = fabs(result);
            result = fmin(fmax(result, 0.0f), 255.0f);

            output[row * width + col] = (unsigned char)result;
        }
    }
}

int main() {
    cv::Mat inputImage = cv::imread("input.jpg", cv::IMREAD_GRAYSCALE);
    if (inputImage.empty()) {
        printf("Error: Could not open or find the image!\n");
        return -1;
    }

    int width = inputImage.cols;
    int height = inputImage.rows;

    cv::Mat outputImage(height, width, CV_8UC1);

    float h_mask[MASK_WIDTH * MASK_WIDTH] = {-1, 0, 1,
                                             -2, 0, 2,
                                             -1, 0, 1};

    unsigned char *input = inputImage.data;
    unsigned char *output = outputImage.data;

    clock_t start, end;
    double cpu_time_used;

    start = clock();
    convolveCPU(input, output, h_mask, width, height);
    end = clock();

    cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC * 1000.0; 

    cv::imwrite("output_cpu.jpg", outputImage);
    printf("CPU execution time: %.3f ms\n", cpu_time_used);

    cv::imshow("Input Image", inputImage);
    cv::imshow("Output Image (CPU)", outputImage);
    cv::waitKey(0);

    return 0;
}
