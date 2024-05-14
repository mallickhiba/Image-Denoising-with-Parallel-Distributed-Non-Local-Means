#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define PATCH_SIZE 7
#define SEARCH_WINDOW_SIZE 21
#define H 10.0

float weighted_average(unsigned char *image, int x, int y, int width, int height, int channel) {
    float sum = 0.0;
    float total_weight = 0.0;
    int half_patch_size = PATCH_SIZE / 2;

    for (int i = -half_patch_size; i <= half_patch_size; i++) {
        for (int j = -half_patch_size; j <= half_patch_size; j++) {
            int current_x = x + i;
            int current_y = y + j;
            if (current_x >= 0 && current_x < width && current_y >= 0 && current_y < height) {
                int diff = image[(current_y * width + current_x) * 3 + channel] - image[(y * width + x) * 3 + channel];
                float weight = exp(-((float)(i * i + j * j)) / (2 * H * H));
                sum += weight * image[(current_y * width + current_x) * 3 + channel];
                total_weight += weight;
            }
        }
    }

    return sum / total_weight;
}


void non_local_means_denoise(unsigned char *image, unsigned char *denoised_image, int width, int height) {
    int half_patch_size = PATCH_SIZE / 2;
    int half_search_window_size = SEARCH_WINDOW_SIZE / 2;

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            for (int c = 0; c < 3; c++) {
                float weighted_avg = weighted_average(image, x, y, width, height, c);
                denoised_image[(y * width + x) * 3 + c] = (unsigned char)weighted_avg;
            }
        }
    }
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        printf("Usage: %s <input_image.jpg>\n", argv[0]);
        return 1;
    }

    char *input_filename = argv[1];
    char *output_filename = "denoised_image.jpg";
    char *residual_filename = "residual_image.jpg";

    int width, height, channels;
    unsigned char *image = stbi_load(input_filename, &width, &height, &channels, 3);
    if (image == NULL) {
        printf("Error loading the image.\n");
        return 1;
    }

    unsigned char *denoised_image = (unsigned char *)malloc(width * height * channels * sizeof(unsigned char));
    if (denoised_image == NULL) {
        printf("Error allocating memory.\n");
        stbi_image_free(image);
        return 1;
    }

    // Perform non-local means denoising
    non_local_means_denoise(image, denoised_image, width, height);

    // Calculate residual image
    unsigned char *residual_image = (unsigned char *)malloc(width * height * channels * sizeof(unsigned char));
    if (residual_image == NULL) {
        printf("Error allocating memory.\n");
        stbi_image_free(image);
        free(denoised_image);
        return 1;
    }

    for (int i = 0; i < width * height * channels; i++) {
        residual_image[i] = abs(image[i] - denoised_image[i]);
    }

    // Save denoised image and residual image
    stbi_write_jpg(output_filename, width, height, channels, denoised_image, 100);
    stbi_write_jpg(residual_filename, width, height, channels, residual_image, 100);

    // Free memory
    stbi_image_free(image);
    free(denoised_image);
    free(residual_image);

    printf("Denoising completed. Denoised image and residual image saved.\n");

    return 0;
} 