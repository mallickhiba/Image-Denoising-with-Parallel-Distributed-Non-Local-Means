#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "mpi.h"

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

void non_local_means_denoise(unsigned char *image, unsigned char *denoised_image, int width, int height, int rank, int size) {
    int half_patch_size = PATCH_SIZE / 2;
    int rows_per_process = (height + size - 1) / size;  // Ensure all rows are covered
    int start_row = rank * rows_per_process;
    int end_row = (start_row + rows_per_process > height) ? height : start_row + rows_per_process;

    for (int y = start_row; y < end_row; y++) {
        for (int x = 0; x < width; x++) {
            for (int c = 0; c < 3; c++) {
                float weighted_avg = weighted_average(image, x, y, width, height, c);
                denoised_image[((y - start_row) * width + x) * 3 + c] = (unsigned char)weighted_avg;
            }
        }
    }
}

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc != 2) {
        if (rank == 0)
            printf("Usage: %s <input_image.jpg>\n", argv[0]);
        MPI_Finalize();
        return 1;
    }

    char *input_filename = argv[1];
    char *output_filename = "denoised_image.jpg";
    char *residual_filename = "residual_image.jpg";

    int width, height, channels;
    unsigned char *image = NULL;
    if (rank == 0) {
        image = stbi_load(input_filename, &width, &height, &channels, 3);
        if (image == NULL) {
            printf("Error loading the image.\n");
            MPI_Finalize();
            return 1;
        }
    }

    MPI_Bcast(&width, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&height, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&channels, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank != 0) {
        image = (unsigned char *)malloc(width * height * channels * sizeof(unsigned char));
        if (image == NULL) {
            printf("Error allocating memory for the image on process %d.\n", rank);
            MPI_Finalize();
            return 1;
        }
    }
    MPI_Bcast(image, width * height * channels, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

    int rows_per_process = (height + size - 1) / size;
    unsigned char *local_denoised_image = (unsigned char *)malloc(rows_per_process * width * channels * sizeof(unsigned char));
    if (local_denoised_image == NULL) {
        printf("Error allocating memory for the local denoised image on process %d.\n", rank);
        if (rank != 0) free(image);
        MPI_Finalize();
        return 1;
    }

    non_local_means_denoise(image, local_denoised_image, width, height, rank, size);

    unsigned char *global_denoised_image = NULL;
    if (rank == 0) {
        global_denoised_image = (unsigned char *)malloc(width * height * channels * sizeof(unsigned char));
        if (global_denoised_image == NULL) {
            printf("Error allocating memory for the global denoised image.\n");
            free(image);
            free(local_denoised_image);
            MPI_Finalize();
            return 1;
        }
    }

    MPI_Gather(local_denoised_image, rows_per_process * width * channels, MPI_UNSIGNED_CHAR,
               global_denoised_image, rows_per_process * width * channels, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        stbi_write_jpg(output_filename, width, height, channels, global_denoised_image, 100);

        unsigned char *residual_image = (unsigned char *)malloc(width * height * channels * sizeof(unsigned char));
        if (residual_image == NULL) {
            printf("Error allocating memory for the residual image.\n");
            free(image);
            free(global_denoised_image);
            MPI_Finalize();
            return 1;
        }

        for (int i = 0; i < width * height * channels; i++) {
            residual_image[i] = abs(image[i] - global_denoised_image[i]);
        }

        stbi_write_jpg(residual_filename, width, height, channels, residual_image, 100);

        free(image);
        free(global_denoised_image);
        free(residual_image);

        printf("Denoising completed. Denoised image and residual image saved.\n");
    }

    free(local_denoised_image);
    if (rank != 0) {
        free(image);
    }

    MPI_Finalize();
    return 0;
}