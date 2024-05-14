#ifndef __UTILS_CUH__
#define __UTILS_CUH__

#include <iostream>
#include <vector>
#include <cmath>
#include <string>
#include <iterator>
#include <fstream>
#include <sstream>
#include <chrono>

extern __shared__ float s[];

namespace util {

/* -------------------------------------------------------------------------- */
/*                                    timer                                   */
/* -------------------------------------------------------------------------- */

class Timer {
  public:
    Timer(bool print) : print(print) {}

    void start(std::string operation_desc)
    {
        _operation_desc = operation_desc;
        t1              = std::chrono::high_resolution_clock::now();
    }

    void stop()
    {
        t2       = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
        if (print)
            std::cout << "\n" << _operation_desc << " time: " << duration / 1e3 << "ms\n" << std::endl;
    }

  private:
    float duration;
    bool print;
    std::string _operation_desc;
    std::chrono::high_resolution_clock::time_point t1, t2;
};

/* -------------------------------------------------------------------------- */
/*                             host & device utils                            */
/* -------------------------------------------------------------------------- */

__host__ __device__ bool isInBounds(int n, int x, int y) 
{
    return x >= 0 && x < n && y >= 0 && y < n;
}

__host__ __device__ float computeWeight(float dist, float sigma) // compute weight without "/z(i)" division
{
    return expf(-dist / (sigma * sigma));
}

// patch-to-patch euclidean distance
__host__ __device__ float computePatchDistance( float * image, 
                                                float * _weights, 
                                                int n, 
                                                int patchSize, 
                                                int p1_rowStart, 
                                                int p1_colStart, 
                                                int p2_rowStart, 
                                                int p2_colStart ) 
{
    float ans = 0;
    float temp;

    for (int i = 0; i < patchSize; i++) {
        for (int j = 0; j < patchSize; j++) {
            if (isInBounds(n, p1_rowStart + i, p1_colStart + j) && isInBounds(n, p2_rowStart + i, p2_colStart + j)) {
                temp = image[(p1_rowStart + i) * n + p1_colStart + j] - image[(p2_rowStart + i) * n + p2_colStart + j];
                ans +=  _weights[i * patchSize + j] * temp * temp;
            }
        }
    }

    return ans;
}

/* -------------------------------------------------------------------------- */
/*                                 host utils                                 */
/* -------------------------------------------------------------------------- */

float * computeInsideWeights(int patchSize, float patchSigma)
{
    float * _weights = new float[patchSize * patchSize];
    int centralPixelRow = patchSize / 2;
    int centralPixelCol = centralPixelRow;
    float _dist;
    float _sumW = 0;

    for (int i = 0; i < patchSize; i++) {
        for (int j = 0; j < patchSize; j++) {
            _dist = (centralPixelRow - i) * (centralPixelRow - i) +
                    (centralPixelCol - j) * (centralPixelCol - j);
            _weights[i * patchSize + j] = exp(-_dist / (2 * (patchSigma * patchSigma)));
            _sumW += _weights[i * patchSize + j];
        }
    }

    for (int i = 0; i < patchSize; i++) {
        for (int j = 0; j < patchSize; j++) {
            _weights[i * patchSize + j] = _weights[i * patchSize + j] / _sumW;
        }
    }

    return _weights;
}

std::vector<float> computeResidual(std::vector<float> image, std::vector<float> filteredImage, int n)
{
    std::vector<float> res(n * n);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            res[i * n + j] = image[i * n + j] - filteredImage[i * n + j];
        }
    }
    std::cout << "Residual calculated" << std::endl << std::endl;

    return res;
}

/* -------------------------------------------------------------------------- */
/*                                device utils                                */
/* -------------------------------------------------------------------------- */

// patch-to-patch euclidean distance
__device__ float cudaComputePatchDistance(  float * image, 
                                            float * _weights,
                                            int n, 
                                            int patchSize, 
                                            int p1_rowStart, 
                                            int p1_colStart, 
                                            int p2_rowStart, 
                                            int p2_colStart  ) 
{
    float *patches = s;

    float ans = 0;
    float temp;

    for (int i = 0; i < patchSize; i++) {
        for (int j = 0; j < patchSize; j++) {
            if ( isInBounds(n, p1_rowStart + i, p1_colStart + j) && isInBounds(n, p2_rowStart + i, p2_colStart + j) ) {
                temp =  patches[i * n + p1_colStart + j] - 
                            image[(p2_rowStart + i) * n + p2_colStart + j];
                ans += _weights[i * patchSize + j] * temp * temp;
            }
        }
    }

    return ans;
}

} // namespace util

namespace prt {

void rowMajorArray(float * arr, int n, int m) 
{
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            std::cout << arr[i * m + j] << "\t";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

void rowMajorVector(std::vector<float> vector, int n, int m)
{
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            std::cout << vector[i * m + j] << "\t";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

void parameters(int patchSize, float filterSigma, float patchSigma)
{
    std::cout   << "Image filtered: "   << std::endl
                << "-Patch size "       << patchSize    << std::endl
                << "-Patch sigma "      << patchSigma   << std::endl
                << "-Filter Sigma "     << filterSigma  << std::endl  << std::endl;
}

} // namespace prt

namespace file {

std::vector<float> read(std::string filePath, int n, int m, char delim) 
{
    std::vector<float > image(n * m);
    std::ifstream myfile(filePath);
    std::ifstream input(filePath);
    std::string s;

    for (int i = 0; i < n; i++) {
        std::getline(input, s);
        std::istringstream iss(s);
        std::string num;
        int j = 0;
        while (std::getline(iss, num, delim)) {
            image[i * m + j++] = std::stof(num);
        }
    }

    return image;
}

std::string write(std::vector<float> image, std::string fileName, int rowNum, int colNum)
{
    std::vector<std::string> out;

    for (int i = 0; i < rowNum; i++) {
        for (int j = 0; j < colNum; j++) {
            out.push_back(std::to_string(image[i * colNum + j]) + " ");
        }
        out.push_back("\n");
    }

    std::ofstream output_file("./data/out/" + fileName + ".txt");
    std::ostream_iterator<std::string> output_iterator(output_file, "");
    std::copy(out.begin(), out.end(), output_iterator);

    return "./data/out/" + fileName + ".txt";
}

std::string write_images(   std::vector<float> filteredImage, 
                            std::vector<float > residual, 
                            int patchSize, 
                            float filterSigma, 
                            float patchSigma, 
                            int rowNum, 
                            int colNum, 
                            bool useGpu   )
{
    std::string ret;
    std::string params = std::to_string(patchSize)   + "_" + 
    std::to_string(filterSigma) + "_" + 
    std::to_string(patchSigma);

    std::string filteredName = "filtered_image_" + params;       
    if (useGpu) {
        filteredName = "cuda_" + filteredName;
    }
    ret = file::write(filteredImage, filteredName, rowNum, colNum);

    std::string resName = "residual_" + params;
    if (useGpu) {
        resName = "cuda_" + resName;
    }
    file::write(residual, resName, rowNum, colNum);

    std::cout << "Filtered image written" << std::endl << std::endl;
    std::cout << "Residual written" << std::endl << std::endl;

    return ret;
}

} // namespace file

namespace test {

void out(std::string standOutPath, std::string outPath, int n)
{
    std::vector<float> standOut = file::read(standOutPath, n, n, ',');
    std::vector<float> out = file::read(outPath, n, n, ',');

    for (int i = 0; i < n * n; i++) {
        if (standOut[i] != out[i]) {
            std::cout << "Error:\t" << standOut[i] << "\t" << out[i] << std::endl;
        }
    }

    if (standOut == out)
        std::cout << "Correct output - Test passed" << std::endl << std::endl;
    else
        std::cout << "Wrong output - Test failed" << std::endl << std::endl;
}

float computeMeanSquaredError(std::string originalOutPath, std::string outPath, int n)
{
    std::vector<float> originalOut = file::read(originalOutPath, n, n, ',');
    std::vector<float> out = file::read(outPath, n, n, ',');
    float res = 0;

    for (int i = 0; i < n * n; i ++) {
        res += pow(originalOut[i] - out[i], 2);
    }
    
    res = res / (n * n);

    return res;
}

} // namespace test

#endif // __UTILS_CUH__