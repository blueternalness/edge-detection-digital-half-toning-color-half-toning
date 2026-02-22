#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <cstdint>
#include <string>

using namespace std;

// Image dimensions
const int WIDTH = 1280;
const int HEIGHT = 852;
const int IMAGE_SIZE = WIDTH * HEIGHT;

// Helper: Read a raw 8-bit grayscale image
bool readRawImage(const string& filename, vector<uint8_t>& buffer) {
    ifstream file(filename, ios::binary);
    if (!file) {
        cerr << "Error: Cannot open " << filename << " for reading." << endl;
        return false;
    }
    file.read(reinterpret_cast<char*>(buffer.data()), IMAGE_SIZE);
    return true;
}

// Helper: Write a raw 8-bit grayscale image
bool writeRawImage(const string& filename, const vector<uint8_t>& buffer) {
    ofstream file(filename, ios::binary);
    if (!file) {
        cerr << "Error: Cannot open " << filename << " for writing." << endl;
        return false;
    }
    file.write(reinterpret_cast<const char*>(buffer.data()), IMAGE_SIZE);
    return true;
}

// 1. Fixed Thresholding
void fixedThresholding(const vector<uint8_t>& input, vector<uint8_t>& output, int T = 128) {
    for (int i = 0; i < IMAGE_SIZE; ++i) {
        output[i] = (input[i] < T) ? 0 : 255;
    }
}

// 2. Random Thresholding
void randomThresholding(const vector<uint8_t>& input, vector<uint8_t>& output) {
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> dist(0, 255);
    
    for (int i = 0; i < IMAGE_SIZE; ++i) {
        int rand_val = dist(gen);
        output[i] = (input[i] < rand_val) ? 0 : 255;
    }
}

// 3. Helper to recursively generate Bayer Index Matrices (I_n)
vector<vector<int>> generateBayerMatrix(int N) {
    // Base case for N=2
    if (N == 2) {
        return {{1, 2}, 
                {3, 0}};
    }
    
    // Recursive step
    int half = N / 2;
    vector<vector<int>> In = generateBayerMatrix(half);
    vector<vector<int>> I2n(N, vector<int>(N));
    
    for (int i = 0; i < half; ++i) {
        for (int j = 0; j < half; ++j) {
            int val = In[i][j];
            I2n[i][j]               = 4 * val + 1; // Top-Left
            I2n[i][j + half]        = 4 * val + 2; // Top-Right
            I2n[i + half][j]        = 4 * val + 3; // Bottom-Left
            I2n[i + half][j + half] = 4 * val;     // Bottom-Right
        }
    }
    return I2n;
}

// 3. Helper to generate Threshold Matrix T(x, y)
vector<vector<float>> generateThresholdMatrix(int N) {
    vector<vector<int>> In = generateBayerMatrix(N);
    vector<vector<float>> T(N, vector<float>(N));
    float N_squared = N * N;
    
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            T[i][j] = ((In[i][j] + 0.5f) / N_squared) * 255.0f;
        }
    }
    return T;
}

// 3. Dithering Matrix Application
void ditherMatrix(const vector<uint8_t>& input, vector<uint8_t>& output, int N) {
    vector<vector<float>> T = generateThresholdMatrix(N);
    
    for (int i = 0; i < HEIGHT; ++i) {
        for (int j = 0; j < WIDTH; ++j) {
            uint8_t F = input[i * WIDTH + j];
            float threshold_val = T[i % N][j % N];
            
            output[i * WIDTH + j] = (F <= threshold_val) ? 0 : 255;
        }
    }
}

int main() {
    vector<uint8_t> inputImage(IMAGE_SIZE);
    
    if (!readRawImage("Reflection.raw", inputImage)) {
        return -1;
    }

    vector<uint8_t> outputImage(IMAGE_SIZE);

    // 1. Fixed Thresholding
    fixedThresholding(inputImage, outputImage, 128);
    writeRawImage("1_fixed_threshold.raw", outputImage);

    // 2. Random Thresholding
    randomThresholding(inputImage, outputImage);
    writeRawImage("2_random_threshold.raw", outputImage);

    // 3. Dithering Matrices (I_2, I_8, I_32)
    ditherMatrix(inputImage, outputImage, 2);
    writeRawImage("3_dither_I2.raw", outputImage);
    
    ditherMatrix(inputImage, outputImage, 8);
    writeRawImage("3_dither_I8.raw", outputImage);
    
    ditherMatrix(inputImage, outputImage, 32);
    writeRawImage("3_dither_I32.raw", outputImage);

    cout << "Processing complete! All .raw files have been generated." << endl;

    return 0;
}