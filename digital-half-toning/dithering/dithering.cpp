#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <cstdint>
#include <string>

using namespace std;

const int WIDTH = 1280;
const int HEIGHT = 852;
const int IMAGE_SIZE = WIDTH * HEIGHT;

bool readRawImage(const string& filename, vector<uint8_t>& buffer) {
    ifstream file(filename, ios::binary);
    file.read(reinterpret_cast<char*>(buffer.data()), IMAGE_SIZE);
    return true;
}

bool writeRawImage(const string& filename, const vector<uint8_t>& buffer) {
    ofstream file(filename, ios::binary);
    file.write(reinterpret_cast<const char*>(buffer.data()), IMAGE_SIZE);
    return true;
}

void fixedThresholding(const vector<uint8_t>& input, vector<uint8_t>& output, int T = 128) {
    for (int i = 0; i < IMAGE_SIZE; ++i) {
        output[i] = (input[i] < T) ? 0 : 255;
    }
}

void randomThresholding(const vector<uint8_t>& input, vector<uint8_t>& output) {
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> dist(0, 255);
    
    for (int i = 0; i < IMAGE_SIZE; ++i) {
        int rand_val = dist(gen);
        output[i] = (input[i] < rand_val) ? 0 : 255;
    }
}

vector<vector<int>> generateBayerMatrix(int N) {
    if (N == 2) {
        return {{1, 2}, {3, 0}};
    }
    int half = N / 2;
    vector<vector<int>> In = generateBayerMatrix(half);
    vector<vector<int>> I2n(N, vector<int>(N));
    
    for (int i = 0; i < half; ++i) {
        for (int j = 0; j < half; ++j) {
            int val = In[i][j];
            I2n[i][j] = 4 * val + 1;
            I2n[i][j + half] = 4 * val + 2;
            I2n[i + half][j]= 4 * val + 3;
            I2n[i + half][j + half] = 4 * val;
        }
    }
    return I2n;
}

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

    fixedThresholding(inputImage, outputImage, 128);
    writeRawImage("1_fixed_threshold.raw", outputImage);

    randomThresholding(inputImage, outputImage);
    writeRawImage("2_random_threshold.raw", outputImage);

    ditherMatrix(inputImage, outputImage, 2);
    writeRawImage("3_dither_I2.raw", outputImage);
    
    ditherMatrix(inputImage, outputImage, 8);
    writeRawImage("3_dither_I8.raw", outputImage);
    
    ditherMatrix(inputImage, outputImage, 32);
    writeRawImage("3_dither_I32.raw", outputImage);
    return 0;
}