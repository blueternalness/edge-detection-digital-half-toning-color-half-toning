#include <iostream>
#include <fstream>
#include <vector>
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

// Generalized Error Diffusion Function
void applyErrorDiffusion(const vector<uint8_t>& input, vector<uint8_t>& output, 
                         const vector<vector<float>>& kernel, int cx, int cy, float divisor, bool serpentine) {
    
    // Use a float buffer to accurately track accumulated error without overflow/clamping
    vector<float> buffer(IMAGE_SIZE);
    for (int i = 0; i < IMAGE_SIZE; ++i) {
        buffer[i] = static_cast<float>(input[i]);
    }
    
    int kernel_h = kernel.size();
    int kernel_w = kernel[0].size();

    for (int y = 0; y < HEIGHT; ++y) {
        // Serpentine logic: reverse direction on odd rows if enabled
        bool rtl = serpentine && (y % 2 != 0); 
        
        int start_x = rtl ? WIDTH - 1 : 0;
        int end_x   = rtl ? -1 : WIDTH;
        int step_x  = rtl ? -1 : 1;

        for (int x = start_x; x != end_x; x += step_x) {
            float old_pixel = buffer[y * WIDTH + x];
            // Threshold at 128
            uint8_t new_pixel = (old_pixel < 128.0f) ? 0 : 255;
            output[y * WIDTH + x] = new_pixel;
            
            float error = old_pixel - new_pixel;

            // Diffuse the error to neighboring pixels
            for (int ky = 0; ky < kernel_h; ++ky) {
                for (int kx = 0; kx < kernel_w; ++kx) {
                    float weight = kernel[ky][kx] / divisor;
                    if (weight == 0.0f) continue;

                    int oy = ky - cy;
                    int ox = kx - cx;
                    
                    // Mirror the x-offset horizontally if scanning right-to-left
                    if (rtl) ox = -ox; 

                    int ny = y + oy;
                    int nx = x + ox;

                    // Apply error if within image boundaries
                    if (ny >= 0 && ny < HEIGHT && nx >= 0 && nx < WIDTH) {
                        buffer[ny * WIDTH + nx] += error * weight;
                    }
                }
            }
        }
    }
}

int main() {
    vector<uint8_t> inputImage(IMAGE_SIZE);
    
    if (!readRawImage("Reflection.raw", inputImage)) {
        return -1;
    }

    vector<uint8_t> outputImage(IMAGE_SIZE);

    // 1. Floyd-Steinberg (with serpentine scanning)
    // Kernel center is at row 1, col 1
    vector<vector<float>> fs_kernel = {
        {0, 0, 0},
        {0, 0, 7},
        {3, 5, 1}
    };
    applyErrorDiffusion(inputImage, outputImage, fs_kernel, 1, 1, 16.0f, true);
    writeRawImage("4_error_diffusion_FS_serpentine.raw", outputImage);

    // 2. Jarvis, Judice, and Ninke (JJN) (standard raster scanning)
    // Kernel center is at row 2, col 2
    vector<vector<float>> jjn_kernel = {
        {0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0},
        {0, 0, 0, 7, 5},
        {3, 5, 7, 5, 3},
        {1, 3, 5, 3, 1}
    };
    applyErrorDiffusion(inputImage, outputImage, jjn_kernel, 2, 2, 48.0f, false);
    writeRawImage("5_error_diffusion_JJN.raw", outputImage);

    // 3. Stucki (standard raster scanning)
    // Kernel center is at row 2, col 2
    vector<vector<float>> stucki_kernel = {
        {0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0},
        {0, 0, 0, 8, 4},
        {2, 4, 8, 4, 2},
        {1, 2, 4, 2, 1}
    };
    applyErrorDiffusion(inputImage, outputImage, stucki_kernel, 2, 2, 42.0f, false);
    writeRawImage("6_error_diffusion_Stucki.raw", outputImage);

    cout << "Error diffusion complete! Outputs saved to current directory." << endl;

    return 0;
}