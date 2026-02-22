#include <iostream>
#include <fstream>
#include <vector>

using namespace std;

int main() {
    const int width = 1280;
    const int height = 853;
    const int channels = 3;
    const int imageSize = width * height * channels;
    const char* inputFilename = "Flowers.raw";
    const char* outputFilename = "Flowers_halftone.raw";

    vector<unsigned char> rgbImage(imageSize);
    ifstream inputFile(inputFilename, ios::binary);
    inputFile.read(reinterpret_cast<char*>(rgbImage.data()), imageSize);
    inputFile.close();
    vector<float> cmyImage(imageSize);
    for (int i = 0; i < imageSize; ++i) {
        cmyImage[i] = 255.0f - static_cast<float>(rgbImage[i]);
    }

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            for (int c = 0; c < channels; ++c) {
                int index = (y * width + x) * channels + c;
                float oldVal = cmyImage[index];
                float newVal = (oldVal >= 128.0f) ? 255.0f : 0.0f;
                cmyImage[index] = newVal;
                
                float error = oldVal - newVal;
                if (x + 1 < width) {
                    cmyImage[(y * width + (x + 1)) * channels + c] += error * (7.0f / 16.0f);
                }
                if (x - 1 >= 0 && y + 1 < height) {
                    cmyImage[((y + 1) * width + (x - 1)) * channels + c] += error * (3.0f / 16.0f);
                }
                if (y + 1 < height) {
                    cmyImage[((y + 1) * width + x) * channels + c] += error * (5.0f / 16.0f);
                }
                if (x + 1 < width && y + 1 < height) {
                    cmyImage[((y + 1) * width + (x + 1)) * channels + c] += error * (1.0f / 16.0f);
                }
            }
        }
    }

    vector<unsigned char> outputRgbImage(imageSize);
    for (int i = 0; i < imageSize; ++i) {
        float rgbVal = 255.0f - cmyImage[i];
        if (rgbVal > 255.0f) rgbVal = 255.0f;
        if (rgbVal < 0.0f) rgbVal = 0.0f;
        
        outputRgbImage[i] = static_cast<unsigned char>(rgbVal);
    }

    ofstream outputFile(outputFilename, ios::binary);
    outputFile.write(reinterpret_cast<const char*>(outputRgbImage.data()), imageSize);
    outputFile.close();
    return 0;
}