#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <string>

using namespace std;

const int WIDTH = 481;
const int HEIGHT = 321;
const int BYTES_PER_PIXEL = 3;

// Function to read a raw RGB image
vector<unsigned char> readRawImage(const string& filename, int width, int height, int bpp) {
    vector<unsigned char> imageData(width * height * bpp);
    ifstream file(filename, ios::binary);
    if (!file) {
        cerr << "Error: Cannot open " << filename << endl;
        exit(1);
    }
    file.read(reinterpret_cast<char*>(imageData.data()), imageData.size());
    file.close();
    return imageData;
}

// Function to write a raw Grayscale/1-byte image
void writeRawImage(const string& filename, const vector<unsigned char>& imageData) {
    ofstream file(filename, ios::binary);
    if (!file) {
        cerr << "Error: Cannot write to " << filename << endl;
        exit(1);
    }
    file.write(reinterpret_cast<const char*>(imageData.data()), imageData.size());
    file.close();
}

// Convert RGB to Grayscale using the provided formula
vector<double> convertToGrayscale(const vector<unsigned char>& rgbImage) {
    vector<double> grayImage(WIDTH * HEIGHT);
    for (int i = 0; i < WIDTH * HEIGHT; ++i) {
        double r = rgbImage[i * 3];
        double g = rgbImage[i * 3 + 1];
        double b = rgbImage[i * 3 + 2];
        grayImage[i] = 0.2989 * r + 0.5870 * g + 0.1140 * b;
    }
    return grayImage;
}

// Normalize a double vector to 0-255 unsigned char vector
vector<unsigned char> normalizeTo255(const vector<double>& input) {
    double minVal = input[0];
    double maxVal = input[0];
    
    for (double val : input) {
        if (val < minVal) minVal = val;
        if (val > maxVal) maxVal = val;
    }

    vector<unsigned char> output(input.size());
    for (size_t i = 0; i < input.size(); ++i) {
        output[i] = static_cast<unsigned char>(((input[i] - minVal) / (maxVal - minVal)) * 255.0);
    }
    return output;
}

// Apply Sobel filter and generate gradient maps and magnitude
void applySobel(const vector<double>& grayImage, string baseFilename, double thresholdPercentage) {
    vector<double> gradX(WIDTH * HEIGHT, 0.0);
    vector<double> gradY(WIDTH * HEIGHT, 0.0);
    vector<double> magnitude(WIDTH * HEIGHT, 0.0);

    int Gx[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
    int Gy[3][3] = {{1, 2, 1}, {0, 0, 0}, {-1, -2, -1}};

    // Convolution (ignoring 1-pixel borders for simplicity)
    for (int y = 1; y < HEIGHT - 1; ++y) {
        for (int x = 1; x < WIDTH - 1; ++x) {
            double sumX = 0.0;
            double sumY = 0.0;

            for (int j = -1; j <= 1; ++j) {
                for (int i = -1; i <= 1; ++i) {
                    double pixelVal = grayImage[(y + j) * WIDTH + (x + i)];
                    sumX += pixelVal * Gx[j + 1][i + 1];
                    sumY += pixelVal * Gy[j + 1][i + 1];
                }
            }

            int index = y * WIDTH + x;
            gradX[index] = sumX;
            gradY[index] = sumY;
            magnitude[index] = sqrt(sumX * sumX + sumY * sumY);
        }
    }

    // Task (1): Normalize and save X and Y gradients
    writeRawImage(baseFilename + "_GradX.raw", normalizeTo255(gradX));
    writeRawImage(baseFilename + "_GradY.raw", normalizeTo255(gradY));

    // Task (2): Normalize and save magnitude map
    writeRawImage(baseFilename + "_Magnitude.raw", normalizeTo255(magnitude));

    // Task (3): Thresholding based on percentage
    vector<double> sortedMag = magnitude;
    sort(sortedMag.begin(), sortedMag.end());
    
    // Find the cutoff value that separates the top 'thresholdPercentage' of pixels
    int thresholdIndex = static_cast<int>((1.0 - (thresholdPercentage / 100.0)) * sortedMag.size());
    if (thresholdIndex >= sortedMag.size()) thresholdIndex = sortedMag.size() - 1;
    if (thresholdIndex < 0) thresholdIndex = 0;
    
    double thresholdValue = sortedMag[thresholdIndex];
    vector<unsigned char> edgeMap(WIDTH * HEIGHT);

    for (int i = 0; i < WIDTH * HEIGHT; ++i) {
        if (magnitude[i] >= thresholdValue) {
            edgeMap[i] = 0;   // Edge pixel
        } else {
            edgeMap[i] = 255; // Background pixel
        }
    }

    writeRawImage(baseFilename + "_EdgeMap.raw", edgeMap);
    cout << "Processed " << baseFilename << " | Threshold cutoff used: " << thresholdValue << " (" << thresholdPercentage << "%)" << endl;
}

int main() {
    // Determine the best percentage visually. 
    // Usually, 10% to 15% yields the strongest edge boundaries without excessive noise.
    double thresholdPercent = 12.5; 

    // Process Bird.raw
    cout << "Reading Bird.raw..." << endl;
    vector<unsigned char> birdRGB = readRawImage("Bird.raw", WIDTH, HEIGHT, BYTES_PER_PIXEL);
    vector<double> birdGray = convertToGrayscale(birdRGB);
    applySobel(birdGray, "Bird", thresholdPercent);

    // Process Deer.raw
    cout << "Reading Deer.raw..." << endl;
    vector<unsigned char> deerRGB = readRawImage("Deer.raw", WIDTH, HEIGHT, BYTES_PER_PIXEL);
    vector<double> deerGray = convertToGrayscale(deerRGB);
    applySobel(deerGray, "Deer", thresholdPercent);

    cout << "All images successfully generated." << endl;
    return 0;
}
