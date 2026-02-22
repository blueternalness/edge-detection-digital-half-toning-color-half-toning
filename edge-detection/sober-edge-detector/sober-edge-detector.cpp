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

vector<unsigned char> readRawImage(const string& filename, int width, int height, int bpp) {
    vector<unsigned char> imageData(width * height * bpp);
    ifstream file(filename, ios::binary);
    file.read(reinterpret_cast<char*>(imageData.data()), imageData.size());
    file.close();
    return imageData;
}
void writeRawImage(const string& filename, const vector<unsigned char>& imageData) {
    ofstream file(filename, ios::binary);
    file.write(reinterpret_cast<const char*>(imageData.data()), imageData.size());
    file.close();
}

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

void applySobel(const vector<double>& grayImage, string baseFilename, double thresholdPercentage) {
    vector<double> gradX(WIDTH * HEIGHT, 0.0);
    vector<double> gradY(WIDTH * HEIGHT, 0.0);
    vector<double> magnitude(WIDTH * HEIGHT, 0.0);

    int Gx[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
    int Gy[3][3] = {{1, 2, 1}, {0, 0, 0}, {-1, -2, -1}};
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
    writeRawImage(baseFilename + "_GradX.raw", normalizeTo255(gradX));
    writeRawImage(baseFilename + "_GradY.raw", normalizeTo255(gradY));
    writeRawImage(baseFilename + "_Magnitude.raw", normalizeTo255(magnitude));

    vector<double> sortedMag = magnitude;
    sort(sortedMag.begin(), sortedMag.end());
    
    int thresholdIndex = static_cast<int>((1.0 - (thresholdPercentage / 100.0)) * sortedMag.size());
    if (thresholdIndex >= sortedMag.size()) thresholdIndex = sortedMag.size() - 1;
    if (thresholdIndex < 0) thresholdIndex = 0;
    
    double thresholdValue = sortedMag[thresholdIndex];
    vector<unsigned char> edgeMap(WIDTH * HEIGHT);

    for (int i = 0; i < WIDTH * HEIGHT; ++i) {
        if (magnitude[i] >= thresholdValue) {
            edgeMap[i] = 0;
        } else {
            edgeMap[i] = 255;
        }
    }

    writeRawImage(baseFilename + "_EdgeMap.raw", edgeMap);
}

int main() {
    double thresholdPercent = 15; 
    vector<unsigned char> birdRGB = readRawImage("Bird.raw", WIDTH, HEIGHT, BYTES_PER_PIXEL);
    vector<double> birdGray = convertToGrayscale(birdRGB);
    applySobel(birdGray, "Bird", thresholdPercent);

    vector<unsigned char> deerRGB = readRawImage("Deer.raw", WIDTH, HEIGHT, BYTES_PER_PIXEL);
    vector<double> deerGray = convertToGrayscale(deerRGB);
    applySobel(deerGray, "Deer", thresholdPercent);

    return 0;
}
