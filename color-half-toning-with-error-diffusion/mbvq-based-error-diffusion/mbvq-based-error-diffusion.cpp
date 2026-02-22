#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <string>
#include <algorithm>

using namespace std;

struct ColorFloat {
    float r, g, b;
};
const ColorFloat W = {255, 255, 255};
const ColorFloat K = {0, 0, 0};
const ColorFloat C = {0, 255, 255};
const ColorFloat M = {255, 0, 255};
const ColorFloat Y = {255, 255, 0};
const ColorFloat R = {255, 0, 0};
const ColorFloat G = {0, 255, 0};
const ColorFloat B = {0, 0, 255};

float colorDistSq(const ColorFloat& c1, const ColorFloat& c2) {
    return pow(c1.r - c2.r, 2) + pow(c1.g - c2.g, 2) + pow(c1.b - c2.b, 2);
}

ColorFloat getClosestVertex(const ColorFloat& pixel, const ColorFloat& v1, const ColorFloat& v2, const ColorFloat& v3, const ColorFloat& v4) {
    ColorFloat vertices[4] = {v1, v2, v3, v4};
    ColorFloat closest = v1;
    float minDist = colorDistSq(pixel, v1);
    
    for (int i = 1; i < 4; ++i) {
        float dist = colorDistSq(pixel, vertices[i]);
        if (dist < minDist) {
            minDist = dist;
            closest = vertices[i];
        }
    }
    return closest;
}

ColorFloat getMBVQVertex(const ColorFloat& origPixel, const ColorFloat& errPixel) {
    float r = origPixel.r;
    float g = origPixel.g;
    float b = origPixel.b;
    if ((r + g) > 255) {
        if ((g + b) > 255) {
            if ((r + g + b) > 510) {
                return getClosestVertex(errPixel, C, M, Y, W);
            } else {
                return getClosestVertex(errPixel, M, Y, G, C);
            }
        } else {
            return getClosestVertex(errPixel, R, G, M, Y);
        }
    } else {
        if (!((g + b) > 255)) {
            if ((r + g + b) <= 255) {
                return getClosestVertex(errPixel, K, R, G, B);
            } else {
                return getClosestVertex(errPixel, R, G, B, M);
            }
        } else {
            return getClosestVertex(errPixel, C, M, G, B);
        }
    }
}

int main() {
    const int width = 1280;
    const int height = 853;
    const int channels = 3;
    const int imageSize = width * height * channels;

    string inputFilename = "Flowers.raw";
    string outputFilename = "Flowers_MBVQ.raw";

    vector<unsigned char> rgbImage(imageSize);
    ifstream inputFile(inputFilename, ios::binary);
    inputFile.read(reinterpret_cast<char*>(rgbImage.data()), imageSize);
    inputFile.close();
    vector<ColorFloat> origImage(width * height);
    vector<ColorFloat> errImage(width * height);
    
    for (int i = 0; i < width * height; ++i) {
        float r = static_cast<float>(rgbImage[i * 3]);
        float g = static_cast<float>(rgbImage[i * 3 + 1]);
        float b = static_cast<float>(rgbImage[i * 3 + 2]);
        origImage[i] = {r, g, b};
        errImage[i] = {r, g, b};
    }

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int idx = y * width + x;
            ColorFloat origPixel = origImage[idx];
            ColorFloat currentPixel = errImage[idx];
            ColorFloat newPixel = getMBVQVertex(origPixel, currentPixel);
            
            errImage[idx] = newPixel;

            ColorFloat error = {
                currentPixel.r - newPixel.r,
                currentPixel.g - newPixel.g,
                currentPixel.b - newPixel.b
            };
            auto diffuseError = [&](int dx, int dy, float weight) {
                int nx = x + dx;
                int ny = y + dy;
                if (nx >= 0 && nx < width && ny < height) {
                    int nIdx = ny * width + nx;
                    errImage[nIdx].r += error.r * weight;
                    errImage[nIdx].g += error.g * weight;
                    errImage[nIdx].b += error.b * weight;
                }
            };

            diffuseError(1, 0, 7.0f / 16.0f);
            diffuseError(-1, 1, 3.0f / 16.0f);
            diffuseError(0, 1, 5.0f / 16.0f);
            diffuseError(1, 1, 1.0f / 16.0f);
        }
    }
    vector<unsigned char> outputImage(imageSize);
    for (int i = 0; i < width * height; ++i) {
        outputImage[i * 3] = static_cast<unsigned char>(clamp(errImage[i].r, 0.0f, 255.0f));
        outputImage[i * 3 + 1] = static_cast<unsigned char>(clamp(errImage[i].g, 0.0f, 255.0f));
        outputImage[i * 3 + 2] = static_cast<unsigned char>(clamp(errImage[i].b, 0.0f, 255.0f));
    }

    ofstream outputFile(outputFilename, ios::binary);
    outputFile.write(reinterpret_cast<const char*>(outputImage.data()), imageSize);
    outputFile.close();
    return 0;
}