#include <opencv2/opencv.hpp>
#include <opencv2/ximgproc.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>

using namespace std;
using namespace cv;
using namespace cv::ximgproc;
const int WIDTH = 481;
const int HEIGHT = 321;
const int CHANNELS = 3;

Mat readRawRGB(const string& filename, int width, int height, int channels) {
    ifstream file(filename, ios::binary);
    vector<unsigned char> buffer(width * height * channels);
    file.read(reinterpret_cast<char*>(buffer.data()), buffer.size());
    file.close();

    Mat img(height, width, CV_8UC3, buffer.data());
    return img.clone(); 
}

void processSE(Ptr<StructuredEdgeDetection> detector, const string& imgName, float thresholdValue) {
    Mat imageRGB = readRawRGB(imgName, WIDTH, HEIGHT, CHANNELS);
    Mat imageFloat;
    imageRGB.convertTo(imageFloat, CV_32FC3, 1.0 / 255.0);

    Mat probEdgeMap;
    detector->detectEdges(imageFloat, probEdgeMap);
    normalize(probEdgeMap, probEdgeMap, 0.0, 1.0, NORM_MINMAX);

    Mat orientationMap, nmsEdgeMap;
    detector->computeOrientation(probEdgeMap, orientationMap);
    detector->edgesNms(probEdgeMap, orientationMap, nmsEdgeMap, 2, 0, 1, true);

    Mat binaryEdgeMap;
    threshold(nmsEdgeMap, binaryEdgeMap, thresholdValue, 1.0, THRESH_BINARY);

    Mat probEdgeMap8U, binaryEdgeMap8U;
    probEdgeMap.convertTo(probEdgeMap8U, CV_8UC1, 255.0);
    binaryEdgeMap.convertTo(binaryEdgeMap8U, CV_8UC1, 255.0);

    bitwise_not(probEdgeMap8U, probEdgeMap8U);
    bitwise_not(binaryEdgeMap8U, binaryEdgeMap8U);

    string baseName = imgName.substr(0, imgName.find_last_of("."));
    imwrite(baseName + "_SE_prob.png", probEdgeMap8U);
    imwrite(baseName + "_SE_binary_" + to_string(thresholdValue).substr(0, 4) + ".png", binaryEdgeMap8U);
}

int main() {
    string modelFilename = "model.yml.gz"; 
    
    Ptr<StructuredEdgeDetection> pDollar;
    pDollar = createStructuredEdgeDetection(modelFilename);
    vector<float> thresholds = {0.05f, 0.1f, 0.15f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f}; 
    vector<string> images = {"Bird.raw", "Deer.raw"};

    for (const float& thresh : thresholds) {
        for (const string& imgName : images) {
            processSE(pDollar, imgName, thresh);
        }
    }

    return 0;
}