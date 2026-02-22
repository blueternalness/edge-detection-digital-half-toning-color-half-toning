#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <vector>

using namespace std;
using namespace cv;

void applyAndSaveCanny(const Mat& grayImage, const string& baseName, double lowThresh, double highThresh) {
    Mat edges;
    Canny(grayImage, edges, lowThresh, highThresh);
    string filename = baseName + "_Canny_" + to_string((int)lowThresh) + "_" + to_string((int)highThresh) + ".jpg";
    bitwise_not(edges, edges);
    imwrite(filename, edges);
}

int main() {
    vector<string> imageNames = {"Bird.jpg", "Deer.jpg"};

    vector<pair<double, double>> thresholds = {
        {10.0, 30.0},
        {60.0, 180.0},
        {120.0, 360.0}
    };

    for (const string& imgName : imageNames) {
        Mat colorImg = imread(imgName, IMREAD_COLOR);
        Mat grayImg;
        cvtColor(colorImg, grayImg, COLOR_BGR2GRAY);
        Mat blurredImg;
        GaussianBlur(grayImg, blurredImg, Size(5, 5), 1.4);

        string baseName = imgName.substr(0, imgName.find_last_of("."));

        for (const auto& thresh : thresholds) {
            applyAndSaveCanny(blurredImg, baseName, thresh.first, thresh.second);
        }
    }

    return 0;
}