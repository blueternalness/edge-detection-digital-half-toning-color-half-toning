#include <opencv2/opencv.hpp>
#include <opencv2/ximgproc.hpp>
#include <iostream>
#include <string>
#include <vector>

using namespace std;
using namespace cv;
using namespace cv::ximgproc;

void processSE(Ptr<StructuredEdgeDetection> detector, const string& imgName, float thresholdValue) {
    // 1. Read the RGB image
    Mat image = imread(imgName, IMREAD_COLOR);
    if (image.empty()) {
        cerr << "Error: Could not load " << imgName << endl;
        return;
    }

    // 2. Convert to 32-bit float and normalize to [0, 1] as required by the SE detector
    Mat imageFloat;
    image.convertTo(imageFloat, CV_32FC3, 1.0 / 255.0);

    // 3. Generate probability edge map
    Mat probEdgeMap;
    detector->detectEdges(imageFloat, probEdgeMap);

    // 4. Binarize the probability edge map
    Mat binaryEdgeMap;
    // probEdgeMap contains float values in [0, 1]
    threshold(probEdgeMap, binaryEdgeMap, thresholdValue, 1.0, THRESH_BINARY);

    // 5. Convert maps back to 8-bit unsigned chars (0-255) for saving
    Mat probEdgeMap8U, binaryEdgeMap8U;
    probEdgeMap.convertTo(probEdgeMap8U, CV_8UC1, 255.0);
    binaryEdgeMap.convertTo(binaryEdgeMap8U, CV_8UC1, 255.0);

    // 6. Save the results
    string baseName = imgName.substr(0, imgName.find_last_of("."));
    imwrite(baseName + "_SE_prob.png", probEdgeMap8U);
    imwrite(baseName + "_SE_binary.png", binaryEdgeMap8U);

    cout << "Processed " << imgName << ":" << endl;
    cout << "  - Saved " << baseName << "_SE_prob.png" << endl;
    cout << "  - Saved " << baseName << "_SE_binary.png" << endl;
}

int main() {
    // Path to the pre-trained model file (downloaded from OpenCV extra data)
    string modelFilename = "model.yml.gz"; 
    
    Ptr<StructuredEdgeDetection> pDollar;
    try {
        pDollar = createStructuredEdgeDetection(modelFilename);
    } catch (const Exception& e) {
        cerr << "Error loading model: Ensure 'model.yml.gz' is in the directory." << endl;
        return -1;
    }

    // The assignment example uses p > 0.8 for the binary edge map
    float binarizationThreshold = 0.8f;

    vector<string> images = {"Bird.jpg", "Deer.jpg"};

    for (const string& imgName : images) {
        processSE(pDollar, imgName, binarizationThreshold);
    }

    cout << "Structured Edge detection complete." << endl;
    return 0;
}