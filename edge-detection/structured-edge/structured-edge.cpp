#include <opencv2/opencv.hpp>
#include <opencv2/ximgproc.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>

using namespace std;
using namespace cv;
using namespace cv::ximgproc;

// Define dimensions based on the assignment images
const int WIDTH = 481;
const int HEIGHT = 321;
const int CHANNELS = 3;

// Function to read a raw RGB image directly into an OpenCV Mat
Mat readRawRGB(const string& filename, int width, int height, int channels) {
    ifstream file(filename, ios::binary);
    if (!file) {
        cerr << "Error: Cannot open " << filename << endl;
        exit(1);
    }
    
    // Read the raw binary data into a vector
    vector<unsigned char> buffer(width * height * channels);
    file.read(reinterpret_cast<char*>(buffer.data()), buffer.size());
    file.close();

    // The data is natively RGB in the .raw file, so we map it directly to a Mat.
    // This avoids the BGR issue caused by imread().
    Mat img(height, width, CV_8UC3, buffer.data());
    return img.clone(); 
}

void processSE(Ptr<StructuredEdgeDetection> detector, const string& imgName, float thresholdValue) {
    // 1. Read the raw RGB image
    Mat imageRGB = readRawRGB(imgName, WIDTH, HEIGHT, CHANNELS);

    // 2. Convert to 32-bit float and scale to [0, 1] for the SE detector
    Mat imageFloat;
    imageRGB.convertTo(imageFloat, CV_32FC3, 1.0 / 255.0);

    // 3. Generate raw probability edge map
    Mat probEdgeMap;
    detector->detectEdges(imageFloat, probEdgeMap);

    // 4. NORMALIZATION: Stretch probabilities to a full [0.0, 1.0] scale
    normalize(probEdgeMap, probEdgeMap, 0.0, 1.0, NORM_MINMAX);

    // 5. NON-MAXIMUM SUPPRESSION: Thin the edges to 1-pixel width
    Mat orientationMap, nmsEdgeMap;
    detector->computeOrientation(probEdgeMap, orientationMap);
    detector->edgesNms(probEdgeMap, orientationMap, nmsEdgeMap, 2, 0, 1, true);

    // 6. Binarize the thinned NMS edge map
    Mat binaryEdgeMap;
    threshold(nmsEdgeMap, binaryEdgeMap, thresholdValue, 1.0, THRESH_BINARY);

    // 7. Convert to 8-bit (0-255) for saving as standard image files
    Mat probEdgeMap8U, binaryEdgeMap8U;
    probEdgeMap.convertTo(probEdgeMap8U, CV_8UC1, 255.0);
    binaryEdgeMap.convertTo(binaryEdgeMap8U, CV_8UC1, 255.0);

    // 8. Invert colors (Black edges on White background) to match the assignment
    bitwise_not(probEdgeMap8U, probEdgeMap8U);
    bitwise_not(binaryEdgeMap8U, binaryEdgeMap8U);

    // 9. Save the results as PNGs
    string baseName = imgName.substr(0, imgName.find_last_of("."));
    imwrite(baseName + "_SE_prob.png", probEdgeMap8U);
    imwrite(baseName + "_SE_binary_" + to_string(thresholdValue).substr(0, 4) + ".png", binaryEdgeMap8U);

    cout << "Processed " << imgName << " | Normalized & NMS Applied | Threshold: " << thresholdValue << endl;
}

int main() {
    string modelFilename = "model.yml.gz"; 
    
    Ptr<StructuredEdgeDetection> pDollar;
    try {
        pDollar = createStructuredEdgeDetection(modelFilename);
    } catch (const Exception& e) {
        cerr << "Error loading model: Ensure 'model.yml.gz' is in the directory." << endl;
        return -1;
    }

    // Since we are applying Normalization AND Non-Maximum Suppression (which lowers overall intensities),
    // testing a range of thresholds will help you find the absolute best visual map.
    vector<float> thresholds = {0.05f, 0.1f, 0.15f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f}; 
    vector<string> images = {"Bird.raw", "Deer.raw"};

    for (const float& thresh : thresholds) {
        for (const string& imgName : images) {
            processSE(pDollar, imgName, thresh);
        }
    }

    cout << "Structured Edge detection complete." << endl;
    return 0;
}