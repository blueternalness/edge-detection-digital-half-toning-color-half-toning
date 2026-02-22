#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <vector>

using namespace std;
using namespace cv;

// Function to apply Canny with specific thresholds and save the result
void applyAndSaveCanny(const Mat& grayImage, const string& baseName, double lowThresh, double highThresh) {
    Mat edges;
    // Apply Canny Edge Detector
    Canny(grayImage, edges, lowThresh, highThresh);

    // Create a filename indicating the thresholds used
    string filename = baseName + "_Canny_" + to_string((int)lowThresh) + "_" + to_string((int)highThresh) + ".jpg";
    
    // 8. Invert colors (Black edges on White background) to match the assignment
    bitwise_not(edges, edges);

    // Save the image
    imwrite(filename, edges);
    cout << "Saved: " << filename << endl;
}

int main() {
    // List of images to process
    vector<string> imageNames = {"Bird.jpg", "Deer.jpg"};

    // Threshold pairs to experiment with (Low, High)
    // OpenCV recommends a 1:2 or 1:3 ratio between low and high thresholds
    vector<pair<double, double>> thresholds = {
        {10.0, 30.0},    // Low thresholds (likely noisy)
        {60.0, 180.0},   // Medium thresholds (usually balanced)
        {120.0, 360.0}   // High thresholds (likely misses weaker edges)
    };

    for (const string& imgName : imageNames) {
        // Read image in color
        Mat colorImg = imread(imgName, IMREAD_COLOR);
        if (colorImg.empty()) {
            cerr << "Error: Could not load " << imgName << ". Ensure it is in the same directory." << endl;
            continue;
        }

        // Convert to grayscale
        Mat grayImg;
        cvtColor(colorImg, grayImg, COLOR_BGR2GRAY);

        // Optional but recommended: Apply Gaussian Blur to reduce noise before Canny
        Mat blurredImg;
        GaussianBlur(grayImg, blurredImg, Size(5, 5), 1.4);

        // Extract base name (e.g., "Bird" from "Bird.jpg")
        string baseName = imgName.substr(0, imgName.find_last_of("."));

        cout << "Processing " << imgName << "..." << endl;

        // Generate edge maps for different threshold combinations
        for (const auto& thresh : thresholds) {
            applyAndSaveCanny(blurredImg, baseName, thresh.first, thresh.second);
        }
        cout << "---" << endl;
    }

    cout << "All Canny edge maps generated successfully." << endl;
    return 0;
}