#include <numeric>
#include "matching2D.hpp"

using namespace std;

// Find best matches for keypoints in two camera images based on several matching methods
void matchDescriptors(std::vector<cv::KeyPoint> &kPtsSource, std::vector<cv::KeyPoint> &kPtsRef, cv::Mat &descSource,
                      cv::Mat &descRef,
                      std::vector<cv::DMatch> &matches, const std::string& descriptorType, const std::string& matcherType,
                      const std::string& selectorType) {
    // configure matcher
    bool crossCheck = false;
    cv::Ptr<cv::DescriptorMatcher> matcher;

    if (matcherType == "MAT_BF") {
        int normType;
        if (descriptorType == "DES_BINARY")
            normType = cv::NORM_HAMMING;
        else
            normType = cv::NORM_L2;
        matcher = cv::BFMatcher::create(normType, crossCheck);
    } else if (matcherType == "MAT_FLANN") {
        if (descriptorType == "DES_BINARY")
            matcher = cv::makePtr<cv::FlannBasedMatcher>(cv::FlannBasedMatcher(cv::makePtr<cv::flann::LshIndexParams>(12, 20, 2)));
        else
            matcher = cv::FlannBasedMatcher::create();
        // std::cout << "FLANN matching" << std::endl;;
    }

    // perform matching task
    if (selectorType == "SEL_NN") { // nearest neighbor (best match)
        matcher->match(descSource, descRef, matches); // Finds the best match for each descriptor in desc1
    } else if (selectorType == "SEL_KNN") { // k nearest neighbors (k=2)
        // implement k-nearest-neighbor matching
        std::vector<std::vector<cv::DMatch>> knn_matches;
        matcher->knnMatch(descSource, descRef, knn_matches, 2);

        // std::cout << knn_matches[0][1].distance << std::endl;
        // filter matches using descriptor distance ratio test
        for (auto &knn_match : knn_matches){
            if ((!knn_match.empty() && (knn_match[0].distance) < 0.8f * knn_match[1].distance))
                matches.push_back(knn_match[0]);
        }
        // cout << "Removed " << knn_matches.size() - matches.size() << " matches." << endl;
    }
}

// Use one of several types of state-of-art descriptors to uniquely identify keypoints
void
descKeypoints(vector<cv::KeyPoint> &keypoints, cv::Mat &img, cv::Mat &descriptors, const std::string &descriptorType) {
    // select appropriate descriptor
    cv::Ptr<cv::DescriptorExtractor> extractor;
    if (descriptorType == "BRISK") {

        int threshold = 30;        // FAST/AGAST detection threshold score.
        int octaves = 3;           // detection octaves (use 0 to do single scale)
        float patternScale = 1.0f; // apply this scale to the pattern used for sampling the neighbourhood of a keypoint.

        extractor = cv::BRISK::create(threshold, octaves, patternScale);
    } else if (descriptorType == "SIFT") {
        extractor = cv::xfeatures2d::SIFT::create();
    } else if (descriptorType == "BRIEF") {
        extractor = cv::xfeatures2d::BriefDescriptorExtractor::create();
    } else if (descriptorType == "ORB") {
        extractor = cv::ORB::create();
    } else if (descriptorType == "FREAK") {
        extractor = cv::xfeatures2d::FREAK::create();
    } else if (descriptorType == "AKAZE") {
        extractor = cv::AKAZE::create();
    }
    // perform feature description
    auto t = (double) cv::getTickCount();
    extractor->compute(img, keypoints, descriptors);
    t = ((double) cv::getTickCount() - t) / cv::getTickFrequency();
    // cout << descriptorType << " descriptor extraction in " << 1000 * t / 1.0 << " ms" << endl;
}

// Detect keypoints in image using the traditional Shi-Thomasi detector
void detKeypointsShiTomasi(vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis) {
    // compute detector parameters based on image size
    int blockSize = 4;       //  size of an average block for computing a derivative covariation matrix over each pixel neighborhood
    double maxOverlap = 0.0; // max. permissible overlap between two features in %
    double minDistance = (1.0 - maxOverlap) * blockSize;
    int maxCorners = img.rows * img.cols / max(1.0, minDistance); // max. num. of keypoints

    double qualityLevel = 0.01; // minimal accepted quality of image corners
    double k = 0.04;

    // Apply corner detection
    auto t = (double) cv::getTickCount();
    vector<cv::Point2f> corners;
    cv::goodFeaturesToTrack(img, corners, maxCorners, qualityLevel, minDistance, cv::Mat(), blockSize, false, k);

    // add corners to result vector
    for (auto &corner : corners) {
        cv::KeyPoint newKeyPoint;
        newKeyPoint.pt = cv::Point2f(corner.x, corner.y);
        newKeyPoint.size = blockSize;
        keypoints.push_back(newKeyPoint);
    }
    t = ((double) cv::getTickCount() - t) / cv::getTickFrequency();
    // cout << "Shi-Tomasi detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;

    // visualize results
    if (bVis) {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = "Shi-Tomasi Corner Detector Results";
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }
}

void detKeypointsHarris(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis) {
    // Detector parameters
    int blockSize = 2;     // for every pixel, a blockSize × blockSize neighborhood is considered
    int apertureSize = 3;  // aperture parameter for Sobel operator (must be odd)
    int minResponse = 100; // minimum value for a corner in the 8bit scaled response matrix, only consider values greater in NMS
    double k = 0.04;       // Harris parameter (see equation for details)

    // Detect Harris corners and normalize output
    cv::Mat dst, dst_norm, dst_norm_scaled;
    dst = cv::Mat::zeros(img.size(), CV_32FC1);
    cv::cornerHarris(img, dst, blockSize, apertureSize, k, cv::BORDER_DEFAULT);
    cv::normalize(dst, dst_norm, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat());
    cv::convertScaleAbs(dst_norm, dst_norm_scaled);

    for (int r = 0; r < dst_norm.rows; ++r) {
        for (int c = 0; c < dst_norm.cols; ++c) {
            int val = (int) dst_norm.at<float>(r, c);
            if (val > minResponse) {
                // Create a new keypoint
                cv::KeyPoint new_keypoint(cv::Point2f(c, r), 2.0f * (float) apertureSize);
                new_keypoint.response = val;

                bool is_overlap = false;
                // Loop over existing keypoints to check if there are overlaps
                for (auto &keypoint : keypoints) {
                    float overlap = cv::KeyPoint::overlap(new_keypoint, keypoint);

                    if (overlap > 0.0f) {
                        is_overlap = true;

                        if (new_keypoint.response > keypoint.response)
                            keypoint = new_keypoint;
                        break;
                    }
                }
                if (!is_overlap)
                    keypoints.push_back(new_keypoint);
            }

        }
    }
    // visualize results
    if (bVis) {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = "Harris Corner Detector Results";
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }
}

void detKeypointsModern(vector<cv::KeyPoint> &keypoints, cv::Mat &img, const std::string &detectorType, bool bVis) {
    auto t = (double) cv::getTickCount();
    if (detectorType == "FAST") {
        cv::FAST(img, keypoints, 30, true);
    } else {
        cv::Ptr<cv::FeatureDetector> detector;
        if (detectorType == "BRISK") {
            detector = cv::BRISK::create();
        } else if (detectorType == "SIFT") {
            detector = cv::xfeatures2d::SIFT::create();
        } else if (detectorType == "ORB") {
            detector = cv::ORB::create();
        } else if (detectorType == "AKAZE") {
            detector = cv::AKAZE::create();
        } else {
            std::cerr << "Detector " << detectorType << " not available, choose a viable detector." << std::endl;
            return;
        }
        detector->detect(img, keypoints);
    }
    t = ((double) cv::getTickCount() - t) / cv::getTickFrequency();
    // cout << detectorType << " with n= " << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;

    // visualize results
    if (bVis) {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = "Harris Corner Detector Results";
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }
}
