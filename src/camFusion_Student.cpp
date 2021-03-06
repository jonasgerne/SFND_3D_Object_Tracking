
#include <iostream>
#include <algorithm>
#include <numeric>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "camFusion.hpp"
#include "dataStructures.h"
#include <pcl/filters/statistical_outlier_removal.h>

using namespace std;

// Create groups of Lidar points whose projection into the camera falls into the same bounding box
void clusterLidarWithROI(std::vector<BoundingBox> &boundingBoxes,
                         std::vector<LidarPoint> &lidarPoints,
                         float shrinkFactor,
                         cv::Mat &P_rect_xx,
                         cv::Mat &R_rect_xx,
                         cv::Mat &RT) {
    // loop over all Lidar points and associate them to a 2D bounding box
    cv::Mat X(4, 1, cv::DataType<double>::type);
    cv::Mat Y(3, 1, cv::DataType<double>::type);

    for (auto it1 = lidarPoints.begin(); it1 != lidarPoints.end(); ++it1) {
        // assemble vector for matrix-vector-multiplication
        X.at<double>(0, 0) = it1->x;
        X.at<double>(1, 0) = it1->y;
        X.at<double>(2, 0) = it1->z;
        X.at<double>(3, 0) = 1;

        // project Lidar point into camera
        Y = P_rect_xx * R_rect_xx * RT * X;
        cv::Point pt;
        pt.x = Y.at<double>(0, 0) / Y.at<double>(0, 2); // pixel coordinates
        pt.y = Y.at<double>(1, 0) / Y.at<double>(0, 2);

        vector<vector<BoundingBox>::iterator>
            enclosingBoxes; // pointers to all bounding boxes which enclose the current Lidar point
        for (vector<BoundingBox>::iterator it2 = boundingBoxes.begin(); it2 != boundingBoxes.end(); ++it2) {
            // shrink current bounding box slightly to avoid having too many outlier points around the edges
            cv::Rect smallerBox;
            smallerBox.x = (*it2).roi.x + shrinkFactor * (*it2).roi.width / 2.0;
            smallerBox.y = (*it2).roi.y + shrinkFactor * (*it2).roi.height / 2.0;
            smallerBox.width = (*it2).roi.width * (1 - shrinkFactor);
            smallerBox.height = (*it2).roi.height * (1 - shrinkFactor);

            // check wether point is within current bounding box
            if (smallerBox.contains(pt)) {
                enclosingBoxes.push_back(it2);
            }

        } // eof loop over all bounding boxes

        // check wether point has been enclosed by one or by multiple boxes
        if (enclosingBoxes.size() == 1) {
            // add Lidar point to bounding box
            enclosingBoxes[0]->lidarPoints.push_back(*it1);
        }

    } // eof loop over all Lidar points
}

void show3DObjects(std::vector<BoundingBox> &boundingBoxes,
                   const cv::Size &worldSize,
                   const cv::Size &imageSize,
                   bool bWait) {
    // create topview image
    cv::Mat topviewImg(imageSize, CV_8UC3, cv::Scalar(255, 255, 255));

    for (auto &boundingBox : boundingBoxes) {
        // create randomized color for current 3D object
        cv::RNG rng(boundingBox.boxID);
        cv::Scalar currColor = cv::Scalar(rng.uniform(0, 150), rng.uniform(0, 150), rng.uniform(0, 150));

        // plot Lidar points into top view image
        int top = 1e8, left = 1e8, bottom = 0.0, right = 0.0;
        float xwmin = 1e8, ywmin = 1e8, ywmax = -1e8;
        for (auto &lidarPoint : boundingBox.lidarPoints) {
            // world coordinates
            float xw = lidarPoint.x; // world position in m with x facing forward from sensor
            float yw = lidarPoint.y; // world position in m with y facing left from sensor
            xwmin = xwmin < xw ? xwmin : xw;
            ywmin = ywmin < yw ? ywmin : yw;
            ywmax = ywmax > yw ? ywmax : yw;

            // top-view coordinates
            int y = (-xw * imageSize.height / worldSize.height) + imageSize.height;
            int x = (-yw * imageSize.width / worldSize.width) + imageSize.width / 2;

            // find enclosing rectangle
            top = top < y ? top : y;
            left = left < x ? left : x;
            bottom = bottom > y ? bottom : y;
            right = right > x ? right : x;

            // draw individual point
            cv::circle(topviewImg, cv::Point(x, y), 4, currColor, -1);
        }

        // draw enclosing rectangle
        cv::rectangle(topviewImg, cv::Point(left, top), cv::Point(right, bottom), cv::Scalar(0, 0, 0), 2);

        // augment object with some key data
        char str1[200], str2[200];
        sprintf(str1, "id=%d, #pts=%d", boundingBox.boxID, (int) boundingBox.lidarPoints.size());
        putText(topviewImg, str1, cv::Point2f(left - 250, bottom + 50), cv::FONT_ITALIC, 2, currColor);
        sprintf(str2, "xmin=%2.2f m, yw=%2.2f m", xwmin, ywmax - ywmin);
        putText(topviewImg, str2, cv::Point2f(left - 250, bottom + 125), cv::FONT_ITALIC, 2, currColor);
    }

    // plot distance markers
    float lineSpacing = 2.0; // gap between distance markers
    int nMarkers = floor(worldSize.height / lineSpacing);
    for (size_t i = 0; i < nMarkers; ++i) {
        int y = (-(i * lineSpacing) * imageSize.height / worldSize.height) + imageSize.height;
        cv::line(topviewImg, cv::Point(0, y), cv::Point(imageSize.width, y), cv::Scalar(255, 0, 0));
    }

    // display image
    string windowName = "3D Objects";
    cv::namedWindow(windowName, 1);
    cv::imshow(windowName, topviewImg);

    if (bWait) {
        cv::waitKey(0); // wait for key to be pressed
    }
}

// associate a given bounding box with the keypoints it contains
void clusterKptMatchesWithROI(BoundingBox &boundingBox,
                              std::vector<cv::KeyPoint> &kptsPrev,
                              std::vector<cv::KeyPoint> &kptsCurr,
                              std::vector<cv::DMatch> &kptMatches) {
    // calculate a robust mean
    std::vector<float> distances;
    std::vector<cv::DMatch> inliers;

    for (const cv::DMatch &match : kptMatches) {
        if (boundingBox.roi.contains(kptsCurr[match.trainIdx].pt)) {
            distances.push_back(match.distance);
            inliers.push_back(match);
        }
    }

    std::sort(distances.begin(), distances.end());

    float distance_median = distances.size() % 2 ? distances[distances.size() / 2] : distances[distances.size() / 2 + 1];
    float std{0.0f};
    for (float d: distances)
        std += (d - distance_median) * (d - distance_median);
    std /= (float) distances.size();
    std = std::sqrt(std);

    for (const cv::DMatch &match : inliers) {
        if ((distance_median - std) < match.distance && match.distance < (distance_median + std))
            boundingBox.kptMatches.push_back(match);
    }
}

// Compute time-to-collision (TTC) based on keypoint correspondences in successive images
void computeTTCCamera(std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr,
                      std::vector<cv::DMatch> kptMatches, double frameRate, double &TTC, cv::Mat *visImg) {
    // compute distance ratios between all matched keypoints
    vector<double> distRatios; // stores the distance ratios for all keypoints between curr. and prev. frame
    for (auto it1 = kptMatches.begin(); it1 != kptMatches.end() - 1; ++it1) { // outer kpt. loop

        // get current keypoint and its matched partner in the prev. frame
        cv::KeyPoint kpOuterCurr = kptsCurr.at(it1->trainIdx);
        cv::KeyPoint kpOuterPrev = kptsPrev.at(it1->queryIdx);

        for (auto it2 = kptMatches.begin() + 1; it2 != kptMatches.end(); ++it2) { // inner kpt.-loop

            double minDist = 100.0; // min. required distance

            // get next keypoint and its matched partner in the prev. frame
            cv::KeyPoint kpInnerCurr = kptsCurr.at(it2->trainIdx);
            cv::KeyPoint kpInnerPrev = kptsPrev.at(it2->queryIdx);

            // compute distances and distance ratios
            double distCurr = cv::norm(kpOuterCurr.pt - kpInnerCurr.pt);
            double distPrev = cv::norm(kpOuterPrev.pt - kpInnerPrev.pt);

            if (distPrev > std::numeric_limits<double>::epsilon() && distCurr >= minDist) { // avoid division by zero

                double distRatio = distCurr / distPrev;
                distRatios.push_back(distRatio);
            }
        } // eof inner loop over all matched kpts
    }     // eof outer loop over all matched kpts

    // only continue if list of distance ratios is not empty
    if (distRatios.empty()) {
        TTC = NAN;
        return;
    }

    // compute camera-based TTC from distance ratios
    double dT = 1 / frameRate;

    double medianDistRatio;
    auto size = distRatios.size();
    std::sort(distRatios.begin(), distRatios.end());
    if (size % 2 == 0)
        medianDistRatio = (distRatios[size / 2 - 1] + distRatios[size / 2]) / 2.0;
    else
        medianDistRatio = distRatios[size / 2];

    TTC = -dT / (1 - medianDistRatio);

}

//static void outlierFiltering(const std::vector<LidarPoint> &points, const pcl::PointCloud<pcl::PointXYZ>::Ptr &result) {
//    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_in(new pcl::PointCloud<pcl::PointXYZ>);
//
//    for (const auto &p : points)
//        cloud_in->push_back(pcl::PointXYZ((float) p.x, (float) p.y, 0.0f));
//
//    pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sorfilter;
//    sorfilter.setInputCloud(cloud_in);
//    sorfilter.setMeanK(20);
//    sorfilter.setStddevMulThresh(1.0);
//    sorfilter.filter(*result);
//    //std::cout << cloud_in->points.size() << " to " << result->points.size() << " points." << std::endl;
//}
//
//void computeTTCLidar(std::vector<LidarPoint> &lidarPointsPrev,
//                     std::vector<LidarPoint> &lidarPointsCurr, double frameRate, double &TTC) {
//    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_prev(new pcl::PointCloud<pcl::PointXYZ>);
//    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_curr(new pcl::PointCloud<pcl::PointXYZ>);
//    outlierFiltering(lidarPointsPrev, cloud_prev);
//    outlierFiltering(lidarPointsCurr, cloud_curr);
//
//    // auxiliary variables
//    double dT = 1 / frameRate; // time between two measurements in seconds
//
//    // find closest distance to Lidar points
//    double minXPrev = 1e9, minXCurr = 1e9;
//    for (const auto &lidar_point_prev : cloud_prev->points) {
//        minXPrev = minXPrev > lidar_point_prev.x ? lidar_point_prev.x : minXPrev;
//    }
//
//    for (const auto &lidar_point_curr : cloud_curr->points) {
//        minXCurr = minXCurr > lidar_point_curr.x ? lidar_point_curr.x : minXCurr;
//    }
//
//    // compute TTC from both measurements
//    TTC = minXCurr * dT / (minXPrev - minXCurr);
//}

void computeTTCLidar(std::vector<LidarPoint> &lidarPointsPrev,
                     std::vector<LidarPoint> &lidarPointsCurr, double frameRate, double &TTC) {

    // second variant above
    double dT = 1 / frameRate; // time between two measurements in seconds

    double laneWidth = 1.46;              // width of the preceding lidar area
    double yEdge = (laneWidth-0.2) / 2;

    auto checkFunc = [&yEdge](const LidarPoint &lp){return abs(lp.y) >= yEdge;};

    lidarPointsPrev.erase(std::remove_if(lidarPointsPrev.begin(), lidarPointsPrev.end(), checkFunc),
                          lidarPointsPrev.end());

    lidarPointsCurr.erase(std::remove_if(lidarPointsCurr.begin(), lidarPointsCurr.end(), checkFunc),
                          lidarPointsCurr.end());

    auto comparison = [](LidarPoint& p1, LidarPoint& p2) -> bool {return p1.x < p2.x;};
    std::sort(lidarPointsPrev.begin(), lidarPointsPrev.end(), comparison);
    std::sort(lidarPointsCurr.begin(), lidarPointsCurr.end(), comparison);

    // take median of lower quartile
    int prev_idx = std::floor(lidarPointsPrev.size() * 0.15 * 0.5);
    int curr_idx = std::floor(lidarPointsCurr.size() * 0.15  * 0.5);
    double prev_median = lidarPointsPrev[prev_idx].x;
    double curr_median = lidarPointsCurr[curr_idx].x;

    // compute TTC from both measurements
    TTC = curr_median * dT / (prev_median - curr_median);
}

void matchBoundingBoxes(std::vector<cv::DMatch> &matches,
                        std::map<int, int> &bbBestMatches,
                        DataFrame &prevFrame,
                        DataFrame &currFrame) {
    // loop over all matches and store the IDs of the enclosing bounding boxes
    std::map<std::pair<int, int>, int> bb_matches;
    for (const cv::DMatch &match : matches) {
        // check if this makes sense, confusing (train/query)
        const cv::Point &keypoint_prev = prevFrame.keypoints.at(match.queryIdx).pt;
        const cv::Point &keypoint_curr = currFrame.keypoints.at(match.trainIdx).pt;
        for (const BoundingBox &bb_prev : prevFrame.boundingBoxes) {
            if (bb_prev.roi.contains(keypoint_prev)) {
                for (const BoundingBox &bb_curr : currFrame.boundingBoxes) {
                    if (bb_curr.roi.contains(keypoint_curr))
                        bb_matches[std::make_pair(bb_prev.boxID, bb_curr.boxID)] += 1;
                }
            }
        }
    }

    // use a multimap to sort the matches
    std::multimap<int, std::pair<int, int>, std::greater<>> bb_match_multimap;

    for (const auto &bb_match : bb_matches) 
        bb_match_multimap.insert(std::make_pair(bb_match.second, bb_match.first));

    // extract the most likely matches
    bbBestMatches.clear();
    for (const auto &bb_match : bb_match_multimap) {
        if (bbBestMatches.count(bb_match.second.first) == 0)
            bbBestMatches.insert(bb_match.second);
    }
}
