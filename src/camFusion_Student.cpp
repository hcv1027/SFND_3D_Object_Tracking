#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <opencv2/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "camFusion.hpp"
#include "dataStructures.h"

using namespace std;

// Create groups of Lidar points whose projection into the camera falls into the
// same bounding box
void clusterLidarWithROI(std::vector<BoundingBox> &boundingBoxes,
                         std::vector<LidarPoint> &lidarPoints,
                         float shrinkFactor, cv::Mat &P_rect_xx,
                         cv::Mat &R_rect_xx, cv::Mat &RT) {
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
    // pixel coordinates
    pt.x = Y.at<double>(0, 0) / Y.at<double>(2, 0);
    pt.y = Y.at<double>(1, 0) / Y.at<double>(2, 0);

    // pointers to all bounding boxes which enclose the current Lidar point
    vector<vector<BoundingBox>::iterator> enclosingBoxes;
    for (vector<BoundingBox>::iterator it2 = boundingBoxes.begin();
         it2 != boundingBoxes.end(); ++it2) {
      // shrink current bounding box slightly to avoid having too many outlier
      // points around the edges
      cv::Rect smallerBox;
      smallerBox.x = (*it2).roi.x + shrinkFactor * (*it2).roi.width / 2.0;
      smallerBox.y = (*it2).roi.y + shrinkFactor * (*it2).roi.height / 2.0;
      smallerBox.width = (*it2).roi.width * (1 - shrinkFactor);
      smallerBox.height = (*it2).roi.height * (1 - shrinkFactor);

      // check wether point is within current bounding box
      if (smallerBox.contains(pt)) {
        enclosingBoxes.push_back(it2);
      }

    }  // eof loop over all bounding boxes

    // check wether point has been enclosed by one or by multiple boxes
    if (enclosingBoxes.size() == 1) {
      // add Lidar point to bounding box
      enclosingBoxes[0]->lidarPoints.push_back(*it1);
    }

  }  // eof loop over all Lidar points
}

/*
 * The show3DObjects() function below can handle different output image sizes,
 * but the text output has been manually tuned to fit the 2000x2000 size.
 * However, you can make this function work for other sizes too.
 * For instance, to use a 1000x1000 size, adjusting the text positions by
 * dividing them by 2.
 */
void show3DObjects(std::vector<BoundingBox> &boundingBoxes, cv::Size worldSize,
                   cv::Size imageSize, const std::string &image_name,
                   bool bWait) {
  // create topview image
  cv::Mat topviewImg(imageSize, CV_8UC3, cv::Scalar(255, 255, 255));

  std::cout << "box size: " << boundingBoxes.size() << std::endl;
  for (auto it1 = boundingBoxes.begin(); it1 != boundingBoxes.end(); ++it1) {
    // create randomized color for current 3D object
    // std::cout << "boxID: " << it1->boxID << std::endl;
    // std::cout << "lidarPoints: " << it1->lidarPoints.size() << std::endl;
    cv::RNG rng(it1->boxID);
    cv::Scalar currColor = cv::Scalar(rng.uniform(0, 150), rng.uniform(0, 150),
                                      rng.uniform(0, 150));

    // plot Lidar points into top view image
    int top = 1e8, left = 1e8, bottom = 0.0, right = 0.0;
    float xwmin = 1e8, ywmin = 1e8, ywmax = -1e8;
    for (auto it2 = it1->lidarPoints.begin(); it2 != it1->lidarPoints.end();
         ++it2) {
      /*  world coordinates */
      // world position in m with x facing forward from sensor
      float xw = (*it2).x;
      // world position in m with y facing left from sensor
      float yw = (*it2).y;
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
    cv::rectangle(topviewImg, cv::Point(left, top), cv::Point(right, bottom),
                  cv::Scalar(0, 0, 0), 2);

    // augment object with some key data
    char str1[200], str2[200];
    // float pos_ratio = (float)worldSize.height / 2000;
    float pos_ratio = 1.0;
    sprintf(str1, "id=%d, #pts=%d", it1->boxID, (int)it1->lidarPoints.size());
    putText(topviewImg, str1,
            cv::Point2f(pos_ratio * (left - 250), pos_ratio * (bottom + 50)),
            cv::FONT_ITALIC, 2, currColor);
    sprintf(str2, "xmin=%2.2f m, yw=%2.2f m", xwmin, ywmax - ywmin);
    putText(topviewImg, str2,
            cv::Point2f(pos_ratio * (left - 250), pos_ratio * (bottom + 125)),
            cv::FONT_ITALIC, 2, currColor);
  }

  // plot distance markers
  float lineSpacing = 2.0;  // gap between distance markers
  int nMarkers = floor(worldSize.height / lineSpacing);
  for (size_t i = 0; i < nMarkers; ++i) {
    int y = (-(i * lineSpacing) * imageSize.height / worldSize.height) +
            imageSize.height;
    cv::line(topviewImg, cv::Point(0, y), cv::Point(imageSize.width, y),
             cv::Scalar(255, 0, 0));
  }

  // display image
  string windowName = "3D Objects";
  cv::namedWindow(windowName, 1);

  if (bWait) {
    cv::imshow(windowName, topviewImg);
    cv::waitKey(0);  // wait for key to be pressed
  } else {
    cv::imwrite("../output/" + image_name + ".png", topviewImg);
  }
}

/* bool PatternDetector::refineMatchesWithHomography(
    const vector<cv::KeyPoint> &queryKeypoints,
    const vector<cv::KeyPoint> &trainKeypoints, float reprojectionThreshold,
    vector<cv::DMatch> &matches, cv::Mat &homography) {
  const int minNumberMatchesAllowed = 8;
  if (matches.size() < minNumberMatchesAllowed) {
    return false;
  }

  // Prepare data for cv::findHomography
  vector<cv::Point2f> srcPoints(matches.size());
  vector<cv::Point2f> dstPoints(matches.size());
  for (size_t i = 0; i < matches.size(); i++) {
    srcPoints[i] = trainKeypoints[matches[i].trainIdx].pt;
    dstPoints[i] = queryKeypoints[matches[i].queryIdx].pt;
  }

  // Find homography matrix and get inliers mask
  vector<unsigned char> inliersMask(srcPoints.size());
  homography = cv::findHomography(srcPoints, dstPoints, CV_FM_RANSAC,
                                  reprojectionThreshold, inliersMask);
  vector<cv::DMatch> inliers;
  for (size_t i = 0; i < inliersMask.size(); i++) {
    if (inliersMask[i]) {
      inliers.push_back(matches[i]);
    }
  }
  matches.swap(inliers);
  return matches.size() > minNumberMatchesAllowed;
} */

// associate a given bounding box with the keypoints it contains
void clusterKptMatchesWithROI(BoundingBox &boundingBox,
                              std::vector<cv::KeyPoint> &kptsPrev,
                              std::vector<cv::KeyPoint> &kptsCurr,
                              std::vector<cv::DMatch> &kptMatches,
                              std::vector<cv::DMatch> &outlierMatches) {
  vector<cv::DMatch> box_matches;
  // Step 1: Add match whose keypoint in current frame is in boundint box
  for (int i = 0; i < kptMatches.size(); i++) {
    cv::DMatch &match = kptMatches[i];
    cv::Point2f &prev_point = kptsPrev[match.queryIdx].pt;
    cv::Point2f &curr_point = kptsCurr[match.trainIdx].pt;
    if (boundingBox.roi.contains(curr_point)) {
      box_matches.push_back(match);
    }
  }

  // Step 2: Computing homography matrix to eliminate the outlier matches.
  vector<cv::Point2f> srcPoints(box_matches.size());
  vector<cv::Point2f> dstPoints(box_matches.size());
  for (size_t i = 0; i < box_matches.size(); i++) {
    srcPoints[i] = kptsPrev[box_matches[i].queryIdx].pt;
    dstPoints[i] = kptsCurr[box_matches[i].trainIdx].pt;
  }

  // Find homography matrix and get inliers mask
  vector<unsigned char> inliersMask(box_matches.size());
  double ransacReprojThreshold = 5;
  cv::Mat homography = cv::findHomography(srcPoints, dstPoints, cv::RANSAC,
                                          ransacReprojThreshold, inliersMask);
  // vector<cv::DMatch> box_matches_inliers;
  for (size_t i = 0; i < inliersMask.size(); i++) {
    if (inliersMask[i]) {
      boundingBox.kptMatches.emplace_back(box_matches[i]);
    } else {
      outlierMatches.emplace_back(box_matches[i]);
    }
  }
}

// Compute time-to-collision (TTC) based on keypoint correspondences in
// successive images
void computeTTCCamera(std::vector<cv::KeyPoint> &kptsPrev,
                      std::vector<cv::KeyPoint> &kptsCurr,
                      std::vector<cv::DMatch> kptMatches, double frameRate,
                      double &TTC, cv::Mat *visImg) {
  // compute distance ratios between all matched keypoints
  vector<double> distRatios;  // stores the distance ratios for all keypoints
                              // between curr. and prev. frame
  // outer kpt. loop
  for (auto it1 = kptMatches.begin(); it1 != kptMatches.end() - 1; ++it1) {
    // get current keypoint and its matched partner in the prev. frame
    cv::KeyPoint kpOuterCurr = kptsCurr.at(it1->trainIdx);
    cv::KeyPoint kpOuterPrev = kptsPrev.at(it1->queryIdx);

    // inner kpt.-loop
    for (auto it2 = kptMatches.begin() + 1; it2 != kptMatches.end(); ++it2) {
      double minDist = 100.0;  // min. required distance

      // get next keypoint and its matched partner in the prev. frame
      cv::KeyPoint kpInnerCurr = kptsCurr.at(it2->trainIdx);
      cv::KeyPoint kpInnerPrev = kptsPrev.at(it2->queryIdx);

      // compute distances and distance ratios
      double distCurr = cv::norm(kpOuterCurr.pt - kpInnerCurr.pt);
      double distPrev = cv::norm(kpOuterPrev.pt - kpInnerPrev.pt);

      // avoid division by zero
      if (distPrev > std::numeric_limits<double>::epsilon() &&
          distCurr >= minDist) {
        double distRatio = distCurr / distPrev;
        distRatios.push_back(distRatio);
      }
    }  // eof inner loop over all matched kpts
  }    // eof outer loop over all matched kpts

  // only continue if list of distance ratios is not empty
  if (distRatios.size() == 0) {
    TTC = NAN;
    return;
  }

  double medianDistRatio = 0.0;
  if (distRatios.size() % 2 == 0) {
    const auto median_it1 = distRatios.begin() + distRatios.size() / 2 - 1;
    const auto median_it2 = std::next(median_it1);

    std::nth_element(distRatios.begin(), median_it1, distRatios.end());
    std::nth_element(distRatios.begin(), median_it2, distRatios.end());

    medianDistRatio = (*median_it1 + *median_it2) / 2;
  } else {
    const auto median_it = distRatios.begin() + distRatios.size() / 2;
    std::nth_element(distRatios.begin(), median_it, distRatios.end());
    medianDistRatio = *median_it;
  }

  double delta_time = 1.0 / frameRate;
  TTC = -delta_time / (1 - medianDistRatio);
}

void computeTTCLidar(std::vector<LidarPoint> &lidarPointsPrev,
                     std::vector<LidarPoint> &lidarPointsCurr,
                     double sensorFrameRate, double &TTC, bool filter_outlier,
                     double outlier_threshold) {
  // auxiliary variables
  // constexpr double dT = 0.1;         // time between two measurements in
  // seconds constexpr double laneWidth = 4.0;  // assumed width of the ego lane
  // constexpr double halfLaneWidth = laneWidth / 2;
  double delta_time = 1.0 / sensorFrameRate;

  vector<LidarPoint> prev_lidar_point(lidarPointsPrev);
  vector<LidarPoint> curr_lidar_point(lidarPointsCurr);
  auto point_compare = [](const LidarPoint &p1, const LidarPoint &p2) -> bool {
    return p1.x < p2.x;
  };
  sort(prev_lidar_point.begin(), prev_lidar_point.end(), point_compare);
  sort(curr_lidar_point.begin(), curr_lidar_point.end(), point_compare);

  // find closest distance to Lidar points within ego lane
  double minXPrev = 1e9;
  double minXCurr = 1e9;

  if (filter_outlier) {
    auto closest_x_without_outlier =
        [outlier_threshold](vector<LidarPoint> &lidar_points) {
          for (int i = 0; i < lidar_points.size() - 1; ++i) {
            auto &p1 = lidar_points[i];
            auto &p2 = lidar_points[i + 1];
            if (fabs(p1.x - p2.x) < outlier_threshold) {
              return lidar_points[i].x;
            }
          }
          return lidar_points[0].x;
        };

    minXPrev = closest_x_without_outlier(prev_lidar_point);
    minXCurr = closest_x_without_outlier(curr_lidar_point);
    cout << "filtered minXPrev: " << minXPrev << endl;
    cout << "filtered minXCurr: " << minXCurr << endl;
  } else {
    minXPrev = prev_lidar_point[0].x;
    minXCurr = curr_lidar_point[0].x;

    cout << "minXPrev: " << minXPrev << endl;
    cout << "minXCurr: " << minXCurr << endl;
  }

  // compute TTC from both measurements
  TTC = minXCurr * delta_time / (minXPrev - minXCurr);
}

void matchBoundingBoxes(std::vector<cv::DMatch> &matches,
                        std::map<int, int> &bbBestMatches, DataFrame &prevFrame,
                        DataFrame &currFrame) {
  vector<BoundingBox> &prev_boxes = prevFrame.boundingBoxes;
  vector<BoundingBox> &curr_boxes = currFrame.boundingBoxes;
  vector<vector<int>> box_matching_count(prev_boxes.size(),
                                         vector<int>(curr_boxes.size(), 0));

  for (int i = 0; i < matches.size(); i++) {
    cv::DMatch &match = matches[i];
    cv::Point2f &prev_point = prevFrame.keypoints[match.queryIdx].pt;
    cv::Point2f &curr_point = currFrame.keypoints[match.trainIdx].pt;
    int prev_box_id = 0;
    int curr_box_id = 0;

    for (auto &box : prev_boxes) {
      if (box.roi.contains(prev_point)) {
        prev_box_id = box.boxID;
        break;
      }
    }
    for (auto &box : curr_boxes) {
      if (box.roi.contains(curr_point)) {
        curr_box_id = box.boxID;
        break;
      }
    }
    box_matching_count[prev_box_id][curr_box_id]++;
  }

  for (size_t i = 0; i < prev_boxes.size(); i++) {
    vector<int> &match_size = box_matching_count[i];
    auto best_match = max_element(match_size.begin(), match_size.end());
    if (*best_match > 50) {
      int match_box_id = std::distance(match_size.begin(), best_match);
      bbBestMatches.insert(make_pair(i, match_box_id));
    }
  }
}
