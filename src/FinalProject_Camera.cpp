
/* INCLUDES FOR THIS PROJECT */
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <list>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <sstream>
#include <vector>
#include "camFusion.hpp"
#include "dataStructures.h"
#include "json.hpp"
#include "lidarData.hpp"
#include "matching2D.hpp"
#include "objectDetection2D.hpp"

using namespace std;

struct Params {
  bool showTopView;
  bool showDebugImg;
  bool showMatch;
  bool limitKpts;
  int maxKeypoints;
  int imgStartIndex;
  int imgEndIndex;
  int imgStepWidth;
  string detectorType;
  string descriptorMethod;
  string matcherType;
  string descriptorType;
  string selectorType;
  float confThreshold;
  float nmsThreshold;
  float shrinkFactor;
  int yoloInputSize;
  float ttc_lidar_outlier_threshold;

  Params() = default;
  ~Params() = default;
};

Params params;

void readConfig() {
  using json = nlohmann::json;

  std::ifstream json_stream("../src/config/params.json");
  json params_json;
  json_stream >> params_json;
  json_stream.close();

  params.showTopView = params_json["showTopView"];
  params.showDebugImg = params_json["showDebugImg"];
  params.showMatch = params_json["showMatch"];
  params.limitKpts = params_json["limitKpts"];
  params.maxKeypoints = params_json["maxKeypoints"];
  params.imgStartIndex = params_json["imgStartIndex"];
  params.imgEndIndex = params_json["imgEndIndex"];
  params.imgStepWidth = params_json["imgStepWidth"];
  params.detectorType = params_json["detectorType"];
  params.descriptorMethod = params_json["descriptorMethod"];
  params.matcherType = params_json["matcherType"];
  params.descriptorType = params_json["descriptorType"];
  params.selectorType = params_json["selectorType"];
  params.confThreshold = params_json["confThreshold"];
  params.nmsThreshold = params_json["nmsThreshold"];
  params.shrinkFactor = params_json["shrinkFactor"];
  params.yoloInputSize = params_json["yoloInputSize"];
  params.ttc_lidar_outlier_threshold =
      params_json["ttc_lidar_outlier_threshold"];

  /* params.filterRes = params_json["filterRes"];
  params.minPoint =
      Eigen::Vector4f(params_json["minPoint"][0], params_json["minPoint"][1],
                      params_json["minPoint"][2], 1);
  params.maxPoint =
      Eigen::Vector4f(params_json["maxPoint"][0], params_json["maxPoint"][1],
                      params_json["maxPoint"][2], 1);
  params.clusterTol = params_json["clusterTol"];
  params.clusterMinSize = params_json["clusterMinSize"];
  params.clusterMaxSize = params_json["clusterMaxSize"]; */
}

/* MAIN PROGRAM */
int main(int argc, const char *argv[]) {
  /* INIT VARIABLES AND DATA STRUCTURES */

  // Read config
  readConfig();

  // data location
  string dataPath = "../";

  // camera
  string imgBasePath = dataPath + "images/";
  // left camera, color
  string imgPrefix = "KITTI/2011_09_26/image_02/data/000000";
  string imgFileType = ".png";
  // first file index to load
  // (assumes Lidar and camera names have identical naming convention)
  int imgStartIndex = params.imgStartIndex;
  // last file index to load
  int imgEndIndex = params.imgEndIndex;
  int imgStepWidth = params.imgStepWidth;
  // no. of digits which make up the file index (e.g. img-0001.png)
  constexpr int imgFillWidth = 4;

  // object detection
  string yoloBasePath = dataPath + "dat/yolo/";
  string yoloClassesFile = yoloBasePath + "coco.names";
  string yoloModelConfiguration = yoloBasePath + "yolov3.cfg";
  string yoloModelWeights = yoloBasePath + "yolov3.weights";

  // Lidar
  string lidarPrefix = "KITTI/2011_09_26/velodyne_points/data/000000";
  string lidarFileType = ".bin";

  /* calibration data for camera and lidar */
  // 3x4 projection matrix after rectification
  cv::Mat P_rect_00(3, 4, cv::DataType<double>::type);
  // 3x3 rectifying rotation to make image planes co-planar
  cv::Mat R_rect_00(4, 4, cv::DataType<double>::type);
  // rotation matrix and translation vector
  cv::Mat RT(4, 4, cv::DataType<double>::type);

  RT.at<double>(0, 0) = 7.533745e-03;
  RT.at<double>(0, 1) = -9.999714e-01;
  RT.at<double>(0, 2) = -6.166020e-04;
  RT.at<double>(0, 3) = -4.069766e-03;
  RT.at<double>(1, 0) = 1.480249e-02;
  RT.at<double>(1, 1) = 7.280733e-04;
  RT.at<double>(1, 2) = -9.998902e-01;
  RT.at<double>(1, 3) = -7.631618e-02;
  RT.at<double>(2, 0) = 9.998621e-01;
  RT.at<double>(2, 1) = 7.523790e-03;
  RT.at<double>(2, 2) = 1.480755e-02;
  RT.at<double>(2, 3) = -2.717806e-01;
  RT.at<double>(3, 0) = 0.0;
  RT.at<double>(3, 1) = 0.0;
  RT.at<double>(3, 2) = 0.0;
  RT.at<double>(3, 3) = 1.0;

  R_rect_00.at<double>(0, 0) = 9.999239e-01;
  R_rect_00.at<double>(0, 1) = 9.837760e-03;
  R_rect_00.at<double>(0, 2) = -7.445048e-03;
  R_rect_00.at<double>(0, 3) = 0.0;
  R_rect_00.at<double>(1, 0) = -9.869795e-03;
  R_rect_00.at<double>(1, 1) = 9.999421e-01;
  R_rect_00.at<double>(1, 2) = -4.278459e-03;
  R_rect_00.at<double>(1, 3) = 0.0;
  R_rect_00.at<double>(2, 0) = 7.402527e-03;
  R_rect_00.at<double>(2, 1) = 4.351614e-03;
  R_rect_00.at<double>(2, 2) = 9.999631e-01;
  R_rect_00.at<double>(2, 3) = 0.0;
  R_rect_00.at<double>(3, 0) = 0;
  R_rect_00.at<double>(3, 1) = 0;
  R_rect_00.at<double>(3, 2) = 0;
  R_rect_00.at<double>(3, 3) = 1;

  P_rect_00.at<double>(0, 0) = 7.215377e+02;
  P_rect_00.at<double>(0, 1) = 0.000000e+00;
  P_rect_00.at<double>(0, 2) = 6.095593e+02;
  P_rect_00.at<double>(0, 3) = 0.000000e+00;
  P_rect_00.at<double>(1, 0) = 0.000000e+00;
  P_rect_00.at<double>(1, 1) = 7.215377e+02;
  P_rect_00.at<double>(1, 2) = 1.728540e+02;
  P_rect_00.at<double>(1, 3) = 0.000000e+00;
  P_rect_00.at<double>(2, 0) = 0.000000e+00;
  P_rect_00.at<double>(2, 1) = 0.000000e+00;
  P_rect_00.at<double>(2, 2) = 1.000000e+00;
  P_rect_00.at<double>(2, 3) = 0.000000e+00;

  /* misc */
  // frames per second for Lidar and camera
  double sensorFrameRate = 10.0 / imgStepWidth;
  // no. of images which are held in memory (ring buffer) at the same time
  constexpr int dataBufferSize = 2;
  // list of data frames which are held in memory at the same time
  list<DataFrame> dataBuffer;
  // visualize results
  bool bVis = false;

  /* MAIN LOOP OVER ALL IMAGES */
  for (size_t imgIndex = 0; imgIndex <= imgEndIndex - imgStartIndex;
       imgIndex += imgStepWidth) {
    /* LOAD IMAGE INTO BUFFER */

    // assemble filenames for current index
    ostringstream imgNumber;
    imgNumber << setfill('0') << setw(imgFillWidth) << imgStartIndex + imgIndex;
    string imgFullFilename =
        imgBasePath + imgPrefix + imgNumber.str() + imgFileType;

    // load image from file
    cv::Mat img = cv::imread(imgFullFilename);

    // push image into data frame buffer
    DataFrame frame;
    frame.cameraImg = img;
    if (dataBuffer.size() == dataBufferSize) {
      dataBuffer.pop_front();
    }
    dataBuffer.push_back(frame);

    cout << endl << "#1 : LOAD IMAGE INTO BUFFER done" << endl;

    /* DETECT & CLASSIFY OBJECTS */

    float confThreshold = params.confThreshold;
    float nmsThreshold = params.nmsThreshold;
    detectObjects(
        dataBuffer.rbegin()->cameraImg, dataBuffer.rbegin()->boundingBoxes,
        confThreshold, nmsThreshold, yoloBasePath, yoloClassesFile,
        yoloModelConfiguration, yoloModelWeights, params.yoloInputSize, bVis);

    cout << "#2 : DETECT & CLASSIFY OBJECTS done" << endl;

    /* CROP LIDAR POINTS */

    // load 3D Lidar points from file
    string lidarFullFilename =
        imgBasePath + lidarPrefix + imgNumber.str() + lidarFileType;
    std::vector<LidarPoint> lidarPoints;
    loadLidarFromFile(lidarPoints, lidarFullFilename);

    // remove Lidar points based on distance properties
    // focus on ego lane
    float minZ = -1.5, maxZ = -0.9, minX = 2.0, maxX = 20.0, maxY = 2.0,
          minR = 0.1;
    cropLidarPoints(lidarPoints, minX, maxX, maxY, minZ, maxZ, minR);

    dataBuffer.rbegin()->lidarPoints = lidarPoints;

    cout << "#3 : CROP LIDAR POINTS done" << endl;

    /* CLUSTER LIDAR POINT CLOUD */

    // associate Lidar points with camera-based ROI
    // shrinks each bounding box by the given percentage to avoid 3D object
    // merging at the edges of an ROI
    float shrinkFactor = params.shrinkFactor;
    clusterLidarWithROI(dataBuffer.rbegin()->boundingBoxes,
                        dataBuffer.rbegin()->lidarPoints, shrinkFactor,
                        P_rect_00, R_rect_00, RT);

    // Visualize 3D objects
    bVis = params.showTopView;
    if (bVis) {
      show3DObjects(dataBuffer.rbegin()->boundingBoxes, cv::Size(4.0, 20.0),
                    cv::Size(1200, 1200), imgNumber.str(), true);
    }
    bVis = false;

    cout << "#4 : CLUSTER LIDAR POINT CLOUD done" << endl;

    // REMOVE THIS LINE BEFORE PROCEEDING WITH THE FINAL PROJECT
    // skips directly to the next image without processing what comes beneath
    // continue;

    /* DETECT IMAGE KEYPOINTS */

    // convert current image to grayscale
    cv::Mat imgGray;
    cv::cvtColor(dataBuffer.rbegin()->cameraImg, imgGray, cv::COLOR_BGR2GRAY);

    /* extract 2D keypoints from current image */
    // create empty feature list for current image
    vector<cv::KeyPoint> keypoints;

    if (params.detectorType.compare("SHITOMASI") == 0) {
      detKeypointsShiTomasi(keypoints, imgGray, false, false);
    } else if (params.detectorType.compare("HARRIS") == 0) {
      detKeypointsShiTomasi(keypoints, imgGray, true, false);
    } else if (params.detectorType.compare("FAST") == 0 ||
               params.detectorType.compare("BRISK") == 0 ||
               params.detectorType.compare("ORB") == 0 ||
               params.detectorType.compare("AKAZE") == 0 ||
               params.detectorType.compare("SIFT") == 0) {
      detKeypointsModern(keypoints, imgGray, params.detectorType, false);
    }

    // optional : limit number of keypoints (helpful for debugging and learning)
    bool bLimitKpts = params.limitKpts;
    if (bLimitKpts) {
      int maxKeypoints = params.maxKeypoints;

      if (params.detectorType.compare("SHITOMASI") == 0) {
        // there is no response info, so keep the first 50 as they are sorted in
        // descending quality order
        keypoints.erase(keypoints.begin() + maxKeypoints, keypoints.end());
      }
      cv::KeyPointsFilter::retainBest(keypoints, maxKeypoints);
      cout << " NOTE: Keypoints have been limited!" << endl;
    }

    // push keypoints and descriptor for current frame to end of data buffer
    dataBuffer.rbegin()->keypoints = keypoints;

    cout << "#5 : DETECT KEYPOINTS done" << endl;

    /* EXTRACT KEYPOINT DESCRIPTORS */

    cv::Mat descriptors;
    // BRISK, BRIEF, ORB, FREAK, AKAZE, SIFT
    // string descriptorMethod = "BRISK";
    descKeypoints(dataBuffer.rbegin()->keypoints,
                  dataBuffer.rbegin()->cameraImg, descriptors,
                  params.descriptorMethod);

    // push descriptors for current frame to end of data buffer
    dataBuffer.rbegin()->descriptors = descriptors;

    cout << "#6 : EXTRACT DESCRIPTORS done" << endl;

    // wait until at least two images have been processed
    if (dataBuffer.size() > 1) {
      /* MATCH KEYPOINT DESCRIPTORS */

      vector<cv::DMatch> matches;
      // string matcherType = "MAT_BF";         // MAT_BF, MAT_FLANN
      // string descriptorType = "DES_BINARY";  // DES_BINARY, DES_HOG
      // string selectorType = "SEL_NN";        // SEL_NN, SEL_KNN

      matchDescriptors(
          next(dataBuffer.rbegin())->keypoints, dataBuffer.rbegin()->keypoints,
          next(dataBuffer.rbegin())->descriptors,
          dataBuffer.rbegin()->descriptors, matches, params.descriptorType,
          params.matcherType, params.selectorType);

      // store matches in current data frame
      dataBuffer.rbegin()->kptMatches = matches;

      cout << "#7 : MATCH KEYPOINT DESCRIPTORS done" << endl;

      /* TRACK 3D OBJECT BOUNDING BOXES */

      //// STUDENT ASSIGNMENT
      //// TASK FP.1 -> match list of 3D objects (vector<BoundingBox>) between
      /// current and previous frame (implement ->matchBoundingBoxes)
      map<int, int> bbBestMatches;
      // associate bounding boxes between current and previous frame using
      // keypoint matches
      matchBoundingBoxes(matches, bbBestMatches, *next(dataBuffer.rbegin()),
                         *dataBuffer.rbegin());
      //// EOF STUDENT ASSIGNMENT

      if (params.showDebugImg) {
        DataFrame &prev_frame = *next(dataBuffer.rbegin());
        DataFrame &curr_frame = *dataBuffer.rbegin();

        // Draw box
        // cv::Mat prev_img = prev_frame.cameraImg.clone();
        // cv::Mat curr_img = curr_frame.cameraImg.clone();
        vector<BoundingBox> &prev_boxes = prev_frame.boundingBoxes;
        vector<BoundingBox> &curr_boxes = curr_frame.boundingBoxes;
        /* for (auto &box : prev_boxes) {
          cv::rectangle(prev_img, box.roi, cv::Scalar(0, 255, 0));
        }
        for (auto &box : curr_boxes) {
          cv::rectangle(curr_img, box.roi, cv::Scalar(0, 255, 0));
        } */

        // cv::Mat img_array[] = {prev_img, curr_img};
        cv::Mat img_array[] = {prev_frame.cameraImg, curr_frame.cameraImg};
        cv::Mat concate_img;
        cv::vconcat(img_array, 2, concate_img);

        int idx = 0;
        vector<cv::Scalar> colors_list;
        colors_list.push_back(cv::Scalar(255, 0, 0));
        colors_list.push_back(cv::Scalar(0, 255, 0));
        colors_list.push_back(cv::Scalar(0, 0, 255));
        colors_list.push_back(cv::Scalar(255, 255, 0));
        colors_list.push_back(cv::Scalar(255, 0, 255));
        colors_list.push_back(cv::Scalar(0, 255, 255));
        for (auto box_pair : bbBestMatches) {
          cv::Point2f p1;
          p1.x = prev_boxes[box_pair.first].roi.x +
                 prev_boxes[box_pair.first].roi.width / 2;
          p1.y = prev_boxes[box_pair.first].roi.y +
                 prev_boxes[box_pair.first].roi.height / 2;
          cv::Point2f p2;
          p2.x = curr_boxes[box_pair.second].roi.x +
                 curr_boxes[box_pair.second].roi.width / 2;
          p2.y = curr_boxes[box_pair.second].roi.y +
                 curr_boxes[box_pair.second].roi.height / 2 +
                 curr_frame.cameraImg.rows;
          cv::line(concate_img, p1, p2, colors_list[idx], 2);

          // Draw prev box
          cv::rectangle(concate_img, prev_boxes[box_pair.first].roi,
                        colors_list[idx], 2);
          // Draw curr box
          cv::Point2f top_left;
          top_left.x = curr_boxes[box_pair.second].roi.x;
          top_left.y =
              curr_boxes[box_pair.second].roi.y + curr_frame.cameraImg.rows;
          cv::Point2f bottom_right;
          bottom_right.x = curr_boxes[box_pair.second].roi.x +
                           curr_boxes[box_pair.second].roi.width;
          bottom_right.y = curr_boxes[box_pair.second].roi.y +
                           curr_boxes[box_pair.second].roi.height +
                           curr_frame.cameraImg.rows;
          cv::rectangle(concate_img, top_left, bottom_right, colors_list[idx],
                        2);

          idx = (++idx) % colors_list.size();
        }

        // Draw match
        if (params.showMatch) {
          cv::RNG rng;
          for (int i = 0; i < matches.size(); i++) {
            cv::Scalar color = cv::Scalar(
                rng.uniform(0, 150), rng.uniform(0, 150), rng.uniform(0, 150));

            cv::DMatch &match = matches[i];
            int prev_idx = match.queryIdx;
            int curr_idx = match.trainIdx;
            cv::Point2f prev_point = prev_frame.keypoints[prev_idx].pt;
            cv::Point2f curr_point = curr_frame.keypoints[curr_idx].pt;
            curr_point.y += curr_frame.cameraImg.rows;
            cv::line(concate_img, prev_point, curr_point, color, 2);
            // cv::circle(img, Point(330, 90), 50, Scalar(0, 255, 0), -1);
          }
        }

        string windowName =
            "Matching keypoints between two camera images (best 10)";
        cv::namedWindow(windowName, 7);
        cv::imshow(windowName, concate_img);
        cv::waitKey(0);
      }
      // continue;

      // store matches in current data frame
      dataBuffer.rbegin()->bbMatches = bbBestMatches;

      cout << "#8 : TRACK 3D OBJECT BOUNDING BOXES done" << endl;

      /* COMPUTE TTC ON OBJECT IN FRONT */

      // loop over all BB match pairs
      for (auto it1 = dataBuffer.rbegin()->bbMatches.begin();
           it1 != dataBuffer.rbegin()->bbMatches.end(); ++it1) {
        // find bounding boxes associates with current match
        BoundingBox *prevBB, *currBB;
        for (auto it2 = dataBuffer.rbegin()->boundingBoxes.begin();
             it2 != dataBuffer.rbegin()->boundingBoxes.end(); ++it2) {
          // check wether current match partner corresponds to this BB
          if (it1->second == it2->boxID) {
            currBB = &(*it2);
          }
        }

        for (auto it2 = next(dataBuffer.rbegin())->boundingBoxes.begin();
             it2 != next(dataBuffer.rbegin())->boundingBoxes.end(); ++it2) {
          // check wether current match partner corresponds to this BB
          if (it1->first == it2->boxID) {
            prevBB = &(*it2);
          }
        }

        /* compute TTC for current match */
        // only compute TTC if we have Lidar points
        if (currBB->lidarPoints.size() > 0 && prevBB->lidarPoints.size() > 0) {
          //// STUDENT ASSIGNMENT
          //// TASK FP.2 -> compute time-to-collision based on Lidar data
          ///(implement -> computeTTCLidar)
          double ttcLidar;
          computeTTCLidar(prevBB->lidarPoints, currBB->lidarPoints,
                          sensorFrameRate, ttcLidar);
          double ttc_lidar_filtered;
          computeTTCLidar(prevBB->lidarPoints, currBB->lidarPoints,
                          sensorFrameRate, ttc_lidar_filtered, true,
                          params.ttc_lidar_outlier_threshold);
          cout << "ttcLidar: " << ttcLidar << endl;
          cout << "ttc_lidar_filtered: " << ttc_lidar_filtered << endl;
          //// EOF STUDENT ASSIGNMENT

          //// STUDENT ASSIGNMENT
          //// TASK FP.3 -> assign enclosed keypoint matches to bounding box
          ///(implement -> clusterKptMatchesWithROI) / TASK FP.4 -> compute
          /// time-to-collision based on camera (implement -> computeTTCCamera)
          double ttcCamera;
          clusterKptMatchesWithROI(
              *currBB, next(dataBuffer.rbegin())->keypoints,
              dataBuffer.rbegin()->keypoints, dataBuffer.rbegin()->kptMatches);
          computeTTCCamera(next(dataBuffer.rbegin())->keypoints,
                           dataBuffer.rbegin()->keypoints, currBB->kptMatches,
                           sensorFrameRate, ttcCamera);
          //// EOF STUDENT ASSIGNMENT

          bVis = true;
          if (bVis) {
            cv::Mat visImg = dataBuffer.rbegin()->cameraImg.clone();
            showLidarImgOverlay(visImg, currBB->lidarPoints, P_rect_00,
                                R_rect_00, RT, &visImg);
            cv::rectangle(visImg, cv::Point(currBB->roi.x, currBB->roi.y),
                          cv::Point(currBB->roi.x + currBB->roi.width,
                                    currBB->roi.y + currBB->roi.height),
                          cv::Scalar(0, 255, 0), 2);

            char str[200];
            sprintf(str, "TTC Lidar : %f s, TTC Camera : %f s", ttcLidar,
                    ttcCamera);
            putText(visImg, str, cv::Point2f(80, 50), cv::FONT_HERSHEY_PLAIN, 2,
                    cv::Scalar(0, 0, 255));

            string windowName = "Final Results : TTC";
            cv::namedWindow(windowName, 4);
            cv::imshow(windowName, visImg);
            cout << "Press key to continue to next frame" << endl;
            cv::waitKey(0);
          }
          bVis = false;

        }  // eof TTC computation
      }    // eof loop over all BB matches
    }

  }  // eof loop over all images

  return 0;
}
