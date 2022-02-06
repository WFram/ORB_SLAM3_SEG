/*
 * --------------------------------------------------------------------------------------------------
 * Copyright (C) 2018, iVip Lab @ EE, THU (https://ivip-tsinghua.github.io/iViP-Homepage/) and 
 * Advanced Mechanism and Roboticized Equipment Lab. All rights reserved.
 *
 * Licensed under the GPLv3 License;
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * https://github.com/ivipsourcecode/DS-SLAM/blob/master/LICENSE
 *--------------------------------------------------------------------------------------------------
 */

#ifndef SEGMENT_H
#define SEGMENT_H

#include "KeyFrame.h"
#include "Map.h"
#include "Tracking.h"
#include "libsegmentation.hpp" 
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <mutex>
#include <condition_variable>

namespace ORB_SLAM3
{

class Tracking;

class Segment
{

public:
    Segment(const string &pascal_prototxt, const string &pascal_caffemodel, const string &pascal_png, const string &strSettingsFile);
    void SetTracker(Tracking* pTracker);
    void Run();
    int conbase = 64, jinzhi=4;
    int labeldata[20]={32,8,40,2,34,10,42,16,48,24,56,18,50,26,58,4,36,12,44,6};

    int s_width;
    int s_height;

    cv::Mat label_colours;
    Classifier* classifier;
    bool isNewImgArrived();
    bool CheckFinish();
    void RequestFinish();
    bool isFinished();
    void SetFinish();
    void Initialize(const cv::Mat& img);
    cv::Mat mImg;
    cv::Mat mImgTemp;
    cv::Mat mImgSegment_color;
    cv::Mat mImgSegment_color_final;
    cv::Mat mImgSegment;
    cv::Mat mImgSegmentLatest;
    Tracking* mpTracker;
    std::mutex mMutexGetNewImg;
    std::mutex mMutexFinish;
    bool mbFinishRequested;
    bool mbFinished;
    void ProduceImgSegment();
    std::mutex mMutexNewImgSegment;
    std::condition_variable mbcvNewImgSegment;
    bool mbNewImgFlag;
    int mSkipIndex;
    double mSegmentTime;
    int imgIndex;
    // Paremeters for caffe
    string model_file;
    string trained_file;
    string LUT_file;
private:
    bool ParseSegmentationParamFile(cv::FileStorage &fSettings);
};

}



#endif
