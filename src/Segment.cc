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

#include "Segment.h"
#include "Tracking.h"
//#include "Camera.h"
#include <fstream>
#define SKIP_NUMBER 1
using namespace std;

namespace ORB_SLAM3
{
Segment::Segment(const string &pascal_prototxt, const string &pascal_caffemodel, const string &pascal_png, const string &strSettingPath)
                :mbFinishRequested(false),mSkipIndex(SKIP_NUMBER),mSegmentTime(0),imgIndex(0)
{

    model_file = pascal_prototxt;
    trained_file = pascal_caffemodel;
    LUT_file = pascal_png;

    label_colours = cv::imread(LUT_file,1);
    cv::cvtColor(label_colours, label_colours, cv::COLOR_RGB2BGR);

    cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);

//    bool bParse = ParseSegmentationParamFile(fSettings);
    s_width = fSettings["Camera.width"];
    s_height = fSettings["Camera.height"];

    // if(!bParse)
    // {
    //     std::cerr << "**ERROR in the config file, the format is not correct**" << std::endl;
    //     try
    //     {
    //         throw -1;
    //     }
    //     catch(exception &e)
    //     {

    //     }
    // }
    // std::cout << "Img width: " << s_width << " Img height: " << s_height << std::endl;
    mImgSegmentLatest=cv::Mat(s_height,s_width,CV_8UC1);
    
    mImgSegment_color_final=cv::Mat(s_height,s_width,CV_8UC3);
    
    mbNewImgFlag=false;

}

void Segment::SetTracker(Tracking *pTracker)
{
    mpTracker=pTracker;
}

bool Segment::isNewImgArrived()
{
    unique_lock<mutex> lock(mMutexGetNewImg);
    if(mbNewImgFlag)
    {
        mbNewImgFlag=false;
        return true;
    }
    else
    return false;
}

void Segment::Run()
{

    classifier=new Classifier(model_file, trained_file);
    cout << "Load model ..."<<endl;
    vector<float> vTimesDetectSegNet;
    while(1)
    {
        usleep(1);
        if(!isNewImgArrived())
            continue;
        if(!mImg.empty())
        {
            cout << "Wait for new RGB img time =" << endl;
            if(mSkipIndex==SKIP_NUMBER)
            {
                std::chrono::steady_clock::time_point t3 = std::chrono::steady_clock::now();
                // Recognize by Semantic segmentation
                mImgSegment=classifier->Predict(mImg, label_colours);
                std::chrono::steady_clock::time_point t4 = std::chrono::steady_clock::now();
                double mPredictionTime = std::chrono::duration_cast<std::chrono::duration<double> >(t4 - t3).count();
                std::cout << "Prediction time =" << mPredictionTime*1000 << std::endl;

                mImgSegment_color = mImgSegment.clone();
                cv::cvtColor(mImgSegment,mImgSegment_color, cv::COLOR_GRAY2BGR);

                LUT(mImgSegment_color, label_colours, mImgSegment_color_final);
                cv::resize(mImgSegment, mImgSegment, cv::Size(s_width,s_height) );
                cv::resize(mImgSegment_color_final, mImgSegment_color_final, cv::Size(s_width,s_height) );
                cv::Mat temp = mImgSegment_color_final;
                int morph_size = 15;
                cv::Mat kernel = getStructuringElement( cv::MORPH_ELLIPSE, cv::Size( 2*morph_size + 1, 2*morph_size+1 ), cv::Point( morph_size, morph_size ) );
                cv::morphologyEx(mImgSegment_color_final, temp, 2, kernel);
                std::chrono::steady_clock::time_point t5 = std::chrono::steady_clock::now();
                double mThreadTime = std::chrono::duration_cast<std::chrono::duration<double> >(t5 - t3).count();
                std::cout << "In thread process time =" << mThreadTime*1000 << std::endl;
                vTimesDetectSegNet.push_back(mSegmentTime);
                mSkipIndex=0;
                imgIndex++;
                mImgSegment_color_final = temp.clone();
                cv::imshow("SegNet", mImgSegment_color_final);
                cv::waitKey(1);
            }
            mSkipIndex++;
            std::chrono::steady_clock::time_point t6 = std::chrono::steady_clock::now();
            ProduceImgSegment();
            std::chrono::steady_clock::time_point t7 = std::chrono::steady_clock::now();
            double mProduceImgTime = std::chrono::duration_cast<std::chrono::duration<double> >(t7 - t6).count();
            std::cout << "Image producing time =" << mProduceImgTime*1000 << std::endl;
            
        }
        if(CheckFinish())
        {
            break;
        }
    }

    // SegNet time statistics
    sort(vTimesDetectSegNet.begin(),vTimesDetectSegNet.end());
    float totaltime = 0;
    for(int ni=0; ni<vTimesDetectSegNet.size(); ni++)
    {
        totaltime+=vTimesDetectSegNet[ni];
    }
    ofstream tStatics;
    cout << "-------" << endl << endl;
    cout << "Total KFs SegNet processed: " << vTimesDetectSegNet.size() << endl;
    cout << "Median SegNet time: " << vTimesDetectSegNet[vTimesDetectSegNet.size()/2] << endl;
    cout << "Mean SegNet time: " << totaltime/vTimesDetectSegNet.size() << endl;
    tStatics.open("time_statistic_SegNet.txt");
    tStatics << "Total KFs SegNet processed: " << vTimesDetectSegNet.size() << endl;
    tStatics << "Median SegNet time: " << vTimesDetectSegNet[vTimesDetectSegNet.size()/2] << endl;
    tStatics << "Mean SegNet time: " << totaltime/vTimesDetectSegNet.size() << endl;
    tStatics.close();
    SetFinish();

}

bool Segment::CheckFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    return mbFinishRequested;
}
  
void Segment::RequestFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    mbFinishRequested=true;
}

bool Segment::isFinished()
{
    unique_lock<mutex> lock(mMutexFinish);
    return mbFinished;
}

void Segment::SetFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    mbFinished = true;    
}

void Segment::ProduceImgSegment()
{
    std::unique_lock <std::mutex> lock(mMutexNewImgSegment);
    mImgTemp=mImgSegmentLatest;
    mImgSegmentLatest=mImgSegment;
    mImgSegment=mImgTemp;
    
}

// bool Segment::ParseSegmentationParamFile(cv::FileStorage &fSettings)
// {
//     bool b_miss_params = false;

//     cv::FileNode node = fSettings["Camera.width"];
//     if(!node.empty())
//     {
//         s_width = node.real();
//     }
//     else
//     {
//         std::cerr << "*Camera.width parameter doesn't exist or is not a real number*" << std::endl;
//         b_miss_params = true;
//     }

//     node = fSettings["Camera.height"];
//     if(!node.empty())
//     {
//         s_height = node.real();
//     }
//     else
//     {
//         std::cerr << "*Camera.height parameter doesn't exist or is not a real number*" << std::endl;
//         b_miss_params = true;
//     }
// }

}   //ORB_SLAM3
