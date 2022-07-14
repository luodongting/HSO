// This file is part of HSO: Hybrid Sparse Monocular Visual Odometry 
// With Online Photometric Calibration
//
// Copyright(c) 2021, Dongting Luo, Dalian University of Technology, Dalian
// Copyright(c) 2021, Robotics Group, Dalian University of Technology
//
// This program is highly based on the previous implementation 
// of SVO: https://github.com/uzh-rpg/rpg_svo
// and PL-SVO: https://github.com/rubengooj/pl-svo
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

/**
 * modified file in ORB-SLAM (https://github.com/raulmur/ORB_SLAM)
 */

#pragma once

#include <mutex>
#include <opencv2/core/core.hpp>
#include <sophus/se3.h>
#include <pangolin/pangolin.h>

namespace hso {
class Frame;
class Map;
class FrameHandlerMono; }


namespace hso {

class Viewer
{
public:
    Viewer(hso::FrameHandlerMono* vo);
    void run();
    bool CheckFinish();
    void DrawKeyFrames(const bool bDrawKF);
    void DrawMapRegionPoints();
    void DrawMapSeeds();
    void DrawConstraints();

    void GetCurrentOpenGLCameraMatrix(pangolin::OpenGlMatrix &M);
    void DrawCurrentCamera(pangolin::OpenGlMatrix &Twc);

private:
    hso::FrameHandlerMono* _vo;

    std::mutex mMutexCurrentPose;
    std::vector< Sophus::SE3 > _pos;
    Sophus::SE3  _CurrentPoseTwc ;
    int _drawedframeID=0;

    void SetFinish();
    bool mbFinished;
    std::mutex mMutexFinish;

    float mKeyFrameSize;
    float mKeyFrameLineWidth;
    float mGraphLineWidth;
    float mPointSize;
    float mCameraSize;
    float mCameraLineWidth;

    float mViewpointX, mViewpointY, mViewpointZ, mViewpointF;
};
}