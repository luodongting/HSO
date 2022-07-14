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


#ifndef HSO_INITIALIZATION_H
#define HSO_INITIALIZATION_H

#include <hso/global.h>

namespace hso {

class FrameHandlerMono;

/// Bootstrapping the map from the first two views.
namespace initialization {

enum InitResult { FAILURE, NO_KEYFRAME, SUCCESS };

enum class InitializerType {
    kHomography,       ///< Estimates a plane from the first two views
    kTwoPoint,         ///< Assumes known rotation from IMU and estimates translation
    kFivePoint,        ///< Estimate relative pose of two cameras using 5pt RANSAC
    kOneShot,          ///< Initialize points on a plane with given depth
    kStereo,           ///< Triangulate from two views with known pose
    kArrayGeometric,   ///< Estimate relative pose of two camera arrays, using 17pt RANSAC
    kArrayOptimization ///< Estimate relative pose of two camera arrays using GTSAM
};


/// Tracks features using Lucas-Kanade tracker and then estimates a homography.
class KltHomographyInit {

friend class hso::FrameHandlerMono;

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    FramePtr frame_ref_;
    KltHomographyInit() {};
    ~KltHomographyInit() {};
    InitResult addFirstFrame(FramePtr frame_ref);
    InitResult addSecondFrame(FramePtr frame_ref);
    void reset();

protected:
    vector<cv::Point2f> px_ref_;      //!< keypoints to be tracked in reference frame.
    vector<cv::Point2f> px_cur_;      //!< tracked keypoints in current frame.

    vector<Vector3d> f_ref_;          //!< bearing vectors corresponding to the keypoints in the reference image.
    vector<Vector3d> f_cur_;          //!< bearing vectors corresponding to the keypoints in the current image.

    vector<double> disparities_;      //!< disparity between first and second frame.
    vector<int> inliers_;             //!< inliers after the geometric check (e.g., Homography).
    vector<Vector3d> xyz_in_cur_;     //!< 3D points computed during the geometric check.
    SE3 T_cur_from_ref_;              //!< computed transformation between the first two frames.

    InitializerType init_type_;       //!< initializer method. See options above.
    cv::Mat img_prev_;
    vector<cv::Point2f> px_prev_;
    vector<Vector3d> ftr_type_;
};



/// Detect Fast corners in the image.
void detectFeatures(FramePtr frame, vector<cv::Point2f>& px_vec, vector<Vector3d>& f_vec, vector<Vector3d>& ftr_type);

/// Compute optical flow (Lucas Kanade) for selected keypoints.
void trackKlt(FramePtr frame_ref,
    FramePtr frame_cur,
    vector<cv::Point2f>& px_ref,
    vector<cv::Point2f>& px_cur,
    vector<Vector3d>& f_ref,
    vector<Vector3d>& f_cur,
    vector<double>& disparities,
    cv::Mat& img_prev, 
    vector<cv::Point2f>& px_prev,
    vector<Vector3d>& fts_type);


void computeInitializeMatrix(
    const vector<Vector3d>& f_ref,
    const vector<Vector3d>& f_cur,
    double focal_length,
    double reprojection_threshold,
    vector<int>& inliers,
    vector<Vector3d>& xyz_in_cur,
    SE3& T_cur_from_ref);

double computeP3D(
    const vector<Vector3d>& vBearingRef,
    const vector<Vector3d>& vBearingCur,
    const Matrix3d& R,
    const Vector3d& t,
    const double reproj_thresh,
    double error_multiplier2,
    vector<Vector3d>& vP3D,
    vector<int>& inliers);

bool patchCheck(
    const cv::Mat& imgPre, const cv::Mat& imgCur, const cv::Point2f& pxPre, const cv::Point2f& pxCur);

bool createPatch(const cv::Mat& img, const cv::Point2f& px, float* patch);

bool checkSSD(float* patch1, float* patch2);

Vector3d distancePointOnce(
    const Vector3d pointW, Vector3d bearingRef, Vector3d bearingCur, SE3 T_c_r);

} // namespace initialization
} // namespace hso

#endif // HSO_INITIALIZATION_H
