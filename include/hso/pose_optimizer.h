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

#ifndef HSO_POSE_OPTIMIZER_H_
#define HSO_POSE_OPTIMIZER_H_

#include <hso/global.h>
#include <hso/feature.h>


namespace hso {

using namespace Eigen;
using namespace Sophus;
using namespace std;

typedef Matrix<double,6,6> Matrix6d;
typedef Matrix<double,2,6> Matrix26d;
typedef Matrix<double,6,1> Vector6d;


/// Motion-only bundle adjustment. Minimize the reprojection error of a single frame.
namespace pose_optimizer {

void optimizeGaussNewton(
    const double reproj_thresh,
    const size_t n_iter,
    const bool verbose,
    FramePtr& frame,
    double& estimated_scale,
    double& error_init,
    double& error_final,
    size_t& num_obs);


void optimizeLevenbergMarquardt2nd(
    const double reproj_thresh, const size_t n_iter, const bool verbose,
    FramePtr& frame, double& estimated_scale, double& error_init, double& error_final,
    size_t& num_obs);

void optimizeLevenbergMarquardt3rd(
    const double reproj_thresh, const size_t n_iter, const bool verbose,
    FramePtr& frame, double& estimated_scale, double& error_init, double& error_final,
    size_t& num_obs);

void optimizeLevenbergMarquardtMagnitude(
    const double reproj_thresh, const size_t n_iter, const bool verbose,
    FramePtr& frame, double& estimated_scale, double& error_init, double& error_final,
    size_t& num_obs);


    // distribution
    static int residual_buffer[10000]={0};

} // namespace pose_optimizer
} // namespace hso

#endif // HSO_POSE_OPTIMIZER_H_
