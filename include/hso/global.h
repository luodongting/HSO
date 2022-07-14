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

#ifndef HSO_GLOBAL_H_
#define HSO_GLOBAL_H_

#include <list>
#include <vector>
#include <string>
#include <stdint.h>
#include <stdio.h>
#include <math.h>

#include <Eigen/Core>
#include <opencv2/opencv.hpp>
#include <sophus/se3.h>
#include <boost/shared_ptr.hpp>
#include <Eigen/StdVector>

#include "hso/vikit/performance_monitor.h"

//the following are UBUNTU/LINUX ONLY terminal color codes.
#define RESET "\033[0m"
#define BLACK "\033[30m" /* Black */
#define RED "\033[31m" /* Red */
#define GREEN "\033[32m" /* Green */
#define YELLOW "\033[33m" /* Yellow */
#define BLUE "\033[34m" /* Blue */
#define MAGENTA "\033[35m" /* Magenta */
#define CYAN "\033[36m" /* Cyan */
#define WHITE "\033[37m" /* White */
#define BOLDBLACK "\033[1m\033[30m" /* Bold Black */
#define BOLDRED "\033[1m\033[31m" /* Bold Red */
#define BOLDGREEN "\033[1m\033[32m" /* Bold Green */
#define BOLDYELLOW "\033[1m\033[33m" /* Bold Yellow */
#define BOLDBLUE "\033[1m\033[34m" /* Bold Blue */
#define BOLDMAGENTA "\033[1m\033[35m" /* Bold Magenta */
#define BOLDCYAN "\033[1m\033[36m" /* Bold Cyan */
#define BOLDWHITE "\033[1m\033[37m" /* Bold White */

#ifndef RPG_HSO_VIKIT_IS_VECTOR_SPECIALIZED //Guard for rpg_vikit
#define RPG_HSO_VIKIT_IS_VECTOR_SPECIALIZED
EIGEN_DEFINE_STL_VECTOR_SPECIALIZATION(Eigen::Vector3d)
EIGEN_DEFINE_STL_VECTOR_SPECIALIZATION(Eigen::Vector2d)
#endif

// #ifdef HSO_USE_ROS
//   #include <ros/console.h>
//   #define HSO_DEBUG_STREAM(x) ROS_DEBUG_STREAM(x)
//   #define HSO_INFO_STREAM(x) ROS_INFO_STREAM(x)
//   #define HSO_WARN_STREAM(x) ROS_WARN_STREAM(x)
//   #define HSO_WARN_STREAM_THROTTLE(rate, x) ROS_WARN_STREAM_THROTTLE(rate, x)
//   #define HSO_ERROR_STREAM(x) ROS_ERROR_STREAM(x)
// #else
  #define HSO_INFO_STREAM(x) std::cerr<<"\033[0;0m[INFO] "<<x<<"\033[0;0m"<<std::endl;
  #ifdef HSO_DEBUG_OUTPUT
    #define HSO_DEBUG_STREAM(x) HSO_INFO_STREAM(x)
  #else
    #define HSO_DEBUG_STREAM(x)
  #endif
  #define HSO_WARN_STREAM(x) std::cerr<<"\033[0;33m[WARN] "<<x<<"\033[0;0m"<<std::endl;
  #define HSO_ERROR_STREAM(x) std::cerr<<"\033[1;31m[ERROR] "<<x<<"\033[0;0m"<<std::endl;
  #include <chrono> // Adapted from rosconsole. Copyright (c) 2008, Willow Garage, Inc.
  #define HSO_WARN_STREAM_THROTTLE(rate, x) \
    do { \
      static double __log_stream_throttle__last_hit__ = 0.0; \
      std::chrono::time_point<std::chrono::system_clock> __log_stream_throttle__now__ = \
      std::chrono::system_clock::now(); \
      if (__log_stream_throttle__last_hit__ + rate <= \
          std::chrono::duration_cast<std::chrono::seconds>( \
          __log_stream_throttle__now__.time_since_epoch()).count()) { \
        __log_stream_throttle__last_hit__ = \
        std::chrono::duration_cast<std::chrono::seconds>( \
        __log_stream_throttle__now__.time_since_epoch()).count(); \
        HSO_WARN_STREAM(x); \
      } \
    } while(0)
// #endif

namespace hso
{
  using namespace Eigen;
  using namespace Sophus;

  const double EPS = 0.0000000001;
  const double PI = 3.14159265;

#ifdef HSO_TRACE
  extern hso::PerformanceMonitor* g_permon;
  #define HSO_LOG(value) g_permon->log(std::string((#value)),(value))
  #define HSO_LOG2(value1, value2) HSO_LOG(value1); HSO_LOG(value2)
  #define HSO_LOG3(value1, value2, value3) HSO_LOG2(value1, value2); HSO_LOG(value3)
  #define HSO_LOG4(value1, value2, value3, value4) HSO_LOG2(value1, value2); HSO_LOG2(value3, value4)
  #define HSO_START_TIMER(name) g_permon->startTimer((name))
  #define HSO_STOP_TIMER(name) g_permon->stopTimer((name))
#else
  #define HSO_LOG(v)
  #define HSO_LOG2(v1, v2)
  #define HSO_LOG3(v1, v2, v3)
  #define HSO_LOG4(v1, v2, v3, v4)
  #define HSO_START_TIMER(name)
  #define HSO_STOP_TIMER(name)
#endif

  
  class Frame;
  typedef boost::shared_ptr<Frame> FramePtr;

} // namespace hso

#endif // HSO_GLOBAL_H_
