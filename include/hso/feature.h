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

#ifndef HSO_FEATURE_H_
#define HSO_FEATURE_H_

#include <hso/frame.h>

namespace hso {

/// A salient image region that is tracked across frames.
struct Feature
{
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  enum FeatureType {CORNER, EDGELET, GRADIENT};

  FeatureType type;     //!< Type can be corner or edgelet.
  Frame* frame;         //!< Pointer to frame in which the feature was detected.
  Vector2d px;          //!< Coordinates in pixels on pyramid level 0.
  Vector3d f;           //!< Unit-bearing vector of the feature.
  int level;            //!< Image pyramid level where feature was extracted.
  Point* point;         //!< Pointer to 3D point which corresponds to the feature.
  Vector2d grad;        //!< Dominant gradient direction for edglets, normalized.

  // used in photometric calibration thread, 这些变量不应该在前端被使用
  vector<double> outputs;
  vector<double> radiances;
  vector<double> outputs_grad;
  vector<double> rad_mean;
  Feature* m_prev_feature = NULL;  
  Feature* m_next_feature = NULL;
  bool m_added = false;  // Flag, used in photomatric calibration
  // bool m_is_seed = false;
  bool m_non_point = false;
  Matrix2d rotate_plane;


  // corner
  Feature(Frame* _frame, const Vector2d& _px, int _level) :
    type(CORNER),
    frame(_frame),
    px(_px),
    f(frame->cam_->cam2world(px)),
    level(_level),
    point(NULL),
    grad(1.0,0.0)
  {
  }

  Feature(Frame* _frame, const Vector2d& _px, const Vector3d& _f, int _level) :
    type(CORNER),
    frame(_frame),
    px(_px),
    f(_f),
    level(_level),
    point(NULL),
    grad(1.0,0.0)
  {
  }

  Feature(Frame* _frame, Point* _point, const Vector2d& _px, const Vector3d& _f, int _level) :
    type(CORNER),
    frame(_frame),
    px(_px),
    f(_f),
    level(_level),
    point(_point),
    grad(1.0,0.0)
  {
  }

  // edgelet
  Feature(Frame* _frame, const Vector2d& _px, const Vector2d& _grad, int _level) :
    type(EDGELET),
    frame(_frame),
    px(_px),
    f(frame->cam_->cam2world(px)),
    level(_level),
    point(NULL),
    grad(_grad)
  {
  }

  Feature(Frame* _frame, Point* _point, const Vector2d& _px, const Vector3d& _f, const Vector2d& _grad, int _level) :
    type(EDGELET),
    frame(_frame),
    px(_px),
    f(_f),
    level(_level),
    point(_point),
    grad(_grad)
  {
  }

  // gradient
  Feature(Frame* _frame, const Vector2d& _px, int _level, FeatureType _type) :
    type(_type),
    frame(_frame),
    px(_px),
    f(frame->cam_->cam2world(px)),
    level(_level),
    point(NULL),
    grad(1.0,0.0)
  {
  }

  Feature(Frame* _frame, Point* _point, const Vector2d& _px, int _level, FeatureType _type) :
    type(_type),
    frame(_frame),
    px(_px),
    f(frame->cam_->cam2world(px)),
    level(_level),
    point(_point),
    grad(1.0,0.0)
  {
  }

};

} // namespace hso

#endif // HSO_FEATURE_H_
