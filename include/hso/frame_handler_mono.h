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

#ifndef HSO_FRAME_HANDLER_H_
#define HSO_FRAME_HANDLER_H_

#include <set>
#include <boost/thread.hpp>
#include <hso/frame_handler_base.h>
#include <hso/reprojector.h>
#include <hso/initialization.h>

// #include "hso/PhotomatricCalibration.h"
#include "hso/camera.h"

namespace hso {

class FrameHandlerMono : public FrameHandlerBase
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  
  FrameHandlerMono(hso::AbstractCamera* cam, bool _use_pc=false);
  virtual ~FrameHandlerMono();

  /// Provide an image.
  void addImage(const cv::Mat& img, double timestamp, string* timestamp_s=NULL);

  /// Set the first frame (used for synthetic datasets in benchmark node)
  void setFirstFrame(const FramePtr& first_frame);

  /// Get the last frame that has been processed.
  FramePtr lastFrame() { return last_frame_; }

  /// Get the set of spatially closest keyframes of the last frame.
  const set<FramePtr>& coreKeyframes() { return core_kfs_; }

  /// Return the feature track to visualize the KLT tracking during initialization.
  const vector<cv::Point2f>& initFeatureTrackRefPx() const { return klt_homography_init_.px_ref_; }
  const vector<cv::Point2f>& initFeatureTrackCurPx() const { return klt_homography_init_.px_cur_; }

  /// Access the depth filter.
  DepthFilter* depthFilter() const { return depth_filter_; }

  /// An external place recognition module may know where to relocalize.
  bool relocalizeFrameAtPose(const int keyframe_id,
                             const SE3& T_kf_f,
                             const cv::Mat& img,
                             const double timestamp);

public:
  hso::AbstractCamera* cam_;                    //!< Camera model, can be ATAN, Pinhole or Ocam (see vikit).

  Reprojector reprojector_;                     //!< Projects points from other keyframes into the current frame

  FramePtr new_frame_;                          //!< Current frame.
  FramePtr last_frame_;                         //!< Last frame, not necessarily a keyframe.
  FramePtr firstFrame_;

  set<FramePtr> core_kfs_;                      //!< Keyframes in the closer neighbourhood.
  vector< pair<FramePtr,size_t> > overlap_kfs_; //!< All keyframes with overlapping field of view. the paired number specifies how many common mappoints are observed TODO: why vector!?

  initialization::KltHomographyInit klt_homography_init_; //!< Used to estimate pose of the first two keyframes by estimating a homography.

  DepthFilter* depth_filter_;                   //!< Depth estimation algorithm runs in a parallel thread and is used to initialize new 3D points.

  SE3 motionModel_;

  bool afterInit_ = false;

  // PhotomatricCalibration* m_photomatric_calib;


  vector<Frame*> vpLocalKeyFrames; 

  

  /// Initialize the visual odometry algorithm.
  virtual void initialize();

  /// Processes the first frame and sets it as a keyframe.
  virtual UpdateResult processFirstFrame();

  /// Processes all frames after the first frame until a keyframe is selected.
  virtual UpdateResult processSecondFrame();

  /// Processes all frames after the first two keyframes.
  virtual UpdateResult processFrame();

  /// Try relocalizing the frame at relative position to provided keyframe.
  virtual UpdateResult relocalizeFrame(const SE3& T_cur_ref, FramePtr ref_keyframe);

  /// Reset the frame handler. Implement in derived class.
  virtual void resetAll();

  /// Keyframe selection criterion.
  virtual bool needNewKf(
  const double& scene_depth_mean, 
  const size_t& num_observations);

  void setCoreKfs(size_t n_closest);

  bool kfOverView();




  set<Frame*> LocalMap_; 
  static bool frameCovisibilityComparator(pair<int, Frame*>& lhs, pair<int, Frame*>& rhs);
  static bool frameCovisibilityComparatorF(pair<float, Frame*>& lhs, pair<float, Frame*>& rhs);

  void createCovisibilityGraph(FramePtr currentFrame, size_t n_closest, bool is_keyframe);

  // void prepareForPhotomatricCalibration();

  // void calcMotionModel();



};

} // namespace hso

#endif // HSO_FRAME_HANDLER_H_
