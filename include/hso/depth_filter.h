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

#ifndef HSO_DEPTH_FILTER_H_
#define HSO_DEPTH_FILTER_H_

#include <queue>
#include <boost/thread.hpp>
#include <boost/function.hpp>
#include <hso/global.h>
#include <hso/feature_detection.h>
#include <hso/reprojector.h>
#include <hso/matcher.h>
#include <hso/IndexThreadReduce.h>

#include "hso/vikit/performance_monitor.h"

namespace hso {

class Frame;
class Feature;
class Point;

/// A seed is a probabilistic depth estimate for a single pixel.
struct Seed
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    static int batch_counter;
    static int seed_counter;
    int batch_id;                //!< Batch id is the id of the keyframe for which the seed was created.
    int id;                      //!< Seed ID, only used for visualization.
    Feature* ftr;                //!< Feature in the keyframe for which the depth should be computed.
    float a;                     //!< a of Beta distribution: When high, probability of inlier is large.
    float b;                     //!< b of Beta distribution: When high, probability of outlier is large.
    float mu;                    //!< Mean of normal distribution.
    float z_range;               //!< Max range of the possible depth.
    float sigma2;                //!< Variance of normal distribution.
    Matrix2d patch_cov;          //!< Patch covariance in reference image.
    vector<FramePtr> pre_frames; //!< Pre frames before the seed is initialized.

    bool isValid;

    Vector2i eplStart;
    Vector2i eplEnd;

    bool haveReprojected;
    Point* temp;

    std::vector<float> vec_distance;
    std::vector<float> vec_sigma2;

    std::vector<FramePtr> optFrames_P;
    std::vector<FramePtr> optFrames_A;
    float opt_id;

    FramePtr last_update_frame;
    Vector2d last_matched_px;
    int last_matched_level;

    float converge_thresh;


    // debug
    bool is_update;

    Seed(Feature* ftr, float depth_mean, float depth_min, float converge_threshold=200);
};

/// Depth filter implements the Bayesian Update proposed in:
/// "Video-based, Real-Time Multi View Stereo" by G. Vogiatzis and C. Hernández.
/// In Image and Vision Computing, 29(7):434-441, 2011.
///
/// The class uses a callback mechanism such that it can be used also by other
/// algorithms than nslam and for simplified testing.
class DepthFilter
{
friend class hso::Reprojector;

public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef boost::unique_lock<boost::mutex> lock_t;
  typedef boost::function<void ( Point*, double )> callback_t;

  /// Depth-filter config parameters
  struct Options
  {
    bool check_ftr_angle;                       //!< gradient features are only updated if the epipolar line is orthogonal to the gradient.
    bool epi_search_1d;                         //!< restrict Gauss Newton in the epipolar search to the epipolar line.
    bool verbose;                               //!< display output.
    bool use_photometric_disparity_error;       //!< use photometric disparity error instead of 1px error in tau computation.
    int max_n_kfs;                              //!< maximum number of keyframes for which we maintain seeds.
    double sigma_i_sq;                          //!< image noise.
    double seed_convergence_sigma2_thresh;      //!< threshold on depth uncertainty for convergence.
    Options()
    : check_ftr_angle(false),
      epi_search_1d(false),
      verbose(true),
      use_photometric_disparity_error(false),
      max_n_kfs(3),
      sigma_i_sq(5e-4),
      seed_convergence_sigma2_thresh(200.0)
    {}
  } options_;

  boost::mutex stats_mut_;
  RunningStats* runningStats_;
  int n_update_last_;



  FramePtr active_frame_;
  

  boost::mutex mean_mutex_;
  size_t nMeanConvergeFrame_ = 6;



  DepthFilter(feature_detection::FeatureExtractor* featureExtractor, callback_t seed_converged_cb);

  virtual ~DepthFilter();

  /// Start this thread when seed updating should be in a parallel thread.
  void startThread();

  /// Stop the parallel thread that is running.
  void stopThread();

  /// Add frame to the queue to be processed.
  void addFrame(FramePtr frame);

  /// Add new keyframe to the queue
  void addKeyframe(FramePtr frame, double depth_mean, double depth_min, float converge_thresh=200.0);

  /// Remove all seeds which are initialized from the specified keyframe. This
  /// function is used to make sure that no seeds points to a non-existent frame
  /// when a frame is removed from the map.
  void removeKeyframe(FramePtr frame);

  /// If the map is reset, call this function such that we don't have pointers
  /// to old frames.
  void reset();

  /// Returns a copy of the seeds belonging to frame. Thread-safe.
  /// Can be used to compute the Next-Best-View in parallel.
  /// IMPORTANT! Make sure you hold a valid reference counting pointer to frame
  /// so it is not being deleted while you use it.
  void getSeedsCopy(const FramePtr& frame, std::list<Seed>& seeds);

  /// Return a reference to the seeds. This is NOT THREAD SAFE!
  std::list<Seed>& getSeeds() { return seeds_; }

  /// Bayes update of the seed, x is the measurement, tau2 the measurement uncertainty
  static void updateSeed(const float x, const float tau2, Seed* seed);

  /// Compute the uncertainty of the measurement.
  static double computeTau(
      const SE3& T_ref_cur,
      const Vector3d& f,
      const double z,
      const double px_error_angle);


  void directPromotionFeature();

protected:
  boost::mutex detector_mut_;

  feature_detection::FeatureExtractor* featureExtractor_;

  callback_t seed_converged_cb_;

  std::list<Seed> seeds_;
  boost::mutex seeds_mut_;

  boost::mutex m_converge_seed_mut;
  std::list<Seed> m_converge_seed;


  bool seeds_updating_halt_;            //!< Set this value to true when seeds updating should be interrupted.
  boost::thread* thread_;
  std::queue<FramePtr> frame_queue_;
  boost::mutex frame_queue_mut_;
  boost::condition_variable frame_queue_cond_;

  FramePtr new_keyframe_;               //!< Next keyframe to extract new seeds.
  bool new_keyframe_set_;               //!< Do we have a new keyframe to process?.
  double new_keyframe_min_depth_;       //!< Minimum depth in the new keyframe. Used for range in new seeds.
  double new_keyframe_mean_depth_;      //!< Maximum depth in the new keyframe. Used for range in new seeds.
  hso::PerformanceMonitor permon_;      //!< Separate performance monitor since the DepthFilter runs in a parallel thread.
  Matcher matcher_; 

  /// Threshold for the uncertainty of the seed. If seed's sigma2 is thresh
  /// smaller than the inital sigma, it is considered as converged.
  /// Default value is 200. If seeds should converge quicker, set it to 50 or
  /// if you want very precise 3d points, set it higher.
  float convergence_sigma2_thresh_ = 200.0;


  lsd_slam::IndexThreadReduce* threadReducer_;
  double px_error_angle_;
  

  /// Initialize new seeds from a frame.
  void initializeSeeds(FramePtr frame);

  /// Update all seeds with a new measurement frame.
  virtual void updateSeeds(FramePtr frame);

  /// When a new keyframe arrives, the frame queue should be cleared.
  void clearFrameQueue();

  /// A thread that is continuously updating the seeds.
  void updateSeedsLoop();

  void observeDepth();
  void observeDepthRow(int yMin, int yMax, RunningStats* stats);


  void observeDepthWithPreviousFrameOnce(std::list<Seed>::iterator& ite);

private:

  std::vector< list<FramePtr> > frame_prior_;

  size_t n_pre_update_, n_pre_try_;

  size_t nPonits, nSkipFrame;
  vector<size_t> m_v_n_converge;

  bool activatePoint(Seed& seed, bool& isValid);
  void seedOptimizer(Seed& seed, 
    const vector<pair<FramePtr, Vector2d> >& targets, 
    const vector<Vector2d>& normals);
};

} // namespace hso

#endif // HSO_DEPTH_FILTER_H_
