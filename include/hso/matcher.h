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

#ifndef HSO_MATCHER_H_
#define HSO_MATCHER_H_

#include <hso/global.h>

namespace hso {
  class AbstractCamera;
  namespace patch_score {
    template<int HALF_PATCH_SIZE> class ZMSSD;
  }
}

namespace hso {

class Point;
class Frame;
class Feature;
class Reprojector;
struct Seed;

/// Warp a patch from the reference view to the current view.
namespace warp {

void getWarpMatrixAffine(
    const hso::AbstractCamera& cam_ref,
    const hso::AbstractCamera& cam_cur,
    const Vector2d& px_ref,
    const Vector3d& f_ref,
    const double depth_ref,
    const SE3& T_cur_ref,
    const int level_ref,
    Matrix2d& A_cur_ref);

int getBestSearchLevel(
    const Matrix2d& A_cur_ref,
    const int max_level);

void warpAffine(
    const Matrix2d& A_cur_ref,
    const cv::Mat& img_ref,
    const Vector2d& px_ref,
    const int level_ref,
    const int level_cur,
    const int halfpatch_size,
    uint8_t* patch);

void warpAffine(
    const Matrix2d& A_cur_ref,
    const cv::Mat& img_ref,
    const Vector2d& px_ref,
    const int level_ref,
    const int level_cur,
    const int halfpatch_size,
    float* patch);

void createPatch(
    float* patch,
    const Vector2d& px_scaled,
    const Frame* cur_frame,
    const int halfpatch_size,
    const int level);

void convertPatchFloat2Int(
    float* patch_f,
    uint8_t* patch_i,
    const int patch_size);

void createPatchFromPatchWithBorder(
    uint8_t* patch,
    uint8_t* patch_with_border,
    const int patch_size);

void createPatchFromPatchWithBorder(
    float* patch,
    float* patch_with_border,
    const int patch_size);
} // namespace warp

/// Patch-matcher for reprojection-matching and epipolar search in triangulation.
class Matcher
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  static const int halfpatch_size_ = 4;
  static const int patch_size_ = 8;

  typedef hso::patch_score::ZMSSD<halfpatch_size_> PatchScore;

  struct Options
  {
    bool align_1d;              //!< in epipolar search: align patch 1D along epipolar line
    int align_max_iter;         //!< number of iterations for aligning the feature patches in gauss newton
    double max_epi_length_optim;//!< max length of epipolar line to skip epipolar search and directly go to img align
    size_t max_epi_search_steps;//!< max number of evaluations along epipolar line
    bool subpix_refinement;     //!< do gauss newton feature patch alignment after epipolar search
    bool epi_search_edgelet_filtering;
    double epi_search_edgelet_max_angle;
    Options() :
      align_1d(false),
      align_max_iter(10),
      max_epi_length_optim(2.0),
      max_epi_search_steps(100),
      subpix_refinement(true),
      epi_search_edgelet_filtering(true),
      epi_search_edgelet_max_angle(0.4)
    {}
  } options_;

  uint8_t patch_[patch_size_*patch_size_] __attribute__ ((aligned (16)));
  uint8_t patch_with_border_[(patch_size_+2)*(patch_size_+2)] __attribute__ ((aligned (16)));

  float patch_f_[patch_size_*patch_size_] __attribute__ ((aligned (16)));
  float patch_with_border_f_[(patch_size_+2)*(patch_size_+2)] __attribute__ ((aligned (16)));

  Matrix2d A_cur_ref_;          //!< affine warp matrix
  Vector2d epi_dir_;
  double epi_length_;           //!< length of epipolar line segment in pixels (only used for epipolar search)
  double h_inv_;                //!< hessian of 1d image alignment along epipolar line
  int search_level_;
  bool reject_;
  Feature* ref_ftr_;
  Vector2d px_cur_;

  Matcher() = default;
  ~Matcher() = default;

  /// Find a match by directly applying subpix refinement.
  /// IMPORTANT! This function assumes that px_cur is already set to an estimate that is within ~2-3 pixel of the final result!
  bool findMatchDirect(const Point& pt, Frame& frame, Vector2d& px_cur);

  bool findMatchSeed(const Seed& seed, const Frame& frame, Vector2d& px_cur, float ncc_thresh = 0.6);

  // /// Find a match by searching along the epipolar line without using any features.
  bool findEpipolarMatchDirect(
      const Frame& ref_frame,
      const Frame& cur_frame,
      const Feature& ref_ftr,
      const double d_estimate,
      const double d_min,
      const double d_max,
      double& depth,
      Vector2i& epl_start,
      Vector2i& epl_end,
      bool homoIsValid = false,
      Matrix3d homography = Eigen::Matrix3d::Identity());

  void createPatchFromPatchWithBorder();

  // does the line-stereo seeking.
  // takes a lot of parameters, because they all have been pre-computed before.
  int doLineStereo(
    const Frame& ref_frame, const Frame& cur_frame, const Feature& ref_ftr,
    const double min_idepth, const double prior_idepth, const double max_idepth,
    double& result_depth, Vector2i& EPL_start, Vector2i& EPL_end);

  bool findEpipolarMatchPrevious(
      Frame& ref_frame,
      Frame& cur_frame,
      const Feature& ref_ftr,
      const double d_estimate,
      const double d_min,
      const double d_max,
      double& depth);

  bool checkNCC(float* patch1, float* patch2, float thresh);
  bool checkNormal(const Frame& frame, int level, Vector2d pxLevel, Vector2d normal, float thresh=0.866);

  bool KLTLimited2D(const cv::Mat& targetImg,
  float* hostPatchWithBorder,
  float* hostPatch,
  const int n_iter,
  Vector2d& targetPxEstimate,
  float* targetPatch = NULL,
  bool debugPrint = false);

  bool KLTLimited1D(const cv::Mat& targetImg,
  float* hostPatchWithBorder,
  float* hostPatch,
  const int n_iter,
  Vector2d& targetPxEstimate,
  const Vector2d& direct,
  float* targetPatch = NULL,
  bool debugPrint = false);
};

namespace patch_utils{

inline void patchToMat(
  const uint8_t* const patch_data,
  const size_t patch_width,
  cv::Mat* img)
{
  // CHECK_NOTNULL(img);
  *img = cv::Mat(patch_width, patch_width, CV_8UC1);
  std::memcpy(img->data, patch_data, patch_width*patch_width);
} 

} // namespace patch_utils
} // namespace hso

#endif // HSO_MATCHER_H_
