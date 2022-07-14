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

#ifndef HSO_REPROJECTION_H_
#define HSO_REPROJECTION_H_

#include <hso/global.h>
#include <hso/matcher.h>
#include "hso/camera.h"

// namespace vk {
// class AbstractCamera;
// }

namespace hso {

class Map;
class Point;
class DepthFilter;
struct Seed;


/// Project points from the map into the image and find the corresponding
/// feature (corner). We don't search a match for every point but only for one
/// point per cell. Thereby, we achieve a homogeneously distributed set of
/// matched features and at the same time we can save processing time by not
/// projecting all points.
class Reprojector
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  /// Reprojector config parameters
  struct Options {
    size_t max_n_kfs;   //!< max number of keyframes to reproject from
    bool find_match_direct;
    bool reproject_unconverged_seeds;
    float reproject_seed_thresh;
    Options()
    : max_n_kfs(10),
    find_match_direct(true),
    reproject_unconverged_seeds(true),
    reproject_seed_thresh(86)
    {}
  } options_;

  size_t n_matches_;
  size_t n_trials_;
  size_t n_seeds_;
  size_t n_filters_;

  FramePtr lastFrame_;

  Reprojector(hso::AbstractCamera* cam, Map& map);

  ~Reprojector();

  int caculateGridSize(const int wight, const int height, const int N);

  /// Project points from the map into the image. First finds keyframes with
  /// overlapping field of view and projects only those map-points.
  void reprojectMap(FramePtr frame, std::vector< std::pair<FramePtr,std::size_t> >& overlap_kfs);
  

  DepthFilter* depth_filter_;

private:

  /// A candidate is a point that projects into the image plane and for which we
  /// will search a maching feature in the image.
  struct Candidate {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    Point* pt;       //!< 3D point.
    Vector2d px;     //!< projected 2D pixel location.

    Candidate(Point* pt, Vector2d& px) :  pt(pt), px(px) 
    {}
  };
  typedef std::list<Candidate > Cell;
  typedef std::vector<Cell*> CandidateGrid;

  struct SeedCandidate{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    Seed& seed;
    Vector2d px;
    list< Seed, aligned_allocator<Seed> >::iterator index;
    SeedCandidate(Seed& _seed, Vector2d _uv, 
      list< Seed, aligned_allocator<Seed> >::iterator _i):seed(_seed), px(_uv), index(_i)
    {}
  };
  typedef std::list<SeedCandidate> Sell;
  typedef std::vector<Sell*> SeedGrid;

  /// The grid stores a set of candidate matches. For every grid cell we try to find one match.
  struct Grid
  {
    CandidateGrid cells;
    SeedGrid seeds;
    vector<int> cell_order;
    int cell_size;
    int grid_n_cols;
    int grid_n_rows;

    int cell_size_w;
    int cell_size_h;
  };

  Grid grid_;
  Matcher matcher_;
  Map& map_;

  static bool pointQualityComparator(Candidate& lhs, Candidate& rhs);
  static bool seedComparator(SeedCandidate& lhs, SeedCandidate& rhs);

  void initializeGrid(hso::AbstractCamera* cam);
  void resetGrid();
  bool reprojectCell(Cell& cell, FramePtr frame, bool is_2nd = false, bool is_3rd = false);
  bool reprojectorSeeds(Sell& sell, FramePtr frame);

  bool reprojectPoint(FramePtr frame, Point* point, vector< pair<Vector2d, Point*> >& cells);

  bool reprojectorSeed(FramePtr frame, Seed& seed, list< Seed, aligned_allocator<Seed> >::iterator index);

  void reprojectCellAll(vector< pair<Vector2d, Point*> >& cell, FramePtr frame);


  // debug
  size_t sum_seed_;
  size_t sum_temp_;

  size_t nFeatures_;
};

} // namespace hso

#endif // HSO_REPROJECTION_H_
