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

#ifndef HSO_FEATURE_DETECTION_H_
#define HSO_FEATURE_DETECTION_H_

#include <hso/global.h>
#include <hso/frame.h>
#include <fast/fast.h>

namespace hso {

/// Implementation of various feature detectors.
namespace feature_detection {

/// Temporary container used for corner detection. Features are initialized from these.
struct Corner
{
  int x;        //!< x-coordinate of corner in the image.
  int y;        //!< y-coordinate of corner in the image.
  int level;    //!< pyramid level of the corner.
  float score;  //!< shi-tomasi score of the corner.
  float angle;  //!< for gradient-features: dominant gradient angle.
  Corner(int x, int y, float score, int level, float angle) :
    x(x), y(y), level(level), score(score), angle(angle)
  {}
};
typedef vector<Corner> Corners;

struct EdgeLet
{
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  Vector2i start;
  Vector2i end;
  Vector2i mid;
  Vector2d grad;
  vector<cv::Point2i> line;
  int level;
  float score;
  bool is_set;

  EdgeLet( Vector2i start_, Vector2i end_, int level_, float score_, bool is_set_ ):
    start(start_), end(end_), level(level_), score(score_), is_set(is_set_) 
  {}
};
using EdgeLets = std::vector<EdgeLet>;

struct Gradient
{
  int x;
  int y;
  float delta;
  int level;
  bool is_set;
  Gradient( int x_, int y_, float delta_, int level_ ):
    x(x_), y(y_), delta(delta_), level(level_), is_set(false)
  {}
};
using Gradients = std::vector<Gradient>;

/// All detectors should derive from this abstract class.
class AbstractDetector
{
public:
  AbstractDetector(
      const int img_width,
      const int img_height,
      const int cell_size,
      const int n_pyr_levels);

  virtual ~AbstractDetector() {};

  virtual void detect(
      Frame* frame,
      const ImgPyr& img_pyr,
      const float& detection_threshold,
      Features& fts) = 0;

  /// Flag the grid cell as occupied
  void setGridOccpuancy(const Vector2d& px);

  /// Set grid cells of existing features as occupied
  void setExistingFeatures(const Features& fts);

protected:

  static const int border_ = 8; //!< no feature should be within 8px of border.
  const int cell_size_;
  const int n_pyr_levels_;
  const int grid_n_cols_;
  const int grid_n_rows_;
  vector<bool> grid_occupancy_;

  void resetGrid();

  inline int getCellIndex(int x, int y, int level)
  {
    const int scale = (1<<level);
    return (scale*y)/cell_size_*grid_n_cols_ + (scale*x)/cell_size_;
  }
};
typedef boost::shared_ptr<AbstractDetector> DetectorPtr;

/// FAST detector by Edward Rosten.
class FastDetector : public AbstractDetector
{
public:
  FastDetector(
      const int img_width,
      const int img_height,
      const int cell_size,
      const int n_pyr_levels);

  virtual ~FastDetector() {}

  virtual void detect(
      Frame* frame,
      const ImgPyr& img_pyr,
      const float& detection_threshold,
      Features& fts);
};

class EdgeletDetector : public AbstractDetector
{
public:
  EdgeletDetector(
      const int img_width,
      const int img_height,
      const int cell_size,
      const int n_pyr_levels);

  virtual ~EdgeletDetector() {}

  virtual void detect(
      Frame* frame,
      const ImgPyr& img_pyr,
      const float& detection_threshold,
      Features& fts);
};

class GradientDetector : public AbstractDetector
{
public:
  GradientDetector(
      const int img_width,
      const int img_height,
      const int cell_size,
      const int n_pyr_levels);

  virtual ~GradientDetector() {}

  virtual void detect(
      Frame* frame,
      const ImgPyr& img_pyr,
      const float& detection_threshold,
      Features& fts);
};


enum FeatureSpecies
{
    kCornerHigh,
    kEdgeLet,
    kGrad,
    kOccur
};

struct KeyPoint
{   
    float x;
    float y;
    float response;
    int level;
    FeatureSpecies species;
    int gx;
    int gy;

    KeyPoint(float _x, float _y, float _response, int _level, FeatureSpecies _species): 
        x(_x), y(_y), response(_response), level(_level), species(_species)
    {

    }

    KeyPoint(): x(0), y(0), response(0), level(0), species(kCornerHigh)
    {

    }
};

// Modified code in ORB-SLAM (https://github.com/raulmur/ORB_SLAM)
class ExtractorNode
{
public:

    ExtractorNode():bNoMore(false){}

    void DivideNode(ExtractorNode &n1, ExtractorNode &n2, ExtractorNode &n3, ExtractorNode &n4)
    {
        const int halfX = ceil(static_cast<float>(UR.x-UL.x)/2);
        const int halfY = ceil(static_cast<float>(BR.y-UL.y)/2);

        //Define boundaries of childs
        n1.UL = UL;
        n1.UR = cv::Point2i(UL.x+halfX,UL.y);
        n1.BL = cv::Point2i(UL.x,UL.y+halfY);
        n1.BR = cv::Point2i(UL.x+halfX,UL.y+halfY);
        n1.vKeys.reserve(vKeys.size());

        n2.UL = n1.UR;
        n2.UR = UR;
        n2.BL = n1.BR;
        n2.BR = cv::Point2i(UR.x,UL.y+halfY);
        n2.vKeys.reserve(vKeys.size());

        n3.UL = n1.BL;
        n3.UR = n1.BR;
        n3.BL = BL;
        n3.BR = cv::Point2i(n1.BR.x,BL.y);
        n3.vKeys.reserve(vKeys.size());

        n4.UL = n3.UR;
        n4.UR = n2.BR;
        n4.BL = n3.BR;
        n4.BR = BR;
        n4.vKeys.reserve(vKeys.size());

        for(size_t i=0;i<vKeys.size();i++)
        {
            // const fast::fast_xy &kp = vKeys[i].first;
            const float u = vKeys[i].x;
            const float v = vKeys[i].y;
            if(u<n1.UR.x)
            {
                if(v<n1.BR.y)
                    n1.vKeys.push_back(vKeys[i]);
                else
                    n3.vKeys.push_back(vKeys[i]);
            }
            else if(v<n1.BR.y)
                n2.vKeys.push_back(vKeys[i]);
            else
                n4.vKeys.push_back(vKeys[i]);
        }

        if(n1.vKeys.size()==1)
            n1.bNoMore = true;
        if(n2.vKeys.size()==1)
            n2.bNoMore = true;
        if(n3.vKeys.size()==1)
            n3.bNoMore = true;
        if(n4.vKeys.size()==1)
            n4.bNoMore = true;
    }

    std::vector<KeyPoint> vKeys;
    cv::Point2i UL, UR, BL, BR;
    std::list<ExtractorNode>::iterator lit; //self ptr
    bool bNoMore;
};

class FeatureExtractor
{
public:

    FeatureExtractor(const int width, const int height, const int cellSize, const int levels, bool isInit=false);

    void detect(Frame* frame, const float initThresh, const float minThresh, Features& fts, Frame* last_frame=NULL);

    void setGridOccpuancy(const Vector2d& px, Feature* occurFeature);

    void setExistingFeatures(const Features& fts);

    void resetGrid();

    inline int getCellIndex(int x, int y, int level)
    {
        // int gridPrySize = gridSize_/(1<<level);
        return static_cast<int>(y/vGrids_[level]*vGridCols_[level] + x/vGridRows_[level]);
    }

protected:

    void fastDetect(const ImgPyr& img_pyr);
    void fastDetectMT(const ImgPyr& img_pyr);
    void fastDetectST(const cv::Mat& imageLevel, const int Level);

    void gradDetect(const ImgPyr& img_pyr);
    void gradDetectMT(const ImgPyr& img_pyr);
    void gradDetectST(const cv::Mat& imageLevel, const int Level);

    void edgeLetDetectMT(const ImgPyr& img_pyr);
    void edgeLetDetectST(const cv::Mat& imageLevel, const int Level);


    void fillingHole(const cv::Mat& imageLevel, const int Level);


    vector<KeyPoint> computeKeyPointsOctTree(
        const vector<KeyPoint>& toDistributeKeys, 
        const int &minX, const int &maxX, const int &minY, const int &maxY, const int &level);

    void findEpiHole();
    bool edgeletFilter(int u_level, int v_level, short gx, short gy, int level, double& angle);

    int width_;
    std::vector<int> vecWidth_;

    int height_;
    std::vector<int> vecHeight_;

    int cellSize_;
    int nLevels_;

    int nFeatures_;
    int extFeatures_;
    int needFeatures_;

    int initThresh_;
    int minThresh_;

    int nCols_;
    int nRows_;

    bool m_egde_filter;
    Vector2d epi_hole;

    Frame* frame_;
    Frame* m_last_frame=NULL;

    // All Features including existing and detected
    std::vector<KeyPoint> allFeturesToDistribute_;
    std::vector<vector<KeyPoint> > featurePerLevel_;
    std::vector<vector<KeyPoint> > cornerPerLevel_;
    std::vector<vector<KeyPoint> > gradPerLevel_;

    static const int gridSize_ = 8;
    std::vector<int> vGrids_;
    std::vector<int> vGridCols_;
    std::vector<int> vGridRows_;
    std::vector<vector<bool> > haveFeatures_;

    std::vector<KeyPoint> resultFeatures_;

    bool isInit_;
};
} // namespace feature_detection
} // namespace hso

#endif // HSO_FEATURE_DETECTION_H_
