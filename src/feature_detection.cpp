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

#include <hso/feature_detection.h>
#include <hso/config.h>
#include <hso/feature.h>
#include <fast/fast.h>
#include <boost/timer.hpp>

#include <thread>

#include "hso/vikit/vision.h"

using namespace cv;

namespace hso {
namespace feature_detection {

AbstractDetector::AbstractDetector(
    const int img_width,
    const int img_height,
    const int cell_size,
    const int n_pyr_levels) :
        cell_size_(cell_size),
        n_pyr_levels_(n_pyr_levels),
        grid_n_cols_(ceil(static_cast<double>(img_width)/cell_size_)),
        grid_n_rows_(ceil(static_cast<double>(img_height)/cell_size_)),
        grid_occupancy_(grid_n_cols_*grid_n_rows_, false)
{}

void AbstractDetector::resetGrid()
{
  std::fill(grid_occupancy_.begin(), grid_occupancy_.end(), false);
}

void AbstractDetector::setExistingFeatures(const Features& fts)
{
  std::for_each(fts.begin(), fts.end(), [&](Feature* i){
    grid_occupancy_.at(
        static_cast<int>(i->px[1]/cell_size_)*grid_n_cols_
        + static_cast<int>(i->px[0]/cell_size_)) = true;
  });
}

void AbstractDetector::setGridOccpuancy(const Vector2d& px)
{
  grid_occupancy_.at(
      static_cast<int>(px[1]/cell_size_)*grid_n_cols_
    + static_cast<int>(px[0]/cell_size_)) = true;
}

FastDetector::FastDetector(
    const int img_width,
    const int img_height,
    const int cell_size,
    const int n_pyr_levels) :
        AbstractDetector(img_width, img_height, cell_size, n_pyr_levels)
{}

void FastDetector::detect(
    Frame* frame,
    const ImgPyr& img_pyr,
    const float& detection_threshold,
    Features& fts)
{
  Corners corners(grid_n_cols_*grid_n_rows_, Corner(0,0,detection_threshold,0,0.0f));
  for(int L=0; L<n_pyr_levels_; ++L)
  {
    const int scale = (1<<L);
    vector<fast::fast_xy> fast_corners;
    fast::fast_corner_detect_plain_8(
      (fast::fast_byte*) img_pyr[L].data, img_pyr[L].cols, img_pyr[L].rows, img_pyr[L].cols, detection_threshold, fast_corners);
      // fast::fast_corner_detect_9_sse2(
      // (fast::fast_byte*) img_pyr[L].data, img_pyr[L].cols,
      // img_pyr[L].rows, img_pyr[L].cols, 20, fast_corners);
// #if __SSE2__
      // fast::fast_corner_detect_10_sse2(
      //     (fast::fast_byte*) img_pyr[L].data, img_pyr[L].cols,
      //     img_pyr[L].rows, img_pyr[L].cols, detection_threshold, fast_corners);

    // fast::fast_corner_detect_plain_12(
    //     (fast::fast_byte*) img_pyr[L].data, img_pyr[L].cols,
    //     img_pyr[L].rows, img_pyr[L].cols, 20, fast_corners);
// #elif HAVE_FAST_NEON
//       fast::fast_corner_detect_9_neon(
//           (fast::fast_byte*) img_pyr[L].data, img_pyr[L].cols,
//           img_pyr[L].rows, img_pyr[L].cols, 20, fast_corners);
// #else
      // fast::fast_corner_detect_10(
      //     (fast::fast_byte*) img_pyr[L].data, img_pyr[L].cols,
      //     img_pyr[L].rows, img_pyr[L].cols, 20, fast_corners);
// #endif
    vector<int> scores, nm_corners;
    fast::fast_corner_score_8((fast::fast_byte*) img_pyr[L].data, img_pyr[L].cols, fast_corners, detection_threshold, scores);
    // fast::fast_corner_score_10((fast::fast_byte*) img_pyr[L].data, img_pyr[L].cols, fast_corners, detection_threshold, scores);
    // fast::fast_corner_score_12((fast::fast_byte*) img_pyr[L].data, img_pyr[L].cols, fast_corners, 20, scores);
    fast::fast_nonmax_3x3(fast_corners, scores, nm_corners);

    for(auto it=nm_corners.begin(), ite=nm_corners.end(); it!=ite; ++it)
    {
      fast::fast_xy& xy = fast_corners.at(*it);
      const int k = static_cast<int>((xy.y*scale)/cell_size_)*grid_n_cols_
                  + static_cast<int>((xy.x*scale)/cell_size_);
      if(grid_occupancy_[k])
        continue;
      const float score = hso::shiTomasiScore(img_pyr[L], xy.x, xy.y);
      // float Thres = frame->localThresh_[L].at(k);
      if(score > corners.at(k).score)
        corners.at(k) = Corner(xy.x*scale, xy.y*scale, score, L, 0.0f);
    }
  }
  // Create feature for every corner that has high enough corner score
  std::for_each(corners.begin(), corners.end(), [&](Corner& c) {
    if(c.score > detection_threshold)
    // if(c.score > c.angle)
      fts.push_back(new Feature(frame, Vector2d(c.x, c.y), c.level));
  });

  resetGrid();
}

EdgeletDetector::EdgeletDetector(
    const int img_width,
    const int img_height,
    const int cell_size,
    const int n_pyr_levels) :
        AbstractDetector(img_width, img_height, cell_size, n_pyr_levels)
{}

void EdgeletDetector::detect(
    Frame* frame,
    const ImgPyr& img_pyr,
    const float& detection_threshold,
    Features& fts)
{
  EdgeLets edgelets(grid_n_cols_*grid_n_rows_, EdgeLet(Vector2i(0,0), Vector2i(0,0), 0, 0.0f, false));
  Vector2d normal;

  for(int L = 0; L < n_pyr_levels_; ++L)
  {
    const int scale = (1<<L);
    const int cell_size_pyr = cell_size_/scale;

    // cv::Mat img_contours = frame->canny_[L];
    // cv::Canny( img_pyr[L], img_contours, detection_threshold, 3*detection_threshold, 3);

    for(size_t index = 0; index < grid_occupancy_.size(); ++index)
    {
      // if(grid_occupancy_.at(index) || edgelets.at(index).is_set)
      //   continue;
      if(grid_occupancy_.at(index))
        continue;
      // const int x = index % grid_n_cols_; 
      // const int y = index / grid_n_cols_;
      const int u = index % grid_n_cols_*cell_size_pyr; 
      const int v = index / grid_n_cols_*cell_size_pyr;
      if( u + cell_size_pyr <= frame->canny_[L].cols && v + cell_size_pyr <= frame->canny_[L].rows )
      {
        float grad_max = 0;  
        for(int i = 0; i < cell_size_pyr; ++i)
          for(int j = 0; j < cell_size_pyr; ++j)
          {
            if(frame->canny_[L].ptr<uchar>(v + i)[u + j] == 0)
              continue;

            int gx = frame->grad_pyr_[L].at<cv::Vec2s>(v+i, u+j)[0];
            int gy = frame->grad_pyr_[L].at<cv::Vec2s>(v+i, u+j)[1];
            float grad = sqrtf(gx*gx + gy*gy);

            if(grad > grad_max && grad > 20*detection_threshold && grad > edgelets.at(index).score)
            {
              grad_max = grad;
              edgelets.at(index).is_set = true;
              edgelets.at(index).mid = Vector2i((u+j)*scale, (v+i)*scale);   //level 0
              // double gx = gradx.ptr<float>(v+i)[u+j];
              // double gy = grady.ptr<float>(v+i)[u+j];
              normal = Vector2d(gx, gy);
              normal.normalize();
              edgelets.at(index).grad = normal;
              edgelets.at(index).level = L;
              edgelets.at(index).score = grad_max;
            }
          }
      }
    }
  }

  for(auto it = edgelets.begin(); it != edgelets.end(); ++it)
  {
    if(!it->is_set)
      continue;
    
    Vector2i center = it->mid;
    if(!frame->cam_->isInFrame(center, border_))
      continue;
    // float gx = gradx.ptr<float>(center.y())[center.x()];
    // float gy = grady.ptr<float>(center.y())[center.x()];
    // Vector2d normal_final(Vector2d(gx, gy));
    // normal_final.normalize();
    fts.push_back( new Feature(frame, Vector2d(center.x(), center.y()), it->grad, it->level) );
  }

  resetGrid();
}

GradientDetector::GradientDetector(
    const int img_width,
    const int img_height,
    const int cell_size,
    const int n_pyr_levels) :
        AbstractDetector(img_width, img_height, cell_size, n_pyr_levels)
{}

void GradientDetector::detect(
    Frame* frame,
    const ImgPyr& img_pyr,
    const float& detection_threshold,
    Features& fts)
{
  Gradients gradients(grid_n_cols_*grid_n_rows_, Gradient(0, 0, 0, 0));
  // for(int L = 0; L < n_pyr_levels_; ++L)
  // {
  //   const int scale = (1<<L);
  //   vector<fast::fast_xy> fast_corners;
  //   fast::fast_corner_detect_plain_7(
  //     (fast::fast_byte*) img_pyr[L].data, img_pyr[L].cols, img_pyr[L].rows, img_pyr[L].cols, 20, fast_corners);
  //   vector<int> scores, nm_corners;
  //   fast::fast_corner_score_7((fast::fast_byte*) img_pyr[L].data, img_pyr[L].cols, fast_corners, 20, scores);
  //   fast::fast_nonmax_3x3(fast_corners, scores, nm_corners);

  //   for(auto it=nm_corners.begin(), ite=nm_corners.end(); it!=ite; ++it)
  //   {
  //     fast::fast_xy& xy = fast_corners.at(*it);
  //     const int k = static_cast<int>((xy.y*scale)/cell_size_)*grid_n_cols_
  //                 + static_cast<int>((xy.x*scale)/cell_size_);
  //     if(grid_occupancy_[k])
  //       continue;
  //     const float delta = hso::shiTomasiScore(img_pyr[L], xy.x, xy.y);
  //     if(delta > gradients.at(k).delta)
  //       gradients.at(k) = Gradient(xy.x*scale, xy.y*scale, delta, L);
  //   }
  // }


  for(int L = 0; L < n_pyr_levels_; ++L)
  {
    const int scale = (1<<L);
    const int cell_size_pyr = cell_size_/scale;

    const int minBorderX = border_;
    const int minBorderY = minBorderX;
    const int maxBorderX = img_pyr[L].cols-border_;
    const int maxBorderY = img_pyr[L].rows-border_;

    for(size_t index = 0; index < grid_occupancy_.size(); ++index)
    {
        if(grid_occupancy_.at(index))
            continue;
        int iniX = index % grid_n_cols_ * cell_size_pyr; 
        int iniY = index / grid_n_cols_ * cell_size_pyr;

        if(iniX > maxBorderX || iniY > maxBorderY)
            continue;

        int maxX = iniX + cell_size_pyr;
        int maxY = iniY + cell_size_pyr;

        if(maxX > maxBorderX)
            maxX = maxBorderX;
        if(maxY > maxBorderY)
            maxY = maxBorderY;
        if(iniX < minBorderX)
            iniX = minBorderX;
        if(iniY < minBorderY)
            iniY = minBorderY;

        float max_delta = 0;

         for(int y = iniY; y < maxY; ++y)
            for(int x = iniX; x < maxX; ++x)
            {
                const int gx = frame->grad_pyr_[L].at<cv::Vec2s>(y, x)[0];
                const int gy = frame->grad_pyr_[L].at<cv::Vec2s>(y, x)[1];
                // int gx = frame->sobelX_[L].at<uchar>(v+y, u+x);
                // int gy = frame->sobelY_[L].at<uchar>(v+y, u+x);
                float grad = sqrtf(gx*gx + gy*gy); 


                // if(grad > max_delta && grad > 15*detection_threshold)
                // {
                //   max_delta = grad;
                //   const float score = hso::shiTomasiScore(img_pyr[L], u+x, v+y);
                //   if(score > gradients.at(index).delta)
                //   {
                //     gradients.at(index) = Gradient((u+x)*scale, (v+y)*scale, score, L);
                //     gradients.at(index).is_set = true;
                //   }
                // }
                if(grad > max_delta)
                {
                    max_delta = grad;
                    if(grad>20*detection_threshold && grad>gradients.at(index).delta)
                    {
                        gradients.at(index) = Gradient(x*scale, y*scale, grad, L);
                    // gradients.at(index).is_set = true;
                    }
                }
                // if(grad > max_delta && grad > 20*detection_threshold && grad > gradients.at(index).delta)
                // {
                //   max_delta = grad;
                //   gradients.at(index) = Gradient((u+x)*scale, (v+y)*scale, max_delta, L);
                //   gradients.at(index).is_set = true;
                // }
            }
    }
  }

  for(auto it = gradients.begin(); it != gradients.end(); ++it)
  {
    // if(it->delta < 15*detection_threshold)
    //   continue;
    if(it->delta < detection_threshold)
      continue;
    // cout << it->level << endl;
    fts.push_back(new Feature(frame, Vector2d(it->x, it->y), it->level, Feature::GRADIENT));
  }

  resetGrid();
}






FeatureExtractor::FeatureExtractor(
    const int width, const int height, const int cellSize, const int levels, bool isInit)
{
    cellSize_ = cellSize;

    width_ = width;
    height_ = height;
    vecWidth_.resize(levels);
    vecHeight_.resize(levels);
    vecWidth_[0] = width;
    vecHeight_[0] = height;
    for(int i=1; i<levels; ++i)
    {
        vecWidth_[i] = static_cast<int>(vecWidth_[i-1]/2);
        vecHeight_[i] = static_cast<int>(vecHeight_[i-1]/2);
    }

    nCols_ = std::ceil(static_cast<double>(width)/cellSize);
    nRows_ = std::ceil(static_cast<double>(height)/cellSize);

    nLevels_ = levels;
    featurePerLevel_.resize(nLevels_);
    cornerPerLevel_.resize(nLevels_);
    gradPerLevel_.resize(nLevels_);

    isInit_ = isInit;
    if(isInit)
        nFeatures_ = 2000;
    else
        nFeatures_ = Config::maxFts()+100;


    extFeatures_ = 0;

    vGrids_.resize(levels);
    vGridCols_.resize(levels);
    vGridRows_.resize(levels);
    haveFeatures_.resize(levels);
    for(int i = 0; i < levels; ++i)
    {
        const int gridPrySize = gridSize_/(1<<i);  
        vGrids_[i] = gridPrySize;
        vGridCols_[i] = std::ceil(static_cast<double>(vecWidth_[i])/gridPrySize);
        vGridRows_[i] = std::ceil(static_cast<double>(vecHeight_[i])/gridPrySize);
        haveFeatures_[i].resize(vGridCols_[i]*vGridRows_[i], false);
    }


    m_egde_filter = false;
    
}

void FeatureExtractor::detect(
    Frame* frame, const float initThresh, const float minThresh, Features& fts, Frame* last_frame)
{
    frame_ = frame;
    initThresh_ = initThresh;
    minThresh_ = minThresh;

    needFeatures_ = nFeatures_ - extFeatures_;

    if(last_frame != NULL)
    {
        m_egde_filter = true;
        m_last_frame = last_frame;
        findEpiHole();
    }
    else
    {
        m_egde_filter = false;
        m_last_frame = NULL;
    }

    featurePerLevel_.resize(nLevels_);
    // cornerPerLevel_.resize(nLevels_);
    // gradPerLevel_.resize(nLevels_);

    // fastDetect(frame->img_pyr_);
    fastDetectMT(frame->img_pyr_);
    // edgeLetDetectMT(frame->img_pyr_);
    // gradDetectMT(frame->img_pyr_);

    if(isInit_)
    {
        // gradDetectMT(frame->img_pyr_);
        fillingHole(frame->img_pyr_[0], 0);
    }
    else
    {
        edgeLetDetectMT(frame->img_pyr_);
        // fillingHole(frame->img_pyr_[0], 0);
        // fillingHole(frame->img_pyr_[1], 1);
        // fillingHole(frame->img_pyr_[2], 2);
    }


    for(size_t level = 0; level < featurePerLevel_.size(); ++level)
        for(size_t j = 0; j < featurePerLevel_[level].size(); ++j)
            allFeturesToDistribute_.push_back(featurePerLevel_[level].at(j));


    resultFeatures_ = computeKeyPointsOctTree(allFeturesToDistribute_, 0, width_, 0, height_, 0);

    for(size_t i=0; i<resultFeatures_.size(); ++i)
    {
        KeyPoint keyPoint = resultFeatures_[i];

        if(keyPoint.species == kCornerHigh)
            fts.push_back(new Feature(frame_, Vector2d(keyPoint.x, keyPoint.y), keyPoint.level));
        else if(keyPoint.species == kGrad)
        {
            Feature* feature = new Feature(frame_, Vector2d(keyPoint.x, keyPoint.y), keyPoint.level, Feature::GRADIENT);

            Vector2d normal(keyPoint.gx, keyPoint.gy);
            normal.normalize();
            feature->grad = normal;

            fts.push_back(feature);
        }
        else
        {
            Feature* feature = new Feature(frame_, Vector2d(keyPoint.x, keyPoint.y), keyPoint.level, Feature::EDGELET);

            Vector2d normal(keyPoint.gx, keyPoint.gy);
            normal.normalize();
            feature->grad = normal;

            fts.push_back(feature);
        }
    }

    resetGrid();

    allFeturesToDistribute_.clear();
    featurePerLevel_.clear();
    // cornerPerLevel_.clear();
    // gradPerLevel_.clear();
    resultFeatures_.clear();

    extFeatures_ = 0;
}

void FeatureExtractor::fastDetectMT(const ImgPyr& img_pyr)
{
    if(nLevels_ == 1)
    {
        fastDetect(img_pyr);
        // fastDetectST(img_pyr[0], 0);
    }
    else
    {
        assert(nLevels_ == 3);
        std::thread thread0(&FeatureExtractor::fastDetectST, this, std::ref(img_pyr[0]), 0);
        std::thread thread1(&FeatureExtractor::fastDetectST, this, std::ref(img_pyr[1]), 1);
        std::thread thread2(&FeatureExtractor::fastDetectST, this, std::ref(img_pyr[2]), 2);

        thread0.join();
        thread1.join();
        thread2.join();
    }
}

void FeatureExtractor::fastDetectST(const cv::Mat& imageLevel, const int Level)
{
    const int border = 8;
    const int scale = (1<<Level);
    const short fastThresh = floor(minThresh_);

    // first detect corner
    vector<fast::fast_xy> fastCorners9;
    fast::fast_corner_detect_9_sse2(
            (fast::fast_byte*)imageLevel.data, imageLevel.cols, imageLevel.rows, imageLevel.cols, fastThresh, fastCorners9);

    vector<int> scores9, nonMaxCornersIndex9;
    fast::fast_corner_score_9((fast::fast_byte*)imageLevel.data, imageLevel.cols, fastCorners9, fastThresh, scores9);
    fast::fast_nonmax_3x3(fastCorners9, scores9, nonMaxCornersIndex9);

    for(vector<int>::iterator it=nonMaxCornersIndex9.begin(), ite=nonMaxCornersIndex9.end(); it!=ite; ++it)
    {
        fast::fast_xy xy = fastCorners9.at(*it);

        if(xy.x < border || xy.x > vecWidth_[Level]-border || xy.y < border || xy.y > vecHeight_[Level]-border)
            continue;

        haveFeatures_[Level].at(getCellIndex(xy.x, xy.y, Level)) = true;

        featurePerLevel_[Level].push_back(
            KeyPoint(xy.x*scale, xy.y*scale, hso::shiTomasiScore(imageLevel, xy.x, xy.y), Level, kCornerHigh));
    }
}

void FeatureExtractor::fastDetect(const ImgPyr& img_pyr)
{
    // Corners corners(nCols_*nRows_, Corner(0,0,minThresh_,0,0.0f));
    const int border = 8;

    for(int L=0; L<nLevels_; ++L)
    {
        const int scale = (1<<L);
        vector<fast::fast_xy> fastCorners;

        fast::fast_corner_detect_9_sse2(
            (fast::fast_byte*) img_pyr[L].data, img_pyr[L].cols, img_pyr[L].rows, img_pyr[L].cols, minThresh_, fastCorners);
        // fast::fast_corner_detect_plain_8(
        //     (fast::fast_byte*) img_pyr[L].data, img_pyr[L].cols, img_pyr[L].rows, img_pyr[L].cols, minThresh_, fastCorners);

        vector<int> scores, nonMaxCornersIndex;
        fast::fast_corner_score_9((fast::fast_byte*)img_pyr[L].data, img_pyr[L].cols, fastCorners, minThresh_, scores);

        fast::fast_nonmax_3x3(fastCorners, scores, nonMaxCornersIndex);

        // for(size_t i = 0; i < fastCorners.size(); ++i)
        for(vector<int>::iterator it=nonMaxCornersIndex.begin(), ite=nonMaxCornersIndex.end(); it!=ite; ++it)
        {
            fast::fast_xy xy = fastCorners.at(*it);
            // fast::fast_xy xy = fastCorners[i];

            if(xy.x < border || xy.x > vecWidth_[L]-border || xy.y < border || xy.y > vecHeight_[L]-border)
                continue;

            const float response = hso::shiTomasiScore(img_pyr[L], xy.x, xy.y);

            // allFeturesToDistribute_.push_back(KeyPoint(xy.x*scale, xy.y*scale, response, kCorner));

            int index = getCellIndex(xy.x, xy.y, L);
            haveFeatures_[L].at(index) = true;

            featurePerLevel_[L].push_back(KeyPoint(xy.x*scale, xy.y*scale, response, L, kCornerHigh));
            // cornerPerLevel_[L].push_back(KeyPoint(xy.x*scale, xy.y*scale, response, L, kCorner));
        }
    }
}

void FeatureExtractor::gradDetectMT(const ImgPyr& img_pyr)
{
    if(nLevels_ == 1)
    {
        gradDetect(img_pyr);
        // gradDetectST(img_pyr[0], 0);
    }
    else
    {
        assert(nLevels_ == 3);
        std::thread thread0(&FeatureExtractor::gradDetectST, this, std::ref(img_pyr[0]), 0);
        std::thread thread1(&FeatureExtractor::gradDetectST, this, std::ref(img_pyr[1]), 1);
        std::thread thread2(&FeatureExtractor::gradDetectST, this, std::ref(img_pyr[2]), 2);

        thread0.join();
        thread1.join();
        thread2.join();
    }
}

void FeatureExtractor::gradDetectST(const cv::Mat& imageLevel, const int Level)
{
    const int border = 8;
    const int scale = (1<<Level);
    const int gridPrySize = vGrids_[Level];
    const int minBorderX = border;
    const int minBorderY = minBorderX;
    const int maxBorderX = imageLevel.cols-border;
    const int maxBorderY = imageLevel.rows-border;

    for(size_t index = 0; index < haveFeatures_[Level].size(); ++index)
    {
        if(haveFeatures_[Level].at(index)) continue;

        int iniX = index % vGridCols_[Level] * gridPrySize;
        int iniY = index / vGridRows_[Level] * gridPrySize;
        if(iniX > maxBorderX || iniY > maxBorderY) continue;

        int maxX = iniX + gridPrySize;
        int maxY = iniY + gridPrySize;

        if(maxX > maxBorderX) maxX = maxBorderX;
        if(maxY > maxBorderY) maxY = maxBorderY;
        if(iniX < minBorderX) iniX = minBorderX;
        if(iniY < minBorderY) iniY = minBorderY;

        KeyPoint kp;
        bool isSet = false;
        float maxGrad = 0;
        for(int y=iniY; y<maxY; ++y)
        {
            for(int x=iniX; x<maxX; ++x)
            {
                const int gx = frame_->grad_pyr_[Level].at<cv::Vec2s>(y, x)[0];
                const int gy = frame_->grad_pyr_[Level].at<cv::Vec2s>(y, x)[1];

                const float grad = sqrtf(gx*gx + gy*gy);

                if(grad>20.0f*minThresh_ && grad>maxGrad)
                {
                    kp = KeyPoint(x*scale, y*scale, grad, Level, kGrad);
                    kp.gx = gx;
                    kp.gy = gy;

                    isSet = true;
                    maxGrad = grad;
                }
            }
        }

        if(isSet) {
            featurePerLevel_[Level].push_back(kp);
            haveFeatures_[Level].at(index) = true;
        }
    }
}


void FeatureExtractor::gradDetect(const ImgPyr& img_pyr)
{
    const int border = 8;

    for(int L = 0; L < nLevels_; ++L)
    {
        const int scale = (1<<L);
        const int gridPrySize = vGrids_[L];
        const int minBorderX = border;
        const int minBorderY = minBorderX;
        const int maxBorderX = img_pyr[L].cols-border;
        const int maxBorderY = img_pyr[L].rows-border;

        for(size_t index = 0; index < haveFeatures_[L].size(); ++index)
        {
            if(haveFeatures_[L].at(index))
                continue;

            int iniX = index % vGridCols_[L] * gridPrySize;
            int iniY = index / vGridRows_[L] * gridPrySize;

            if(iniX > maxBorderX || iniY > maxBorderY)
                continue;

            int maxX = iniX + gridPrySize;
            int maxY = iniY + gridPrySize;

            if(maxX > maxBorderX) maxX = maxBorderX;
            if(maxY > maxBorderY) maxY = maxBorderY;
            if(iniX < minBorderX) iniX = minBorderX;
            if(iniY < minBorderY) iniY = minBorderY;

            KeyPoint kp;
            bool isSet = false;
            float maxGrad = 0;
            for(int y=iniY; y<maxY; ++y)
            {
                for(int x=iniX; x<maxX; ++x)
                {
                    const int gx = frame_->grad_pyr_[L].at<cv::Vec2s>(y, x)[0];
                    const int gy = frame_->grad_pyr_[L].at<cv::Vec2s>(y, x)[1];
                    const float grad = sqrtf(gx*gx + gy*gy);


                    if(grad > 20.0f*minThresh_ && grad > maxGrad)
                    {
                        kp = KeyPoint(x*scale, y*scale, grad, L, kGrad);
                        kp.gx = gx;
                        kp.gy = gy;

                        isSet = true;
                        maxGrad = grad;
                    }
                }
            }

            if(isSet) {
                featurePerLevel_[L].push_back(kp);
                haveFeatures_[L].at(index) = true;
            }
        }
    }
}

void FeatureExtractor::edgeLetDetectMT(const ImgPyr& img_pyr)
{
    if(nLevels_ == 1)
        edgeLetDetectST(img_pyr[0], 0);
    else
    {
        assert(nLevels_ == 3);

        std::thread thread0(&FeatureExtractor::edgeLetDetectST, this, std::ref(img_pyr[0]), 0);
        std::thread thread1(&FeatureExtractor::edgeLetDetectST, this, std::ref(img_pyr[1]), 1);
        std::thread thread2(&FeatureExtractor::edgeLetDetectST, this, std::ref(img_pyr[2]), 2);

        thread0.join();
        thread1.join();
        thread2.join();
    }
}

void FeatureExtractor::edgeLetDetectST(const cv::Mat& imageLevel, const int Level)
{
    cv::Mat imgEdge;
    // cv::Canny(imageLevel, imgEdge, 40*minThresh_, 100*minThresh_, 5, true);

    // // cv::Mat dx, dy;
    // cv::Sobel(imageLevel, frame_->sobelX_[Level], CV_16S, 1, 0, 5, 1, 0, BORDER_REPLICATE);
    // cv::Sobel(imageLevel, frame_->sobelY_[Level], CV_16S, 0, 1, 5, 1, 0, BORDER_REPLICATE);
    cv::Canny(frame_->sobelX_[Level], frame_->sobelY_[Level], imgEdge, 31*minThresh_, 70*minThresh_, true);

    // Mat dx,dy;
    // cv::Scharr(imageLevel,dx,CV_16S,1,0);
    // cv::Scharr(imageLevel,dy,CV_16S,0,1);
    // cv::Canny(dx, dy, imgEdge, 6*minThresh_, 15*minThresh_, true);

    // cv::imshow("Canny", imgEdge);
    // cv::waitKey();

    const int border = 8;
    const int scale = (1<<Level);
    const int gridPrySize = vGrids_[Level];
    const int minBorderX = border;
    const int minBorderY = minBorderX;
    const int maxBorderX = imageLevel.cols-border;
    const int maxBorderY = imageLevel.rows-border;

    for(size_t index = 0; index < haveFeatures_[Level].size(); ++index)
    {
        if(haveFeatures_[Level].at(index)) continue;

        int iniX = index % vGridCols_[Level] * gridPrySize;
        int iniY = index / vGridRows_[Level] * gridPrySize;
        if(iniX > maxBorderX || iniY > maxBorderY) continue;

        int maxX = iniX + gridPrySize;
        int maxY = iniY + gridPrySize;

        if(maxX > maxBorderX) maxX = maxBorderX;
        if(maxY > maxBorderY) maxY = maxBorderY;
        if(iniX < minBorderX) iniX = minBorderX;
        if(iniY < minBorderY) iniY = minBorderY;

        KeyPoint kp;
        bool isSet = false;
        float maxGrad = 0;
        for(int y=iniY; y<maxY; ++y)
        {
            for(int x=iniX; x<maxX; ++x)
            {
                if(imgEdge.at<uchar>(y,x) == 0) continue;

                // const int gx = frame_->grad_pyr_[Level].at<cv::Vec2s>(y, x)[0];
                // const int gy = frame_->grad_pyr_[Level].at<cv::Vec2s>(y, x)[1];
                const short gx = frame_->sobelX_[Level].at<short>(y,x);
                const short gy = frame_->sobelY_[Level].at<short>(y,x);

                // double angle = 1.0;
                // if(m_egde_filter)
                //     if(!edgeletFilter(x,y,gx,gy,Level, angle))
                //         continue;

                float grad = sqrtf(gx*gx + gy*gy);
                // grad *= angle;
                
                if(grad > maxGrad)
                {
                    kp = KeyPoint(x*scale, y*scale, grad, Level, kEdgeLet);
                    kp.gx = gx;
                    kp.gy = gy;

                    isSet = true;
                    maxGrad = grad;
                }
            }
        }

        if(isSet) {
            featurePerLevel_[Level].push_back(kp);
            haveFeatures_[Level].at(index) = true;
        }
    }
}

// Modified code in ORB-SLAM (https://github.com/raulmur/ORB_SLAM)
vector<KeyPoint> FeatureExtractor::computeKeyPointsOctTree(
    const vector<KeyPoint>& toDistributeKeys, 
    const int &minX, const int &maxX, const int &minY, const int &maxY, const int &level)
{
    // Compute how many initial nodes   
    const int nIni = round(static_cast<float>(maxX-minX)/(maxY-minY));

    const float hX = static_cast<float>(maxX-minX)/nIni;

    list<ExtractorNode> lNodes;

    vector<ExtractorNode*> vpIniNodes;
    vpIniNodes.resize(nIni);

    for(int i=0; i<nIni; ++i)
    {
        ExtractorNode ni;
        ni.UL = cv::Point2i(hX*static_cast<float>(i),minY);
        ni.UR = cv::Point2i(hX*static_cast<float>(i+1),minY);
        ni.BL = cv::Point2i(ni.UL.x,maxY);
        ni.BR = cv::Point2i(ni.UR.x,maxY);
        ni.vKeys.reserve(toDistributeKeys.size());

        lNodes.push_back(ni);
        vpIniNodes[i] = &lNodes.back();
    }

    for(size_t i=0; i<toDistributeKeys.size(); ++i)
    {
        const int x = toDistributeKeys[i].x;
        // if(x/hX >= nIni)
        //     continue;

        vpIniNodes[x/hX]->vKeys.push_back(toDistributeKeys[i]);
    }

    list<ExtractorNode>::iterator lit = lNodes.begin();
    while(lit != lNodes.end())
    {
        if(lit->vKeys.size() == 1)
        {
            lit->bNoMore = true;
            lit++;
        }
        else if(lit->vKeys.empty())
            lit = lNodes.erase(lit);
        else
            lit++;
    }

    bool bFinish = false;

    int iteration = 0;

    vector<pair<int,ExtractorNode*> > vSizeAndPointerToNode;
    vSizeAndPointerToNode.reserve(lNodes.size()*4);

    while(!bFinish)
    {
        iteration++;

        int prevSize = lNodes.size();

        lit = lNodes.begin();

        int nToExpand = 0;

        vSizeAndPointerToNode.clear();

        while(lit != lNodes.end())
        {
            if(lit->bNoMore)
            {
                ++lit;
                continue;
            }
            else
            {
                ExtractorNode n1,n2,n3,n4;
                lit->DivideNode(n1,n2,n3,n4);

                if(n1.vKeys.size() > 0)
                {
                    lNodes.push_front(n1);
                    if(n1.vKeys.size() > 1)
                    {
                        nToExpand++;
                        vSizeAndPointerToNode.push_back(make_pair(n1.vKeys.size(),&lNodes.front()));
                        lNodes.front().lit = lNodes.begin();
                    }
                }

                if(n2.vKeys.size()>0)
                {
                    lNodes.push_front(n2);
                    if(n2.vKeys.size()>1)
                    {
                        nToExpand++;
                        vSizeAndPointerToNode.push_back(make_pair(n2.vKeys.size(),&lNodes.front()));
                        lNodes.front().lit = lNodes.begin();
                    }
                }

                if(n3.vKeys.size()>0)
                {
                    lNodes.push_front(n3);
                    if(n3.vKeys.size()>1)
                    {
                        nToExpand++;
                        vSizeAndPointerToNode.push_back(make_pair(n3.vKeys.size(),&lNodes.front()));
                        lNodes.front().lit = lNodes.begin();
                    }
                }

                if(n4.vKeys.size()>0)
                {
                    lNodes.push_front(n4);
                    if(n4.vKeys.size()>1)
                    {
                        nToExpand++;
                        vSizeAndPointerToNode.push_back(make_pair(n4.vKeys.size(),&lNodes.front()));
                        lNodes.front().lit = lNodes.begin();
                    }
                }

                lit=lNodes.erase(lit);
                continue;
            }
        }

        if((int)lNodes.size()>=nFeatures_ || (int)lNodes.size()==prevSize)
        {
            bFinish = true;
        }
        else if(((int)lNodes.size()+nToExpand*3)>nFeatures_)
        {
            while(!bFinish)
            {

                prevSize = lNodes.size();

                vector<pair<int,ExtractorNode*> > vPrevSizeAndPointerToNode = vSizeAndPointerToNode;
                vSizeAndPointerToNode.clear();

                std::sort(vPrevSizeAndPointerToNode.begin(),vPrevSizeAndPointerToNode.end());
                for(int j=vPrevSizeAndPointerToNode.size()-1;j>=0;j--)
                {
                    ExtractorNode n1,n2,n3,n4;
                    vPrevSizeAndPointerToNode[j].second->DivideNode(n1,n2,n3,n4);

                    // Add childs if they contain points
                    if(n1.vKeys.size()>0)
                    {
                        lNodes.push_front(n1);
                        if(n1.vKeys.size()>1)
                        {
                            vSizeAndPointerToNode.push_back(make_pair(n1.vKeys.size(),&lNodes.front()));
                            lNodes.front().lit = lNodes.begin();
                        }
                    }
                    if(n2.vKeys.size()>0)
                    {
                        lNodes.push_front(n2);
                        if(n2.vKeys.size()>1)
                        {
                            vSizeAndPointerToNode.push_back(make_pair(n2.vKeys.size(),&lNodes.front()));
                            lNodes.front().lit = lNodes.begin();
                        }
                    }
                    if(n3.vKeys.size()>0)
                    {
                        lNodes.push_front(n3);
                        if(n3.vKeys.size()>1)
                        {
                            vSizeAndPointerToNode.push_back(make_pair(n3.vKeys.size(),&lNodes.front()));
                            lNodes.front().lit = lNodes.begin();
                        }
                    }
                    if(n4.vKeys.size()>0)
                    {
                        lNodes.push_front(n4);
                        if(n4.vKeys.size()>1)
                        {
                            vSizeAndPointerToNode.push_back(make_pair(n4.vKeys.size(),&lNodes.front()));
                            lNodes.front().lit = lNodes.begin();
                        }
                    }

                    lNodes.erase(vPrevSizeAndPointerToNode[j].second->lit);

                    if((int)lNodes.size()>=nFeatures_)
                        break;
                }

                if((int)lNodes.size()>=nFeatures_ || (int)lNodes.size()==prevSize)
                    bFinish = true;
            }
        }
    }

    std::vector<KeyPoint> vResultKeys;
    vResultKeys.reserve(nFeatures_);
    for(list<ExtractorNode>::iterator lit = lNodes.begin(); lit != lNodes.end(); ++lit)
    {
        // vector<KeyPoint> &vNodeKeys = lit->vKeys;

        // KeyPoint resultKeyPoint; resultKeyPoint.species = kGrad;
        // KeyPoint highestOccur; highestOccur.species = kOccur; highestOccur.secondSpecies = kGrad;

        // float scoreMax = 0;
        // bool haveOccur = false, isValid = false;

        // for(size_t k = 0; k < vNodeKeys.size(); ++k)
        // {
        //     if(vNodeKeys[k].species == kOccur)
        //     {
        //         haveOccur = true;
        //         if(vNodeKeys[k].secondSpecies < highestOccur.secondSpecies)
        //             highestOccur = vNodeKeys[k];

        //         continue;
        //     }

        //     if(resultKeyPoint.species > vNodeKeys[k].species)
        //     {
        //         resultKeyPoint = vNodeKeys[k];
        //         scoreMax = vNodeKeys[k].response;
        //         isValid = true;
        //     }
        //     else if(resultKeyPoint.species == vNodeKeys[k].species)
        //     {
        //         if(vNodeKeys[k].response > scoreMax)
        //         {
        //             resultKeyPoint = vNodeKeys[k];
        //             scoreMax = vNodeKeys[k].response;
        //             isValid = true;
        //         }
        //     }
        // }

        // if(!haveOccur)
        // {
        //     if(isValid) vResultKeys.push_back(resultKeyPoint);
        // }
        // else
        // {
        //     if(isValid && resultKeyPoint.species < highestOccur.secondSpecies && resultKeyPoint.species == kCornerHigh)
        //     {
        //         // cout << "HAHAHAHAHAHAHA" << endl;
        //         vResultKeys.push_back(resultKeyPoint);
        //     }
        // }

        vector<KeyPoint> &vNodeKeys = lit->vKeys;

        KeyPoint* pKP = &vNodeKeys[0];
        float maxScore = pKP->response;;
        bool haveOccur = false;

        if(pKP->species == kOccur) continue;

        if(vNodeKeys.size() > 1)
        {
            for(size_t k = 1; k < vNodeKeys.size(); ++k)
            {
                KeyPoint* pt = &vNodeKeys[k];

                if(pt->species == kOccur) {
                    haveOccur = true;
                    break;
                }

                if(pKP->species > vNodeKeys[k].species)
                {
                    pKP = &vNodeKeys[k];
                    maxScore = vNodeKeys[k].response;
                }
                else if(pKP->species == vNodeKeys[k].species)
                {
                    if(vNodeKeys[k].response > maxScore)
                    {
                        pKP = &vNodeKeys[k];
                        maxScore = vNodeKeys[k].response;
                    }
                }
            }
        }
        if(!haveOccur) vResultKeys.push_back(*pKP);
    }

    return vResultKeys;
}

void FeatureExtractor::fillingHole(const cv::Mat& imageLevel, const int Level)
{
    const int border = 8;
    const int scale = (1<<Level);
    const short fastThresh = 0.6*minThresh_ > 6? 0.6*minThresh_ : 6;

    vector<fast::fast_xy> fastCorners;
    fast::fast_corner_detect_plain_12(
            (fast::fast_byte*)imageLevel.data, imageLevel.cols, imageLevel.rows, imageLevel.cols, fastThresh, fastCorners);

    vector<int> scores, nonMaxCornersIndex;
    fast::fast_corner_score_12((fast::fast_byte*)imageLevel.data, imageLevel.cols, fastCorners, fastThresh, scores);
    fast::fast_nonmax_3x3(fastCorners, scores, nonMaxCornersIndex);

    for(vector<int>::iterator it=nonMaxCornersIndex.begin(), ite=nonMaxCornersIndex.end(); it!=ite; ++it)
    {
        fast::fast_xy xy = fastCorners.at(*it);
        if(xy.x < border || xy.x > vecWidth_[Level]-border || xy.y < border || xy.y > vecHeight_[Level]-border)
            continue;

        int index = getCellIndex(xy.x, xy.y, Level);
        if(haveFeatures_[Level].at(index)) continue;

        haveFeatures_[Level].at(index) = true;

        featurePerLevel_[Level].push_back(
            KeyPoint(xy.x*scale, xy.y*scale, hso::shiTomasiScore(imageLevel, xy.x, xy.y), Level, kGrad));
    }
}

void FeatureExtractor::resetGrid()
{
    for(size_t i = 0; i < haveFeatures_.size(); ++i)
        std::fill(haveFeatures_[i].begin(), haveFeatures_[i].end(), false);
}

void FeatureExtractor::setGridOccpuancy(const Vector2d& px, Feature* occurFeature)
{
    allFeturesToDistribute_.push_back(KeyPoint(px.cast<float>().x(), px.cast<float>().y(), 0, 0, kOccur));

    extFeatures_++;
}

void FeatureExtractor::setExistingFeatures(const Features& fts)
{
    for(auto& ftr : fts)
    {
        allFeturesToDistribute_.push_back(KeyPoint(ftr->px.cast<float>().x(), ftr->px.cast<float>().y(), 0, 0, kOccur));
    }   

    extFeatures_ += fts.size();
}


void FeatureExtractor::findEpiHole()
{
    assert(m_last_frame != NULL);
    assert(m_last_frame == frame_->m_last_frame.get());
    epi_hole = frame_->w2c(m_last_frame->pos());
}

bool FeatureExtractor::edgeletFilter(int u_level, int v_level, short gx, short gy, int level, double& angle)
{
    Vector2d grad_dir(gx,gy);
    grad_dir.normalize();

    Vector2d hole_level(epi_hole/(1<<level));
    Vector2d epi_dir = (hole_level - Vector2d(u_level,v_level)).normalized();

    angle = fabs(grad_dir.dot(epi_dir));
    if(angle < 0.1) return false;

    return true;
}

} // namespace feature_detection
} // namespace hso

