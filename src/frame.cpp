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


#include <stdexcept>
#include <hso/frame.h>
#include <hso/feature.h>
#include <hso/point.h>
#include <hso/config.h>
#include <boost/bind.hpp>
#include <fast/fast.h>

// #include "hso/PhotomatricCalibration.h"
#include "hso/vikit/math_utils.h"
#include "hso/vikit/vision.h"

using namespace cv;

namespace hso {

int Frame::frame_counter_ = 0;
int Frame::keyFrameCounter_ = 0;


Frame::Frame(hso::AbstractCamera* cam, const cv::Mat& img, double timestamp) :
             id_(frame_counter_++), timestamp_(timestamp), 
             cam_(cam), key_pts_(5), is_keyframe_(false), v_kf_(NULL), gradMean_(0)
{
    // if(opc != NULL) m_pc = opc;

    initFrame(img);
}

Frame::~Frame()
{
    std::for_each(fts_.begin(), fts_.end(), [&](Feature* i)
    {
        if(i->m_prev_feature != NULL)
        {
            assert(i->m_prev_feature->frame->isKeyframe());
            i->m_prev_feature->m_next_feature = NULL;
            i->m_prev_feature = NULL;
        }

        if(i->m_next_feature != NULL)
        {
            i->m_next_feature->m_prev_feature = NULL;
            i->m_next_feature = NULL;
        }

        delete i; i=NULL;
    });

    img_pyr_.clear();
    grad_pyr_.clear();
    sobelX_.clear();
    sobelY_.clear();
    canny_.clear();
    m_pyr_raw.clear();
}

void Frame::initFrame(const cv::Mat& img)
{
    // check image
    if(img.empty() || img.type() != CV_8UC1 || img.cols != cam_->width() || img.rows != cam_->height())
        throw std::runtime_error("Frame: provided image has not the same size as the camera model or image is not grayscale");

    // Set keypoints to NULL
    std::for_each(key_pts_.begin(), key_pts_.end(), [&](Feature* ftr){ ftr=NULL; });

    // Build Image Pyramid
    frame_utils::createImgPyramid(img, max(Config::nPyrLevels(), Config::kltMaxLevel()+1), img_pyr_);
    // photometricallyCorrectPyramid(img, img_pyr_, m_pyr_raw, max(Config::nPyrLevels(), Config::kltMaxLevel()+1));

    prepareForFeatureDetect();    
}

void Frame::setKeyframe()
{
    is_keyframe_ = true;
    setKeyPoints();

    keyFrameCounter_++;
    keyFrameId_ = keyFrameCounter_;
}

void Frame::addFeature(Feature* ftr)
{
    boost::unique_lock<boost::mutex> lock(m_fts_mutex);
    fts_.push_back(ftr);
}

void Frame::getFeaturesCopy(Features& list_copy)
{
    boost::unique_lock<boost::mutex> lock(m_fts_mutex);
    for(auto it = fts_.begin(); it != fts_.end(); ++it)
        list_copy.push_back(*it);
}

void Frame::setKeyPoints() // thread safe
{
  for(size_t i = 0; i < 5; ++i)
    if(key_pts_[i] != NULL)
      if(key_pts_[i]->point == NULL)
        key_pts_[i] = NULL;

  std::for_each(fts_.begin(), fts_.end(), [&](Feature* ftr){ if(ftr->point != NULL) checkKeyPoints(ftr); });
}

void Frame::checkKeyPoints(Feature* ftr)
{
  const int cu = cam_->width()/2;
  const int cv = cam_->height()/2;
  const Vector2d uv = ftr->px;

  // center point
  if(key_pts_[0] == NULL)
    key_pts_[0] = ftr;
  else if(std::max(std::fabs(ftr->px[0]-cu), std::fabs(ftr->px[1]-cv))  
          < std::max(std::fabs(key_pts_[0]->px[0]-cu), std::fabs(key_pts_[0]->px[1]-cv)))
    key_pts_[0] = ftr;
  // right dn
  if(uv[0] >= cu && uv[1] >= cv)
  {
    if(key_pts_[1] == NULL)
      key_pts_[1] = ftr;
    else if((uv[0] - cu) * (uv[1] - cv)
        >(key_pts_[1]->px[0] - cu) * (key_pts_[1]->px[1] - cv))
      key_pts_[1] = ftr;
  }
  // right up
  if(uv[0] >= cu && uv[1] < cv)
  {
    if(key_pts_[2] == NULL)
      key_pts_[2] = ftr;
    else if((uv[0] - cu) * -(uv[1] - cv)
        >(key_pts_[2]->px[0] - cu) * -(key_pts_[2]->px[1] - cv))
      key_pts_[2] = ftr;
  }
  // left dn
  if(uv[0] < cu && uv[1] >= cv)
  {
    if(key_pts_[3] == NULL)
      key_pts_[3] = ftr;
    else if(-(uv[0] - cu) * (uv[1] - cv)
        >-(key_pts_[3]->px[0] - cu) * (key_pts_[3]->px[1] - cv))
      key_pts_[3] = ftr;
  }
  // left up
  if(uv[0] < cu && uv[1] < cv)
  {
    if(key_pts_[4] == NULL)
      key_pts_[4] = ftr;
    else if(-(uv[0] - cu) * -(uv[1] - cv)
        >-(key_pts_[4]->px[0] - cu) * -(key_pts_[4]->px[1] - cv))
      key_pts_[4] = ftr;
  }
}

void Frame::removeKeyPoint(Feature* ftr)
{
    bool found = false;
    std::for_each(key_pts_.begin(), key_pts_.end(), [&](Feature*& i){
    if(i == ftr) {
        i = NULL;
        found = true;
    }
    });

    if(found) setKeyPoints();
}

bool Frame::isVisible(const Vector3d& xyz_w) const
{
    Vector3d xyz_f = T_f_w_*xyz_w;
    if(xyz_f.z() < 0.0) return false; // point is behind the camera

    Vector2d px = f2c(xyz_f);
    if(px[0] >= 0.0 && px[1] >= 0.0 && px[0] < cam_->width() && px[1] < cam_->height())
        return true;

    return false;
}

void Frame::prepareForFeatureDetect()
{   
    //TODO:delete it   in Initialization
    // frame_utils::createImgGrad(img_pyr_, grad_pyr_, std::max(Config::nPyrLevels(), Config::kltMaxLevel()+1));

    // For change normal
    sobelX_.resize(Config::nPyrLevels());
    sobelY_.resize(Config::nPyrLevels());

    assert(Config::nPyrLevels() == 3);

    for(int i = 0; i < 3; ++i)
    {
        cv::Sobel(img_pyr_[i], sobelX_[i], CV_16S, 1, 0, 5, 1, 0, BORDER_REPLICATE);
        cv::Sobel(img_pyr_[i], sobelY_[i], CV_16S, 0, 1, 5, 1, 0, BORDER_REPLICATE);
    }


    float intSum = 0, gradSum = 0;
    int sum = 0;
    for(int y=16;y<img_pyr_[0].rows-16;y++)
        for(int x=16;x<img_pyr_[0].cols-16;x++)
        {
            sum++;
            // float gradx = grad_pyr_[0].at<cv::Vec2s>(y, x)[0];
            // float grady = grad_pyr_[0].at<cv::Vec2s>(y, x)[1];
            float gradx = sobelX_[0].at<short>(y,x);
            float grady = sobelY_[0].at<short>(y,x);
            gradSum += sqrtf(gradx*gradx + grady*grady);

            intSum += img_pyr_[0].ptr<uchar>(y)[x];

        }

    integralImage_ = intSum/sum;

    gradMean_ = gradSum/sum;
    gradMean_ /= 30;
    // gradMean_ += 0.5f;
    if(gradMean_ > 20) gradMean_ = 20;
    if(gradMean_ < 7)  gradMean_ = 7;
}

// void Frame::photometricallyCorrectPyramid(const cv::Mat& img_level_0, ImgPyr& pyr_correct, ImgPyr& pyr_raw, int n_levels)
// {
//     pyr_correct.resize(n_levels);
//     cv::Mat image_corrected_0 = img_level_0.clone();
//     m_pc->photometricallyCorrectImage(image_corrected_0);
//     pyr_correct[0] = image_corrected_0;
//     for(size_t L=1; L<pyr_correct.size(); ++L)
//     {
//         if(img_level_0.cols % 16 == 0 && img_level_0.rows % 16 == 0)
//         {
//             pyr_correct[L] = cv::Mat(pyr_correct[L-1].rows/2, pyr_correct[L-1].cols/2, CV_8U);
//             hso::halfSample(pyr_correct[L-1], pyr_correct[L]);
//         }
//         else
//         {
//             float scale = 1.0/(1<<L);
//             cv::Size sz(cvRound((float)img_level_0.cols*scale), cvRound((float)img_level_0.rows*scale));
//             cv::resize(pyr_correct[L-1], pyr_correct[L], sz, 0, 0, cv::INTER_LINEAR);
//         }
//     }
    
//     pyr_raw.resize(Config::nPyrLevels()); //3
//     pyr_raw[0] = img_level_0.clone();
//     for(size_t L=1; L<pyr_raw.size(); ++L)
//     {
//         if(img_level_0.cols % 16 == 0 && img_level_0.rows % 16 == 0)
//         {
//             pyr_raw[L] = cv::Mat(pyr_raw[L-1].rows/2, pyr_raw[L-1].cols/2, CV_8U);
//             hso::halfSample(pyr_raw[L-1], pyr_raw[L]);
//         }
//         else
//         {
//             float scale = 1.0/(1<<L);
//             cv::Size sz(cvRound((float)img_level_0.cols*scale), cvRound((float)img_level_0.rows*scale));
//             cv::resize(pyr_raw[L-1], pyr_raw[L], sz, 0, 0, cv::INTER_LINEAR);
//         }
//     }
// }

void Frame::finish()
{
    grad_pyr_.clear();
    canny_.clear();   
}

/// Utility functions for the Frame class
namespace frame_utils {

void createImgPyramid(const cv::Mat& img_level_0, int n_levels, ImgPyr& pyr)
{
    pyr.resize(n_levels);
    pyr[0] = img_level_0;
    for(int i=1; i<n_levels; ++i)
    {
        if(img_level_0.cols % 16 == 0 && img_level_0.rows % 16 == 0)
        {
            pyr[i] = cv::Mat(pyr[i-1].rows/2, pyr[i-1].cols/2, CV_8U);
            hso::halfSample(pyr[i-1], pyr[i]);
        }
        else
        {
            float scale = 1.0/(1<<i);
            cv::Size sz(cvRound((float)img_level_0.cols*scale), cvRound((float)img_level_0.rows*scale));
            cv::resize(pyr[i-1], pyr[i], sz, 0, 0, cv::INTER_LINEAR);
        }     
    }
}

void createImgGrad(const ImgPyr& pyr_img, ImgPyr& scharr, int n_levels)
{ 
    scharr.resize(n_levels);
    for(int i = 0; i < n_levels; ++i)
        hso::calcSharrDeriv(pyr_img[i], scharr[i]);
}

bool getSceneDepth(const Frame& frame, double& depth_mean, double& depth_min)
{
    vector<double> depth_vec;
    depth_vec.reserve(frame.fts_.size());
    depth_min = std::numeric_limits<double>::max();
    for(auto it=frame.fts_.begin(), ite=frame.fts_.end(); it!=ite; ++it)
    {
        if((*it)->point != NULL)
        {
            const double z = frame.w2f((*it)->point->pos_).z();
            depth_vec.push_back(z);
            depth_min = fmin(z, depth_min);
        }
    }
    if(depth_vec.empty())
    {
        HSO_WARN_STREAM("Cannot set scene depth. Frame has no point-observations!");
        return false;
    }
    // std::sort(depth_vec.begin(), depth_vec.end());
    depth_mean = hso::getMedian(depth_vec);
    return true;
}

bool getSceneDistance( const Frame& frame, double& distance_mean)
{
    vector<double> distance_vec;
    distance_vec.reserve(frame.fts_.size());
    for(auto& ft: frame.fts_)
    {
        if(ft->point == NULL) continue;

        const double distance = frame.w2f(ft->point->pos_).norm();
        distance_vec.push_back(distance);
    }
    if(distance_vec.empty())
    {
        HSO_WARN_STREAM("Cannot set scene distance. Frame has no point-observations!");
        return false;
    }
    // std::sort(distance_vec.begin(), distance_vec.end());
    distance_mean = hso::getMedian(distance_vec);
    return true;
}

//TODO: add it to prepareForFeatureDetect()
void createIntegralImage(const cv::Mat& image, float& integralImage)
{
    float sum = 0;
    int num = 0;
    int height = image.rows;
    int weight = image.cols;
    for(int y=8;y<height-8;y++)
        for(int x=8;x<weight-8;x++)
        {
            sum += image.ptr<uchar>(y)[x];
            num++; 
        }

    integralImage = sum/num;
}
} // namespace frame_utils
} // namespace hso
