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


#ifndef HSO_FRAME_H_
#define HSO_FRAME_H_

#include <sophus/se3.h>
#include <boost/noncopyable.hpp>
#include <boost/thread.hpp>
#include <hso/global.h>

#include "hso/camera.h"

namespace g2o {
class VertexSE3Expmap;
}
typedef g2o::VertexSE3Expmap g2oFrameSE3;

namespace hso {

class Point;
// class PhotomatricCalibration;
struct Feature;

typedef list<Feature*> Features;
typedef vector<cv::Mat> ImgPyr;


#define FRAME_GRID_ROWS 48
#define FRAME_GRID_COLS 64

/// A frame saves the image, the associated features and the estimated pose.
class Frame : boost::noncopyable
{
public:

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    // Counts the number of created frames. Used to set the unique id.
    static int frame_counter_;         
    static int keyFrameCounter_;

    // Unique id of the frame.
    int id_;                    
    int keyFrameId_;

    // Timestamp of when the image was recorded.
    double timestamp_; 

    string m_timestamp_s;

    // Camera model.
    hso::AbstractCamera* cam_;  

    // Transform (f)rame from (w)orld.
    Sophus::SE3 T_f_w_;

    // Covariance.                 
    Matrix<double, 6, 6> Cov_;

    // Image Pyramid.                   
    ImgPyr img_pyr_;

    // List of features in the image.               
    Features fts_;
    boost::mutex m_fts_mutex;

    // Five features and associated 3D points which are used to detect if two frames have overlapping field of view.                   
    vector<Feature*> key_pts_; 

    // Was this frames selected as keyframe?
    bool is_keyframe_;

    // Temporary pointer to the g2o node object of the keyframe.           
    g2oFrameSE3* v_kf_;

    //!< Timestamp of last publishing.                  
    int last_published_ts_;     
    
    ImgPyr grad_pyr_, sobelX_, sobelY_, canny_;

    float integralImage_;

    std::vector<Frame*> connectedKeyFrames;
    int lastReprojectFrameId_;

    // Photometric calibration parameters
    // PhotomatricCalibration* m_pc=NULL;
    ImgPyr m_pyr_raw;
    double m_exposure_time = -1;
    bool m_exposure_finish = false;
    bool m_added=false;
    bool m_kf_pc=false;

    FramePtr m_last_frame;

    // Average Gradient of Image
    float gradMean_;

    // residual after pose refinement used in depth filter
    float m_error_in_px = 1.f;


    int m_n_inliers;

    // correct:0-4  raw:0-2
    // void photometricallyCorrectPyramid(const cv::Mat& img_level_0, ImgPyr& pyr_correct, ImgPyr& pyr_raw, int n_levels);


    Frame(hso::AbstractCamera* cam, const cv::Mat& img, double timestamp);
    ~Frame();

    /// Initialize new frame and create image pyramid.
    void initFrame(const cv::Mat& img);

    /// Select this frame as keyframe.
    void setKeyframe();

    /// Add a feature to the image
    void addFeature(Feature* ftr);

    /// Thread safe !!!   used in radiance self cailbration 
    void getFeaturesCopy(Features& list_copy);

    /// The KeyPoints are those five features which are closest to the 4 image corners
    /// and to the center and which have a 3D point assigned. These points are used
    /// to quickly check whether two frames have overlapping field of view.
    void setKeyPoints();

    /// Check if we can select five better key-points.
    void checkKeyPoints(Feature* ftr);

    /// If a point is deleted, we must remove the corresponding key-point.
    void removeKeyPoint(Feature* ftr);

    /// Return number of point observations.
    inline size_t nObs() const { return fts_.size(); }

    /// Check if a point in (w)orld coordinate frame is visible in the image.
    bool isVisible(const Vector3d& xyz_w) const;

    /// Full resolution image stored in the frame.
    inline const cv::Mat& img() const { return img_pyr_[0]; }

    /// Was this frame selected as keyframe?
    inline bool isKeyframe() const { return is_keyframe_; }

    /// Transforms point coordinates in world-frame (w) to camera pixel coordinates (c).
    inline Vector2d w2c(const Vector3d& xyz_w) const { return cam_->world2cam( T_f_w_ * xyz_w ); }

    /// Transforms pixel coordinates (c) to frame unit sphere coordinates (f).
    inline Vector3d c2f(const Vector2d& px) const { return cam_->cam2world(px[0], px[1]); }

    /// Transforms pixel coordinates (c) to frame unit sphere coordinates (f).
    inline Vector3d c2f(const double x, const double y) const { return cam_->cam2world(x, y); }

    /// Transforms point coordinates in world-frame (w) to camera-frams (f).
    inline Vector3d w2f(const Vector3d& xyz_w) const { return T_f_w_ * xyz_w; }

    /// Transforms point from frame unit sphere (f) frame to world coordinate frame (w).
    inline Vector3d f2w(const Vector3d& f) const { return T_f_w_.inverse() * f; }

    /// Projects Point from unit sphere (f) in camera pixels (c).
    inline Vector2d f2c(const Vector3d& f) const { return cam_->world2cam( f ); }

    /// Return the pose of the frame in the (w)orld coordinate frame.
    inline Vector3d pos() const { return T_f_w_.inverse().translation(); }

    /// Frame jacobian for projection of 3D point in (f)rame coordinate to
    /// unit plane coordinates uv (focal length = 1).
    inline static void jacobian_xyz2uv(const Vector3d& xyz_in_f, Matrix<double,2,6>& J)
    {
        const double x = xyz_in_f[0];
        const double y = xyz_in_f[1];
        const double z_inv = 1./xyz_in_f[2];
        const double z_inv_2 = z_inv*z_inv;

        J(0,0) = -z_inv;              // -1/z
        J(0,1) = 0.0;                 // 0
        J(0,2) = x*z_inv_2;           // x/z^2
        J(0,3) = y*J(0,2);            // x*y/z^2
        J(0,4) = -(1.0 + x*J(0,2));   // -(1.0 + x^2/z^2)
        J(0,5) = y*z_inv;             // y/z

        J(1,0) = 0.0;                 // 0
        J(1,1) = -z_inv;              // -1/z
        J(1,2) = y*z_inv_2;           // y/z^2
        J(1,3) = 1.0 + y*J(1,2);      // 1.0 + y^2/z^2
        J(1,4) = -J(0,3);             // -x*y/z^2
        J(1,5) = -x*z_inv;            // x/z
    }


    void prepareForFeatureDetect();

    void finish();
};


/// Some helper functions for the frame object.
namespace frame_utils {

    /// Creates an image pyramid of half-sampled images.
    void createImgPyramid(const cv::Mat& img_level_0, int n_levels, ImgPyr& pyr);

    /// Get the average depth of the features in the image.
    bool getSceneDepth(const Frame& frame, double& depth_mean, double& depth_min);

    /// Get the average distance of the features in the image.
    bool getSceneDistance(const Frame& frame, double& distance_mean);

    /// Creates an image gradients pyramid.
    void createImgGrad(const ImgPyr& pyr_img, ImgPyr& scharr, int n_levels);


    void createIntegralImage(const cv::Mat& image, float& integralImage);

} // namespace frame_utils
} // namespace hso

#endif // HSO_FRAME_H_
