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


#ifndef HSO_POINT_H_
#define HSO_POINT_H_

#include <boost/noncopyable.hpp>
#include <boost/thread.hpp>
#include <hso/global.h>

#include "hso/vikit/math_utils.h"

namespace g2o {
class VertexSBAPointXYZ; }
typedef g2o::VertexSBAPointXYZ g2oPoint;



namespace hso {

class Feature;
class VertexSBAPointID;

typedef Matrix<double, 2, 3> Matrix23d;

/// A 3D point on the surface of the scene.
class Point : boost::noncopyable
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    enum PointType {TYPE_DELETED, TYPE_TEMPORARY, TYPE_CANDIDATE, TYPE_UNKNOWN, TYPE_GOOD};
    enum FeatureType {FEATURE_GRADIENT, FEATURE_EDGELET, FEATURE_CORNER};

    //!< Counts the number of created points. Used to set the unique id.
    static int point_counter_;

    //!< Unique ID of the point.           
    int id_;

    //!< 3d pos of the point in the world coordinate frame.                 
    Vector3d pos_;

    //!< Surface normal at point.                     
    Vector3d normal_; 

    //!< Inverse covariance matrix of normal estimation.                 
    Matrix3d normal_information_;

    //!< Flag whether the surface normal was estimated or not.      
    bool normal_set_;

    //!< References to keyframes which observe the point.
    list<Feature*> obs_;

    //!< Number of obervations: Keyframes AND successful reprojections in intermediate frames.                     
    size_t n_obs_;

    //!< Temporary pointer to the point-vertex in g2o during bundle adjustment.
    g2oPoint* v_pt_;
    
    //!< Timestamp of last publishing.
    int last_published_ts_;

    //!< Flag for the reprojection: don't reproject a pt twice.       
    int last_projected_kf_id_;

    //!< Quality of the point.
    PointType type_;

    //!< Number of failed reprojections. Used to assess the quality of the point.
    int n_failed_reproj_;

    //!< Number of succeeded reprojections. Used to assess the quality of the point.         
    int n_succeeded_reproj_;

    //!< Timestamp of last point optimization      
    int last_structure_optim_;

    //!< Feature type of the point.    
    FeatureType ftr_type_; 

    // //!< Last nonkeyframe id the point have oberserved
    // //!< Used in matcher-findMatchDirect()  set in pose_optimizer
    // int last_obs_keyframeId_=-1;
    // Feature* last_nonkeyframe_ft_;


    int seedStates_;

    // only for temp point
    bool isBad_;  

    double idist_;
    Feature* hostFeature_;
    VertexSBAPointID* vPoint_;
    size_t nBA_;

    //Viewer
    float color_ = 128;


    /// photomatric calibration parameters
    Feature* m_last_feature = NULL;
    int m_last_feature_kf_id = -1;
    vector<double> m_rad_est; //[TODO] Radiation estimate, fixed at exposure estimate, changed in radiation optimization


    Point(const Vector3d& pos);
    Point(const Vector3d& pos, Feature* ftr);
    ~Point();

    /// Add a reference to a frame.
    void addFrameRef(Feature* ftr);

    /// Remove reference to a frame.
    bool deleteFrameRef(Frame* frame);

    /// Initialize point normal. The inital estimate will point towards the frame.
    void initNormal();

    /// Check whether mappoint has reference to a frame.
    Feature* findFrameRef(Frame* frame);

    /// Get Frame with similar viewpoint.
    bool getCloseViewObs(const Vector3d& pos, Feature*& obs) const;

    /// Get number of observations.
    inline size_t nRefs() const { return obs_.size(); }

    /// Optimize point position through minimizing the reprojection error.
    void optimize(const size_t n_iter);
    void optimizeLM(const size_t n_iter); 
    void optimizeID(const size_t n_iter);


    /// Jacobian of point projection on unit plane (focal length = 1) in frame (f).
    inline static void jacobian_xyz2uv(
        const Vector3d& p_in_f,const Matrix3d& R_f_w, Matrix23d& point_jac)
    {
        const double z_inv = 1.0/p_in_f[2];
        const double z_inv_sq = z_inv*z_inv;
        point_jac(0, 0) = z_inv;
        point_jac(0, 1) = 0.0;
        point_jac(0, 2) = -p_in_f[0] * z_inv_sq;
        point_jac(1, 0) = 0.0;
        point_jac(1, 1) = z_inv;
        point_jac(1, 2) = -p_in_f[1] * z_inv_sq;
        point_jac = - point_jac * R_f_w;
    }


    inline static void jacobian_id2uv(
        const Vector3d& p_in_f, const SE3& Tth, const double idH, const Vector3d& fH, Vector2d& point_jac)
    {
        Vector2d proj = hso::project2d(p_in_f);
        Vector3d t_th = Tth.translation();
        Matrix3d R_th = Tth.rotation_matrix();
        Vector3d Rf = R_th*fH;

        point_jac[0] = -(t_th[0] - proj[0]*t_th[2]) / (Rf[2] + t_th[2]*idH);
        point_jac[1] = -(t_th[1] - proj[1]*t_th[2]) / (Rf[2] + t_th[2]*idH);
    }

    // boost::mutex readMutex_;
};

} // namespace hso

#endif // HSO_POINT_H_
