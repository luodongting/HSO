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

#ifndef HSO_BUNDLE_ADJUSTMENT_H_
#define HSO_BUNDLE_ADJUSTMENT_H_

#include <hso/global.h>

#include <g2o/core/base_binary_edge.h>
#include <g2o/core/base_multi_edge.h>
#include <g2o/types/types_six_dof_expmap.h>
#include <g2o/core/robust_kernel.h>

#include "hso/vikit/math_utils.h"

using namespace g2o;

namespace g2o {
class EdgeSE3ProjectXYZ;
class SparseOptimizer;
class VertexSE3Expmap;
class VertexSBAPointXYZ;
}

namespace hso {

typedef g2o::EdgeSE3ProjectXYZ g2oEdgeSE3;
typedef g2o::VertexSE3Expmap g2oFrameSE3;
typedef g2o::VertexSBAPointXYZ g2oPoint;

class Frame;
class Point;
class Feature;
class Map;
class rdvoEdgeProjectXYZ2UV;
class EdgeProjectID2UV;
class EdgeProjectID2UVEdgeLet;

/// Local, global and 2-view bundle adjustment with g2o
namespace ba {

/// Temporary container to hold the g2o edge with reference to frame and point.
struct EdgeContainerSE3
{
  g2oEdgeSE3*     edge;
  Frame*          frame;
  Feature*        feature;
  bool            is_deleted;
  EdgeContainerSE3(g2oEdgeSE3* e, Frame* frame, Feature* feature) :
    edge(e), frame(frame), feature(feature), is_deleted(false)
  {}
};

struct EdgeContainerEdgelet
{
  rdvoEdgeProjectXYZ2UV*  edge;
  Frame*                  frame;
  Feature*                feature;
  bool                    is_deleted;
  EdgeContainerEdgelet(rdvoEdgeProjectXYZ2UV* e, Frame* frame, Feature* feature) :
    edge(e), frame(frame), feature(feature), is_deleted(false)
  {}
};

//TODO: abstract
struct EdgeContainerID
{
    EdgeProjectID2UV* edge;
    Frame* frame;
    Feature* feature;
    bool is_deleted;

    EdgeContainerID(EdgeProjectID2UV* e, Frame* frame, Feature* feature) :
    edge(e), frame(frame), feature(feature), is_deleted(false) {}
};
//TODO: abstract
struct EdgeContainerIDEdgeLet
{
    EdgeProjectID2UVEdgeLet* edge;
    Frame* frame;
    Feature* feature;
    bool is_deleted;

    EdgeContainerIDEdgeLet(EdgeProjectID2UVEdgeLet* e, Frame* frame, Feature* feature):
    edge(e), frame(frame), feature(feature), is_deleted(false) {}
};


/// Optimize two camera frames and their observed 3D points.
/// Is used after initialization.
void twoViewBA(Frame* frame1, Frame* frame2, double reproj_thresh, Map* map);

/// Local bundle adjustment.
/// Optimizes core_kfs and their observed map points while keeping the
/// neighbourhood fixed.
void localBA(
    Frame* center_kf,
    set<Frame*>* core_kfs,
    Map* map,
    size_t& n_incorrect_edges_1,
    size_t& n_incorrect_edges_2,
    double& init_error,
    double& final_error,
    FramePtr LastKeyFrame = NULL);

/// Global bundle adjustment.
/// Optimizes the whole map. Is currently not used in HSO.
void globalBA(Map* map);

/// Initialize g2o with solver type, optimization strategy and camera model.
void setupG2o(g2o::SparseOptimizer * optimizer);

/// Run the optimization on the provided graph.
void runSparseBAOptimizer(
    g2o::SparseOptimizer* optimizer,
    unsigned int num_iter,
    double& init_error,
    double& final_error);

/// Create a g2o vertice from a keyframe object.
g2oFrameSE3* createG2oFrameSE3(
    Frame* kf,
    size_t id,
    bool fixed);

/// Creates a g2o vertice from a mappoint object.
g2oPoint* createG2oPoint(
    Vector3d pos,
    size_t id,
    bool fixed);

/// Creates a g2o edge between a g2o keyframe and mappoint vertice with the provided measurement.
g2oEdgeSE3* createG2oEdgeSE3(
    g2oFrameSE3* v_kf,
    g2oPoint* v_mp,
    const Vector2d& f_up,
    bool robust_kernel,
    double huber_width,
    double weight = 1);

rdvoEdgeProjectXYZ2UV* createG2oEdgeletSE3(
    g2oFrameSE3* v_kf,
    g2oPoint* v_mp,
    const Vector2d& f_up,
    bool robust_kernel,
    double huber_width,
    double weight = 1,
    const Vector2d& grad = Vector2d(0,0));

void LocalBundleAdjustment(Frame* center_kf,
    set<Frame*>* core_kfs,
    Map* map,
    size_t& n_incorrect_edges_1,
    size_t& n_incorrect_edges_2,
    double& init_error,
    double& final_error);

void initializationBA(Frame* frame1, Frame* frame2, double reproj_thresh, Map* map);

} // namespace ba




class VertexSBAPointID : public BaseVertex<1, double>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    VertexSBAPointID() : BaseVertex<1, double>()
    {}

    virtual bool read(std::istream& is) { return true; }
    virtual bool write(std::ostream& os) const { return true; } 

    virtual void setToOriginImpl() {
        _estimate = 0;
    }

    virtual void oplusImpl(const double* update) {
        _estimate += (*update);
    }
};

class EdgeProjectID2UV : public BaseMultiEdge<2, Vector2d>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    EdgeProjectID2UV()
    {
        // _cam = 0;
        // resizeParameters(1);
        // installParameter(_cam, 0);
    }

    virtual bool read(std::istream& is) {return true;}

    virtual bool write(std::ostream& os) const {return true;}

    void computeError()
    {
        const VertexSBAPointID* point = static_cast<const VertexSBAPointID*>(_vertices[0]);
        const VertexSE3Expmap* host   = static_cast<const VertexSE3Expmap*>(_vertices[1]); 
        const VertexSE3Expmap* target = static_cast<const VertexSE3Expmap*>(_vertices[2]); 
        // const CameraParameters * cam  = static_cast<const CameraParameters *>(parameter(0));

        SE3 Ttw = SE3(target->estimate().rotation(), target->estimate().translation());
        SE3 Thw = SE3(host->estimate().rotation(), host->estimate().translation());
        SE3 Tth = Ttw*Thw.inverse();

        Vector2d obs(_measurement);
        _error = obs - cam_project( Tth * (_fH*(1.0/point->estimate())) );
        
    }

    virtual void linearizeOplus()
    {
        VertexSBAPointID* vp = static_cast<VertexSBAPointID*>(_vertices[0]);
        double idHost = vp->estimate();

        VertexSE3Expmap * vh = static_cast<VertexSE3Expmap *>(_vertices[1]);
        SE3 Thw(vh->estimate().rotation(), vh->estimate().translation());
        VertexSE3Expmap * vt = static_cast<VertexSE3Expmap *>(_vertices[2]);
        SE3 Ttw(vt->estimate().rotation(), vt->estimate().translation());

        SE3 Tth = Ttw*Thw.inverse();
        Vector3d t_th = Tth.translation();
        Matrix3d R_th = Tth.rotation_matrix();
        Vector3d Rf = R_th*_fH;
        Vector3d pTarget = Tth * (_fH*(1.0/idHost));
        Vector2d proj = hso::project2d(pTarget);


        Vector2d Juvdd;
        Juvdd[0] = -(t_th[0] - proj[0]*t_th[2]) / (Rf[2] + idHost*t_th[2]);
        Juvdd[1] = -(t_th[1] - proj[1]*t_th[2]) / (Rf[2] + idHost*t_th[2]);
        _jacobianOplus[0] = Juvdd;

        Matrix<double,2,6> Jpdxi;
        double x = pTarget[0];
        double y = pTarget[1];
        double z = pTarget[2];
        double z_2 = z*z;

        Jpdxi(0,0) = x*y/z_2;
        Jpdxi(0,1) = -(1+(x*x/z_2));
        Jpdxi(0,2) = y/z;
        Jpdxi(0,3) = -1./z;
        Jpdxi(0,4) = 0;
        Jpdxi(0,5) = x/z_2;

        Jpdxi(1,0) = (1+y*y/z_2);
        Jpdxi(1,1) = -x*y/z_2;
        Jpdxi(1,2) = -x/z;
        Jpdxi(1,3) = 0;
        Jpdxi(1,4) = -1./z;
        Jpdxi(1,5) = y/z_2;

        Matrix<double,6,6> adHost;
        adHost = -Tth.Adj();

        Matrix<double,6,6> adTarget;
        adTarget = Matrix<double,6,6>::Identity();

        _jacobianOplus[1] = Jpdxi*adHost;
        _jacobianOplus[2] = Jpdxi*adTarget;
        
    }

    Vector3d _fH;
    void setHostBearing(Vector3d f) 
    {
        _fH = f;
    }

    // CameraParameters * _cam;
    Vector2d cam_project(const Vector3d & trans_xyz)
    {
        return hso::project2d(trans_xyz);
    }
};

class EdgeProjectID2UVEdgeLet : public BaseMultiEdge<1, double>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    EdgeProjectID2UVEdgeLet()
    {
        // _cam = 0;
        // resizeParameters(1);
        // installParameter(_cam, 0);
    }

    virtual bool read(std::istream& is) {return true;}

    virtual bool write(std::ostream& os) const {return true;}

    void computeError()
    {
        const VertexSBAPointID* point = static_cast<const VertexSBAPointID*>(_vertices[0]);
        const VertexSE3Expmap* host = static_cast<const VertexSE3Expmap*>(_vertices[1]); 
        const VertexSE3Expmap* target = static_cast<const VertexSE3Expmap*>(_vertices[2]); 
        // const CameraParameters * cam = static_cast<const CameraParameters *>(parameter(0));

        SE3 Ttw = SE3(target->estimate().rotation(), target->estimate().translation());
        SE3 Thw = SE3(host->estimate().rotation(), host->estimate().translation());
        SE3 Tth = Ttw*Thw.inverse();

        double obs = _measurement;
        // _error = obs - cam->cam_map( Tth * (_fH*(1.0/point->estimate())) );
        _error(0,0) = obs - _normal.transpose()*cam_project( Tth * (_fH*(1.0/point->estimate())) );     
    }

    virtual void linearizeOplus()
    {
        VertexSBAPointID* vp = static_cast<VertexSBAPointID*>(_vertices[0]);
        double idHost = vp->estimate();

        VertexSE3Expmap * vh = static_cast<VertexSE3Expmap *>(_vertices[1]);
        SE3 Thw(vh->estimate().rotation(), vh->estimate().translation());
        VertexSE3Expmap * vt = static_cast<VertexSE3Expmap *>(_vertices[2]);
        SE3 Ttw(vt->estimate().rotation(), vt->estimate().translation());

        SE3 Tth = Ttw*Thw.inverse();
        Vector3d t_th = Tth.translation();
        Matrix3d R_th = Tth.rotation_matrix();
        Vector3d Rf = R_th*_fH;
        Vector3d pTarget = Tth * (_fH*(1.0/idHost));
        Vector2d proj = hso::project2d(pTarget);

        Vector2d Juvdd;
        Juvdd[0] = -(t_th[0] - proj[0]*t_th[2]) / (Rf[2] + idHost*t_th[2]);
        Juvdd[1] = -(t_th[1] - proj[1]*t_th[2]) / (Rf[2] + idHost*t_th[2]);
        _jacobianOplus[0] = _normal.transpose()*Juvdd;

        Matrix<double,2,6> Jpdxi;
        double x = pTarget[0];
        double y = pTarget[1];
        double z = pTarget[2];
        double z_2 = z*z;

        Jpdxi(0,0) = x*y/z_2;
        Jpdxi(0,1) = -(1+(x*x/z_2));
        Jpdxi(0,2) = y/z;
        Jpdxi(0,3) = -1./z;
        Jpdxi(0,4) = 0;
        Jpdxi(0,5) = x/z_2;

        Jpdxi(1,0) = (1+y*y/z_2);
        Jpdxi(1,1) = -x*y/z_2;
        Jpdxi(1,2) = -x/z;
        Jpdxi(1,3) = 0;
        Jpdxi(1,4) = -1./z;
        Jpdxi(1,5) = y/z_2;

        Matrix<double,6,6> adHost;
        adHost = -Tth.Adj();

        Matrix<double,6,6> adTarget;
        adTarget = Matrix<double,6,6>::Identity();

        _jacobianOplus[1] = _normal.transpose()*Jpdxi*adHost;
        _jacobianOplus[2] = _normal.transpose()*Jpdxi*adTarget;
        
    }

    Vector3d _fH;
    void setHostBearing(Vector3d f) 
    {
        _fH = f;
    }

    Vector2d _normal;
    void setTargetNormal(Vector2d n) 
    {
        _normal = n;
    }


    // CameraParameters * _cam;
    Vector2d cam_project(const Vector3d & trans_xyz)
    {
        return hso::project2d(trans_xyz);
    }
};


class rdvoEdgeProjectXYZ2UV : public BaseBinaryEdge<1, double, VertexSBAPointXYZ, VertexSE3Expmap>{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  rdvoEdgeProjectXYZ2UV() : BaseBinaryEdge<1, double, VertexSBAPointXYZ, VertexSE3Expmap>() {
    // _cam = 0;
    // resizeParameters(1);
    // installParameter(_cam, 0);
  }

  virtual bool read(std::istream& is) {return true;}

  virtual bool write(std::ostream& os) const {return true;}

  void computeError()  {
    // const VertexSE3Expmap* v1 = static_cast<const VertexSE3Expmap*>(_vertices[1]);
    // const VertexSBAPointXYZ* v2 = static_cast<const VertexSBAPointXYZ*>(_vertices[0]);
    // const CameraParameters * cam = static_cast<const CameraParameters *>(parameter(0));
    // double obs(_measurement);
    // double est = _grad.transpose() * (cam->cam_map(v1->estimate().map(v2->estimate())));
    // _error(0,0) = obs - est;
  }

  virtual void linearizeOplus() {
    // VertexSE3Expmap * vj = static_cast<VertexSE3Expmap *>(_vertices[1]);
    // SE3Quat T(vj->estimate());
    // VertexSBAPointXYZ* vi = static_cast<VertexSBAPointXYZ*>(_vertices[0]);
    // Vector3d xyz = vi->estimate();
    // Vector3d xyz_trans = T.map(xyz);

    // double x = xyz_trans[0];
    // double y = xyz_trans[1];
    // double z = xyz_trans[2];
    // double z_2 = z*z;

    // const CameraParameters * cam = static_cast<const CameraParameters *>(parameter(0));

    // Matrix<double,2,3,Eigen::ColMajor> tmp;
    // tmp(0,0) = cam->focal_length;
    // tmp(0,1) = 0;
    // tmp(0,2) = -x/z*cam->focal_length;

    // tmp(1,0) = 0;
    // tmp(1,1) = cam->focal_length;
    // tmp(1,2) = -y/z*cam->focal_length;

    // Matrix<double,2,3> jacobianOplusXi = -1./z * tmp * T.rotation().toRotationMatrix();
    // Matrix<double,2,6> jacobianOplusXj;
    // jacobianOplusXj(0,0) =  x*y/z_2 *cam->focal_length;
    // jacobianOplusXj(0,1) = -(1+(x*x/z_2)) *cam->focal_length;
    // jacobianOplusXj(0,2) = y/z *cam->focal_length;
    // jacobianOplusXj(0,3) = -1./z *cam->focal_length;
    // jacobianOplusXj(0,4) = 0;
    // jacobianOplusXj(0,5) = x/z_2 *cam->focal_length;

    // jacobianOplusXj(1,0) = (1+y*y/z_2) *cam->focal_length;
    // jacobianOplusXj(1,1) = -x*y/z_2 *cam->focal_length;
    // jacobianOplusXj(1,2) = -x/z *cam->focal_length;
    // jacobianOplusXj(1,3) = 0;
    // jacobianOplusXj(1,4) = -1./z *cam->focal_length;
    // jacobianOplusXj(1,5) = y/z_2 *cam->focal_length;

    // _jacobianOplusXi = _grad.transpose() * jacobianOplusXi;
    // _jacobianOplusXj = _grad.transpose() * jacobianOplusXj;
  }

  void setGrad(const Vector2d& g) { _grad = g; }

  // CameraParameters * _cam;
  Vector2d _grad;

};
} // namespace hso

#endif // HSO_BUNDLE_ADJUSTMENT_H_
