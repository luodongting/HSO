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

#include <boost/thread.hpp>

#include <g2o/core/sparse_optimizer.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/solver.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/linear_solver_eigen.h>

#include <g2o/types/types_six_dof_expmap.h>

#include <hso/bundle_adjustment.h>
#include <hso/frame.h>
#include <hso/feature.h>
#include <hso/point.h>
#include <hso/config.h>
#include <hso/map.h>
#include <hso/matcher.h>

#include "hso/vikit/math_utils.h"

#define SCHUR_TRICK 1

namespace hso {
namespace ba {

void twoViewBA(
    Frame* frame1,
    Frame* frame2,
    double reproj_thresh,
    Map* map)
{
  // scale reprojection threshold in pixels to unit plane
  reproj_thresh /= frame1->cam_->errorMultiplier2();

  // init g2o
  g2o::SparseOptimizer optimizer;
  setupG2o(&optimizer);

  list<EdgeContainerSE3> edges;
  size_t v_id = 0;

  // New Keyframe Vertex 1: This Keyframe is set to fixed!
  g2oFrameSE3* v_frame1 = createG2oFrameSE3(frame1, v_id++, true);
  optimizer.addVertex(v_frame1);

  // New Keyframe Vertex 2
  g2oFrameSE3* v_frame2 = createG2oFrameSE3(frame2, v_id++, false);
  optimizer.addVertex(v_frame2);

  // Create Point Vertices
  for(Features::iterator it_ftr=frame1->fts_.begin(); it_ftr!=frame1->fts_.end(); ++it_ftr)
  {
    Point* pt = (*it_ftr)->point;
    if(pt == NULL) continue;
    
    g2oPoint* v_pt = createG2oPoint(pt->pos_, v_id++, false);
    optimizer.addVertex(v_pt);
    pt->v_pt_ = v_pt;
    g2oEdgeSE3* e = createG2oEdgeSE3(v_frame1, v_pt, hso::project2d((*it_ftr)->f), true, reproj_thresh*Config::lobaRobustHuberWidth());
    optimizer.addEdge(e);
    edges.push_back(EdgeContainerSE3(e, frame1, *it_ftr)); // TODO feature now links to frame, so we can simplify edge container!

    // find at which index the second frame observes the point
    Feature* ftr_frame2 = pt->findFrameRef(frame2);
    e = createG2oEdgeSE3(v_frame2, v_pt, hso::project2d(ftr_frame2->f), true, reproj_thresh*Config::lobaRobustHuberWidth());
    optimizer.addEdge(e);
    edges.push_back(EdgeContainerSE3(e, frame2, ftr_frame2));
  }

  // Optimization
  double init_error, final_error;
  runSparseBAOptimizer(&optimizer, 30, init_error, final_error);
  printf("2-View BA: Error before/after = %f / %f\n", init_error, final_error);

  // Update Keyframe Positions
  frame1->T_f_w_.rotation_matrix() = v_frame1->estimate().rotation().toRotationMatrix();
  frame1->T_f_w_.translation() = v_frame1->estimate().translation();
  frame2->T_f_w_.rotation_matrix() = v_frame2->estimate().rotation().toRotationMatrix();
  frame2->T_f_w_.translation() = v_frame2->estimate().translation();

  // Update Mappoint Positions
  for(Features::iterator it=frame1->fts_.begin(); it!=frame1->fts_.end(); ++it)
  {
    if((*it)->point == NULL)
     continue;
    (*it)->point->pos_ = (*it)->point->v_pt_->estimate();
    (*it)->point->v_pt_ = NULL;
  }

  // Find Mappoints with too large reprojection error
  const double reproj_thresh_squared = reproj_thresh*reproj_thresh;
  size_t n_incorrect_edges = 0;
  for(list<EdgeContainerSE3>::iterator it_e = edges.begin(); it_e != edges.end(); ++it_e)
    if(it_e->edge->chi2() > reproj_thresh_squared)
    {
      if(it_e->feature->point != NULL)
      {
        map->safeDeletePoint(it_e->feature->point);
        it_e->feature->point = NULL;
      }
      ++n_incorrect_edges;
    }

  printf("2-View BA: Wrong edges =  %zu\n", n_incorrect_edges);
}

void localBA(
    Frame* center_kf,
    set<Frame*>* core_kfs,
    Map* map,
    size_t& n_incorrect_edges_1,
    size_t& n_incorrect_edges_2,
    double& init_error,
    double& final_error,
    FramePtr LastKeyFrame)
{

    // init g2o
    g2o::SparseOptimizer optimizer;
    setupG2o(&optimizer);

    list<EdgeContainerSE3> edges;
    list<EdgeContainerEdgelet> edgelets;
    set<Point*> mps;
    list<Frame*> neib_kfs;
    size_t v_id = 0;
    size_t n_mps = 0;
    size_t n_fix_kfs = 0;
    size_t n_var_kfs = 1;
    size_t n_edges = 0;
    n_incorrect_edges_1 = 0;
    n_incorrect_edges_2 = 0;

    // Add all core keyframes
    for(set<Frame*>::iterator it_kf = core_kfs->begin(); it_kf != core_kfs->end(); ++it_kf)
    {
        g2oFrameSE3* v_kf;
        if((*it_kf)->id_ == 0)
            v_kf = createG2oFrameSE3((*it_kf), v_id++, true);
        else
            v_kf = createG2oFrameSE3((*it_kf), v_id++, false);

        (*it_kf)->v_kf_ = v_kf;
        ++n_var_kfs;
        assert(optimizer.addVertex(v_kf));

        // all points that the core keyframes observe are also optimized:
        for(Features::iterator it_pt=(*it_kf)->fts_.begin(); it_pt!=(*it_kf)->fts_.end(); ++it_pt)
        {
            if((*it_pt)->point != NULL)
            {
                // assert((*it_pt)->point->type_ != Point::TYPE_CANDIDATE);
                mps.insert((*it_pt)->point);
            }
        }
    }
    // cout << "OK1" << endl;
    // Now go throug all the points and add a measurement. Add a fixed neighbour
    // Keyframe if it is not in the set of core kfs
    double reproj_thresh_2 = Config::lobaThresh() / center_kf->cam_->errorMultiplier2();
    double reproj_thresh_1 = Config::poseOptimThresh() / center_kf->cam_->errorMultiplier2();
    double reproj_thresh_1_squared = reproj_thresh_1*reproj_thresh_1;
    for(set<Point*>::iterator it_pt = mps.begin(); it_pt!=mps.end(); ++it_pt)
    {
        // Create point vertex
        g2oPoint* v_pt = createG2oPoint((*it_pt)->pos_, v_id++, false);
        (*it_pt)->v_pt_ = v_pt;
        assert(optimizer.addVertex(v_pt));
        ++n_mps;

        // Add edges
        list<Feature*>::iterator it_obs=(*it_pt)->obs_.begin();
        while(it_obs!=(*it_pt)->obs_.end())
        {
            Vector2d error = hso::project2d((*it_obs)->f) - hso::project2d((*it_obs)->frame->w2f((*it_pt)->pos_));

            if((*it_obs)->frame->v_kf_ == NULL)
            {
                // frame does not have a vertex yet -> it belongs to the neib kfs and
                // is fixed. create one:
                g2oFrameSE3* v_kf = createG2oFrameSE3((*it_obs)->frame, v_id++, true);
                (*it_obs)->frame->v_kf_ = v_kf;
                ++n_fix_kfs;
                assert(optimizer.addVertex(v_kf));
                neib_kfs.push_back((*it_obs)->frame);
            }

            // create edge
            if((*it_obs)->type != Feature::EDGELET)
            {
                g2oEdgeSE3* e = createG2oEdgeSE3(
                    (*it_obs)->frame->v_kf_, v_pt, hso::project2d((*it_obs)->f), true,
                    reproj_thresh_2*Config::lobaRobustHuberWidth(), 1.0 / (1<<(*it_obs)->level));

                edges.push_back(EdgeContainerSE3(e, (*it_obs)->frame, *it_obs));
                assert(optimizer.addEdge(e));
            }
            else
            {
                rdvoEdgeProjectXYZ2UV* e = createG2oEdgeletSE3(
                    (*it_obs)->frame->v_kf_, v_pt, hso::project2d((*it_obs)->f), true,
                    reproj_thresh_2*Config::lobaRobustHuberWidth(), 1.0 / (1<<(*it_obs)->level), (*it_obs)->grad);

                edgelets.push_back(EdgeContainerEdgelet(e, (*it_obs)->frame, *it_obs));
                assert(optimizer.addEdge(e));
            }
            ++n_edges;
            ++it_obs;
        }
    }
    // cout << "OK2" << endl;
    // structure only
    // g2o::StructureOnlySolver<3> structure_only_ba;
    // g2o::OptimizableGraph::VertexContainer points;
    // for (g2o::OptimizableGraph::VertexIDMap::const_iterator it = optimizer.vertices().begin(); it != optimizer.vertices().end(); ++it)
    // {
    //   g2o::OptimizableGraph::Vertex* v = static_cast<g2o::OptimizableGraph::Vertex*>(it->second);
    //     if (v->dimension() == 3 && v->edges().size() >= 2)
    //       points.push_back(v);
    // }
    // structure_only_ba.calc(points, 10);

    // Optimization
    if(Config::lobaNumIter() > 0)
        runSparseBAOptimizer(&optimizer, Config::lobaNumIter(), init_error, final_error);

    // cout << "OK3" << endl;
    // Update Keyframes
    for(set<Frame*>::iterator it = core_kfs->begin(); it != core_kfs->end(); ++it)
    {
        (*it)->T_f_w_ = SE3( (*it)->v_kf_->estimate().rotation(), (*it)->v_kf_->estimate().translation());
        (*it)->v_kf_ = NULL;


        // if(LastKeyFrame != NULL && (*it)->id_ == LastKeyFrame->id_)
        //     for(auto ite = LastKeyFrame->featureChild_.begin(); ite != LastKeyFrame->featureChild_.end(); ++ite)
        //     {
        //         if((*ite)->point == NULL) continue;

        //         if((*ite)->point->v_pt_ == NULL && (*ite)->point->type_ == Point::TYPE_CANDIDATE)
        //         {
        //             (*ite)->point->pos_ = LastKeyFrame->T_f_w_.inverse() * ((*ite)->f * (1.0/(*ite)->point->idist_));
        //         }
        //     }
    }

    for(list<Frame*>::iterator it = neib_kfs.begin(); it != neib_kfs.end(); ++it)
        (*it)->v_kf_ = NULL;

    // Update Mappoints
    for(set<Point*>::iterator it = mps.begin(); it != mps.end(); ++it)
    {
        (*it)->pos_ = (*it)->v_pt_->estimate();
        (*it)->v_pt_ = NULL;
    }
    // cout << "OK4" << endl;
    // Remove Measurements with too large reprojection error
    double reproj_thresh_2_squared = reproj_thresh_2*reproj_thresh_2;
    for(list<EdgeContainerSE3>::iterator it = edges.begin(); it != edges.end(); ++it)
    {
        if(it->feature->point == NULL)
            continue;
        // We only delete the temprorary point in reprojector
        if(it->edge->chi2() > reproj_thresh_2_squared) //*(1<<it->feature_->level))
        {
            if(it->feature->point->type_ == Point::TYPE_TEMPORARY)
            {
                it->feature->point->isBad_ = true;
                continue;
            }
            // assert(it->feature->point->type_ != Point::TYPE_TEMPORARY &&
            //        it->feature->point->type_ != Point::TYPE_CANDIDATE &&
            //        it->feature->point->type_ != Point::TYPE_DELETED );

            // cout << "OK0.1" << endl;
            map->removePtFrameRef(it->frame, it->feature);
            // cout << "OK0.2" << endl;
            ++n_incorrect_edges_2;
        }
    }
    // for(list<EdgeContainerEdgelet>::iterator it = edgelets.begin(); it != edgelets.end(); ++it)
    // {
    //   if(it->edge->chi2() > reproj_thresh_2_squared && it->feature->point->type_ != Point::TYPE_TEMPORARY) //*(1<<it->feature_->level))
    //   {
    //     map->removePtFrameRef(it->frame, it->feature);
    //     ++n_incorrect_edges_2;
    //   }
    // }
    // cout << "OK5" << endl;
    // TODO: delete points and edges!
    init_error = sqrt(init_error)*center_kf->cam_->errorMultiplier2();
    final_error = sqrt(final_error)*center_kf->cam_->errorMultiplier2();
}

void setupG2o(g2o::SparseOptimizer * optimizer)
{
  // TODO: What's happening with all this HEAP stuff? Memory Leak?
  optimizer->setVerbose(false);

#if SCHUR_TRICK
  // solver
  g2o::BlockSolver_6_3::LinearSolverType* linearSolver;
  linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolver_6_3::PoseMatrixType>();
  //linearSolver = new g2o::LinearSolverCSparse<g2o::BlockSolver_6_3::PoseMatrixType>();

  g2o::BlockSolver_6_3 * solver_ptr = new g2o::BlockSolver_6_3(linearSolver);
  g2o::OptimizationAlgorithmLevenberg * solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
#else
  g2o::BlockSolverX::LinearSolverType * linearSolver;
  linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolverX::PoseMatrixType>();
  //linearSolver = new g2o::LinearSolverCSparse<g2o::BlockSolverX::PoseMatrixType>();
  g2o::BlockSolverX * solver_ptr = new g2o::BlockSolverX(linearSolver);
  g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
#endif

  solver->setMaxTrialsAfterFailure(5);
  optimizer->setAlgorithm(solver);

  // // setup camera
  // g2o::CameraParameters * cam_params = new g2o::CameraParameters(1.0, Vector2d(0.,0.), 0.);
  // cam_params->setId(0);
  // if (!optimizer->addParameter(cam_params)) {
  //   assert(false);
  // }
}

void
runSparseBAOptimizer(g2o::SparseOptimizer* optimizer,
                     unsigned int num_iter,
                     double& init_error, double& final_error)
{
    optimizer->initializeOptimization();
    optimizer->computeActiveErrors();
    init_error = optimizer->activeChi2();
    optimizer->optimize(num_iter);
    final_error = optimizer->activeChi2();
}

g2oFrameSE3*
createG2oFrameSE3(Frame* frame, size_t id, bool fixed)
{
    g2oFrameSE3* v = new g2oFrameSE3();
    v->setId(id);
    v->setFixed(fixed);

    v->setEstimate(g2o::SE3Quat(frame->T_f_w_.unit_quaternion(), frame->T_f_w_.translation()));
    return v;
}

g2oPoint*
createG2oPoint(Vector3d pos,
               size_t id,
               bool fixed)
{
  g2oPoint* v = new g2oPoint();
  v->setId(id);
#if SCHUR_TRICK
  v->setMarginalized(true);
#endif
  v->setFixed(fixed);
  v->setEstimate(pos);
  return v;
}

g2oEdgeSE3*
createG2oEdgeSE3( g2oFrameSE3* v_frame,
                  g2oPoint* v_point,
                  const Vector2d& f_up,
                  bool robust_kernel,
                  double huber_width,
                  double weight)
{
  g2oEdgeSE3* e = new g2oEdgeSE3();
  e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(v_point));
  e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(v_frame));
  e->setMeasurement(f_up);
  // e->information() = weight * Eigen::Matrix2d::Identity(2,2);
  e->setInformation(Eigen::Matrix2d::Identity()*weight);
  g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber();      // TODO: memory leak
  rk->setDelta(huber_width);
  e->setRobustKernel(rk);
  e->setParameterId(0, 0); //old: e->setId(v_point->id());
  return e;
}

rdvoEdgeProjectXYZ2UV* 
createG2oEdgeletSE3(g2oFrameSE3* v_frame,
                    g2oPoint* v_point,
                    const Vector2d& f_up,
                    bool robust_kernel,
                    double huber_width,
                    double weight,
                    const Vector2d& grad)
{
  rdvoEdgeProjectXYZ2UV* e = new rdvoEdgeProjectXYZ2UV();
  e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(v_point));
  e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(v_frame));
  e->setMeasurement(grad.transpose() * f_up);
  e->information() = weight * Eigen::Matrix<double,1,1>::Identity(1,1);
  g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber();      
  rk->setDelta(huber_width);
  e->setRobustKernel(rk);
  e->setParameterId(0, 0); 
  e->setGrad(grad);
  return e;
}

void initializationBA(Frame* frame1, Frame* frame2, double reproj_thresh, Map* map)
{
    // scale reprojection threshold in pixels to unit plane
    reproj_thresh /= frame1->cam_->errorMultiplier2();

    // init g2o
    g2o::SparseOptimizer optimizer;
    optimizer.setVerbose(false);

    g2o::BlockSolverX::LinearSolverType * linearSolver;
    linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolverX::PoseMatrixType>();
    g2o::BlockSolverX * solver_ptr = new g2o::BlockSolverX(linearSolver);
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);

    solver->setMaxTrialsAfterFailure(5);
    optimizer.setAlgorithm(solver);

    // g2o::CameraParameters * cam_params = new g2o::CameraParameters(1.0, Vector2d(0.,0.), 0.);
    // cam_params->setId(0);
    // if(!optimizer.addParameter(cam_params)) assert(false);


    list<EdgeContainerID> edges;
    size_t v_id = 0;

    // New Keyframe Vertex 1: This Keyframe is set to fixed!
    g2oFrameSE3* v_frame1 = createG2oFrameSE3(frame1, v_id++, true);
    optimizer.addVertex(v_frame1);

    // New Keyframe Vertex 2
    g2oFrameSE3* v_frame2 = createG2oFrameSE3(frame2, v_id++, false);
    optimizer.addVertex(v_frame2);

    // Create Point Vertices
    for(Features::iterator it_ftr=frame1->fts_.begin(); it_ftr!=frame1->fts_.end(); ++it_ftr)
    {
        Point* pt = (*it_ftr)->point;
        // if(pt == NULL) continue;
        assert(pt != NULL);

        VertexSBAPointID* vPoint = new VertexSBAPointID();
        vPoint->setId(v_id++);
        vPoint->setFixed(false);
        vPoint->setEstimate(pt->idist_);
        pt->vPoint_ = vPoint;
        pt->nBA_++;
        optimizer.addVertex(vPoint);

        EdgeProjectID2UV* edge = new EdgeProjectID2UV();
        edge->resize(3);
        edge->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(vPoint));
        edge->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(v_frame1));
        edge->setVertex(2, dynamic_cast<g2o::OptimizableGraph::Vertex*>(v_frame2));

        edge->setHostBearing(pt->hostFeature_->f);


        // Feature* ftr_frame2 = pt->findFrameRef(frame2);
        Feature* ftr_frame2 = NULL;
        for(auto ite = frame2->fts_.begin(); ite != frame2->fts_.end(); ++ite)
            if((*ite)->point == pt) 
            {
                ftr_frame2 = *ite;
                break;
            }
        // assert(ftr_frame2 != NULL);
        if(ftr_frame2 == NULL) continue;
        edge->setMeasurement(hso::project2d(ftr_frame2->f));


        edge->setInformation(Eigen::Matrix2d::Identity() * (1.0/(1<<(*it_ftr)->level)));
        g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber();      
        rk->setDelta((1.0/frame1->cam_->errorMultiplier2())*Config::lobaRobustHuberWidth());
        edge->setRobustKernel(rk);
        edge->setParameterId(0, 0); 


        edges.push_back(EdgeContainerID(edge, (*it_ftr)->frame, *it_ftr));  
        optimizer.addEdge(edge);
    }

    // Optimization
    double init_error, final_error;
    runSparseBAOptimizer(&optimizer, 50, init_error, final_error);
    printf("2-View BA: Error before/after = %f / %f\n", init_error, final_error);

    // Update Keyframe Positions
    frame1->T_f_w_.rotation_matrix() = v_frame1->estimate().rotation().toRotationMatrix();
    frame1->T_f_w_.translation() = v_frame1->estimate().translation();
    frame2->T_f_w_.rotation_matrix() = v_frame2->estimate().rotation().toRotationMatrix();
    frame2->T_f_w_.translation() = v_frame2->estimate().translation();

    // Update Mappoint Positions
    for(Features::iterator it=frame1->fts_.begin(); it!=frame1->fts_.end(); ++it)
    {
        // if((*it)->point == NULL) continue;
        assert((*it)->point != NULL);

        Point* pt = (*it)->point;
        pt->idist_ = pt->vPoint_->estimate();
        pt->vPoint_ = NULL;

        //update position
        pt->pos_ = pt->hostFeature_->f*(1.0/pt->idist_);
    }

    // Find Mappoints with too large reprojection error
    const double reproj_thresh_squared = reproj_thresh*reproj_thresh;
    size_t n_incorrect_edges = 0;
    for(list<EdgeContainerID>::iterator it_e = edges.begin(); it_e != edges.end(); ++it_e)
        if(it_e->edge->chi2() > reproj_thresh_squared)
        {
            if(it_e->feature->point != NULL)
            {
                map->safeDeletePoint(it_e->feature->point);
                it_e->feature->point = NULL;
            }
            ++n_incorrect_edges;
        }

    printf("2-View BA: Wrong edges =  %zu\n", n_incorrect_edges);
}


void LocalBundleAdjustment(Frame* center_kf,
    set<Frame*>* core_kfs,
    Map* map,
    size_t& n_incorrect_edges_1,
    size_t& n_incorrect_edges_2,
    double& init_error,
    double& final_error)
{
    g2o::SparseOptimizer optimizer;
    optimizer.setVerbose(false);

    g2o::BlockSolverX::LinearSolverType* linearSolver;
    linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolverX::PoseMatrixType>();
    g2o::BlockSolverX* solver_ptr = new g2o::BlockSolverX(linearSolver);
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    solver->setMaxTrialsAfterFailure(5);
    
    optimizer.setAlgorithm(solver);




    list<EdgeContainerID> edges;
    list<EdgeContainerIDEdgeLet> edgeLets;

    set<Point*> mps;
    list<Frame*> neib_kfs;
    list<Frame*> hostKeyFrame;
    size_t v_id = 0;
    size_t n_mps = 0;
    size_t n_fix_kfs = 0;
    size_t n_var_kfs = 1;
    size_t n_edges = 0;
    n_incorrect_edges_1 = 0;
    n_incorrect_edges_2 = 0;

    for(set<Frame*>::iterator it_kf = core_kfs->begin(); it_kf != core_kfs->end(); ++it_kf)
    {
        g2oFrameSE3* v_kf;
        if((*it_kf)->id_ == 0 || (*it_kf)->keyFrameId_+20 < center_kf->keyFrameId_)
            v_kf = createG2oFrameSE3((*it_kf), v_id++, true);
        else
            v_kf = createG2oFrameSE3((*it_kf), v_id++, false);

        (*it_kf)->v_kf_ = v_kf;
        ++n_var_kfs;
        assert(optimizer.addVertex(v_kf));

        // all points that the core keyframes observe are also optimized:
        for(Features::iterator it_pt=(*it_kf)->fts_.begin(); it_pt!=(*it_kf)->fts_.end(); ++it_pt)
        {
            if((*it_pt)->point == NULL) continue;



            assert((*it_pt)->point->type_ != Point::TYPE_CANDIDATE);
            mps.insert((*it_pt)->point);
        }
    }

    
    
    vector<float> errors_pt, errors_ls, errors_tt;



    int n_pt=0, n_ls=0, n_tt=0;

    for(set<Point*>::iterator it_pt = mps.begin(); it_pt!=mps.end(); ++it_pt)
    {
        Frame* host_frame = (*it_pt)->hostFeature_->frame;
        Vector3d pHost = (*it_pt)->hostFeature_->f * (1.0/(*it_pt)->idist_);
        for(auto it_ft = (*it_pt)->obs_.begin(); it_ft != (*it_pt)->obs_.end(); ++it_ft)
        {
            // skip host frame
            if((*it_ft)->frame->id_ == host_frame->id_) continue;
            assert((*it_ft)->point == *it_pt);

            SE3 Tth = (*it_ft)->frame->T_f_w_ * host_frame->T_f_w_.inverse();
            Vector3d pTarget = Tth * pHost;
            Vector2d e = hso::project2d((*it_ft)->f) - hso::project2d(pTarget);
            e *= 1.0 / (1<<(*it_ft)->level);

            // float e_norm = e.norm();

            if((*it_ft)->type == Feature::EDGELET)
            {
                errors_ls.push_back(fabs((*it_ft)->grad.transpose()*e));
                n_ls++;


            }
            else
            {
                errors_pt.push_back(e.norm());
                n_pt++;


            }


        }
    }


    // calc huber threshold
    float huber_corner,huber_edge;
    if(!errors_pt.empty() && !errors_ls.empty())
    {
        huber_corner = 1.4826*hso::getMedian(errors_pt);
        huber_edge = 1.4826*hso::getMedian(errors_ls);
    }
    else if(errors_pt.empty() && !errors_ls.empty())
    {
        huber_corner = 1.0 / center_kf->cam_->errorMultiplier2();
        huber_edge = 1.4826*hso::getMedian(errors_ls);
    }
    else if(!errors_pt.empty() && errors_ls.empty())
    {
        huber_corner = 1.4826*hso::getMedian(errors_pt);
        huber_edge   = 0.5 / center_kf->cam_->errorMultiplier2();
    }
    else
    {
    }
    








    for(set<Point*>::iterator it_pt = mps.begin(); it_pt!=mps.end(); ++it_pt)
    {
        VertexSBAPointID* vPoint = new VertexSBAPointID();
        vPoint->setId(v_id++);
        vPoint->setFixed(false);
        vPoint->setEstimate((*it_pt)->idist_);
        (*it_pt)->vPoint_ = vPoint;
        (*it_pt)->nBA_++;

        // Add Host Frame 
        g2oFrameSE3* vHost = NULL;
        if((*it_pt)->hostFeature_->frame->v_kf_ == NULL)
        {
            g2oFrameSE3* v_kf = createG2oFrameSE3((*it_pt)->hostFeature_->frame, v_id++, true);
            (*it_pt)->hostFeature_->frame->v_kf_ = v_kf;
            ++n_fix_kfs;
            assert(optimizer.addVertex(v_kf));
            hostKeyFrame.push_back((*it_pt)->hostFeature_->frame);
            vHost = v_kf;
        }
        else
            vHost = (*it_pt)->hostFeature_->frame->v_kf_;

        assert(optimizer.addVertex(vPoint));
        ++n_mps;

        // add other target frame
        // assert((*it_pt)->obs_.size() > 1);
        list<Feature*>::iterator it_obs=(*it_pt)->obs_.begin();
        while(it_obs!=(*it_pt)->obs_.end())
        {
            if((*it_obs)->frame->id_ == (*it_pt)->hostFeature_->frame->id_)
            {
                ++it_obs;
                continue;
            }

            g2oFrameSE3* vTarget = NULL;
            if((*it_obs)->frame->v_kf_ == NULL)
            {
                g2oFrameSE3* v_kf = createG2oFrameSE3((*it_obs)->frame, v_id++, true);
                (*it_obs)->frame->v_kf_ = v_kf;
                ++n_fix_kfs;
                assert(optimizer.addVertex(v_kf));
                neib_kfs.push_back((*it_obs)->frame);

                vTarget = v_kf;
            }
            else
                vTarget = (*it_obs)->frame->v_kf_;

            if((*it_obs)->type != Feature::EDGELET)
            {
                EdgeProjectID2UV* edge = new EdgeProjectID2UV();
                edge->resize(3);
                edge->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(vPoint));
                edge->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(vHost));
                edge->setVertex(2, dynamic_cast<g2o::OptimizableGraph::Vertex*>(vTarget));

                edge->setHostBearing((*it_pt)->hostFeature_->f);

                // // Gamma fuction
                // edge->setFocalLenght(center_kf->cam_->errorMultiplier2());
                // edge->setLevelScale(1.0/(1<<(*it_obs)->level));
                // edge->setGammaWeight(gamma_weight);
                // edge->setGamma(use_gamma);


                edge->setMeasurement(hso::project2d((*it_obs)->f));

                float inv_sigma2 = 1.0/((1<<(*it_obs)->level)*(1<<(*it_obs)->level));
                // inv_sigma2 *= inv_sigma2;
                edge->setInformation(Eigen::Matrix2d::Identity() * inv_sigma2);

                g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber();      
                rk->setDelta(huber_corner);
                edge->setRobustKernel(rk);
            

                edge->setParameterId(0, 0); 
     
                edges.push_back(EdgeContainerID(edge, (*it_obs)->frame, *it_obs));  
                assert(optimizer.addEdge(edge));
            }
            else
            {
                EdgeProjectID2UVEdgeLet* edgeLet = new EdgeProjectID2UVEdgeLet();
                edgeLet->resize(3);
                edgeLet->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(vPoint));
                edgeLet->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(vHost));
                edgeLet->setVertex(2, dynamic_cast<g2o::OptimizableGraph::Vertex*>(vTarget));

                edgeLet->setHostBearing((*it_pt)->hostFeature_->f);
                edgeLet->setTargetNormal((*it_obs)->grad);

                // // // Gamma fuction
                // edgeLet->setmeasurement2D(hso::project2d((*it_obs)->f));
                // edgeLet->setFocalLenght(center_kf->cam_->errorMultiplier2());
                // edgeLet->setLevelScale(1.0/(1<<(*it_obs)->level));
                // edgeLet->setGammaWeight(gamma_weight);
                // edgeLet->setGamma(use_gamma);

                edgeLet->setMeasurement((*it_obs)->grad.transpose()*hso::project2d((*it_obs)->f));

                float inv_sigma2 = 1.0/((1<<(*it_obs)->level)*(1<<(*it_obs)->level));
                // inv_sigma2 *= inv_sigma2;
                edgeLet->setInformation(Eigen::Matrix<double,1,1>::Identity()*inv_sigma2);

                g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber();
                rk->setDelta(huber_edge);
                edgeLet->setRobustKernel(rk);
                

                edgeLet->setParameterId(0, 0); 

                edgeLets.push_back(EdgeContainerIDEdgeLet(edgeLet, (*it_obs)->frame, *it_obs));  
                assert(optimizer.addEdge(edgeLet));
            }

            ++n_edges;
            ++it_obs;
        }
    }


    if(map->size() > 5)
    {
        if(center_kf->fts_.size() < 100)
            runSparseBAOptimizer(&optimizer, Config::lobaNumIter()+10, init_error, final_error);
        else
            runSparseBAOptimizer(&optimizer, Config::lobaNumIter(), init_error, final_error);
    }
    else
        runSparseBAOptimizer(&optimizer, 100, init_error, final_error);


    for(set<Frame*>::iterator it = core_kfs->begin(); it != core_kfs->end(); ++it)
    {
        (*it)->T_f_w_ = SE3( (*it)->v_kf_->estimate().rotation(), (*it)->v_kf_->estimate().translation());
        (*it)->v_kf_ = NULL;


        // if(center_kf->keyFrameId_ - (*it)->keyFrameId_ <= 4)
        map->point_candidates_.changeCandidatePosition(*it);
    }

    for(list<Frame*>::iterator it = neib_kfs.begin(); it != neib_kfs.end(); ++it)
        (*it)->v_kf_ = NULL;

    for(list<Frame*>::iterator it = hostKeyFrame.begin(); it != hostKeyFrame.end(); ++it)
        (*it)->v_kf_ = NULL;

    // Update Mappoints
    for(set<Point*>::iterator it = mps.begin(); it != mps.end(); ++it)
    {
        (*it)->idist_ = (*it)->vPoint_->estimate();
        (*it)->vPoint_ = NULL;

        //update position
        Vector3d pHost = (*it)->hostFeature_->f*(1.0/(*it)->idist_);
        (*it)->pos_ = (*it)->hostFeature_->frame->T_f_w_.inverse() * pHost;
    }



    const double reproj_thresh_2 = 2.0 / center_kf->cam_->errorMultiplier2();
    const double reproj_thresh_1 = 1.2 / center_kf->cam_->errorMultiplier2();

    const double reproj_thresh_2_squared = reproj_thresh_2*reproj_thresh_2;
    for(list<EdgeContainerID>::iterator it = edges.begin(); it != edges.end(); ++it)
    {
        if(it->feature->point == NULL) continue;

        // We only delete the temprorary point in reprojector
        if(it->edge->chi2() > reproj_thresh_2_squared)
        {
            if(it->feature->point->type_ == Point::TYPE_TEMPORARY) {
                it->feature->point->isBad_ = true;
                continue;
            }
            map->removePtFrameRef(it->frame, it->feature);
            ++n_incorrect_edges_1;
        }
    }

    const double reproj_thresh_1_squared = reproj_thresh_1*reproj_thresh_1;
    for(list<EdgeContainerIDEdgeLet>::iterator it = edgeLets.begin(); it != edgeLets.end(); ++it)
    {
        if(it->feature->point == NULL) continue;


        if(it->edge->chi2() > reproj_thresh_1_squared)
        {
            if(it->feature->point->type_ == Point::TYPE_TEMPORARY) {
                it->feature->point->isBad_ = true;
                continue;
            }
            map->removePtFrameRef(it->frame, it->feature);
            ++n_incorrect_edges_2;

            continue;
        }
    }


    init_error  = sqrt(init_error) *center_kf->cam_->errorMultiplier2();
    final_error = sqrt(final_error)*center_kf->cam_->errorMultiplier2();
}
} // namespace ba
} // namespace hso


