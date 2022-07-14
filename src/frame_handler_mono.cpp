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

#include <boost/timer.hpp>
#include <boost/thread.hpp>
#include <hso/config.h>
#include <hso/frame_handler_mono.h>
#include <hso/map.h>
#include <hso/frame.h>
#include <hso/feature.h>
#include <hso/point.h>
#include <hso/pose_optimizer.h>
#include <hso/matcher.h>
#include <hso/feature_alignment.h>
#include <hso/global.h>
#include <hso/depth_filter.h>
#include <hso/bundle_adjustment.h>


#include "hso/CoarseTracker.h"
#include "hso/camera.h"
#include "hso/vikit/performance_monitor.h"

namespace hso {

FrameHandlerMono::FrameHandlerMono(hso::AbstractCamera* cam, bool _use_pc) :
                                  FrameHandlerBase(),
                                  cam_(cam),
                                  reprojector_(cam_, map_),
                                  depth_filter_(NULL),
                                  motionModel_(SE3(Matrix3d::Identity(), Vector3d::Zero()))
{    
    // if(_use_pc)
    //     m_photomatric_calib = new PhotomatricCalibration(2, cam_->width(), cam_->height());
    // else
    //     m_photomatric_calib = NULL;

    initialize();
}

void FrameHandlerMono::initialize()
{
    feature_detection::FeatureExtractor* featureExt(
        new feature_detection::FeatureExtractor(cam_->width(), cam_->height(), Config::gridSize(), Config::nPyrLevels()));

    DepthFilter::callback_t depth_filter_cb = boost::bind(&MapPointCandidates::newCandidatePoint, &map_.point_candidates_, _1, _2);

    depth_filter_ = new DepthFilter(featureExt, depth_filter_cb);
    depth_filter_->startThread();

    reprojector_.depth_filter_ = depth_filter_;
}

FrameHandlerMono::~FrameHandlerMono()
{
    delete depth_filter_;
    // if(m_photomatric_calib != NULL) delete m_photomatric_calib;
}

void FrameHandlerMono::addImage(const cv::Mat& img, const double timestamp, string* timestamp_s)
{
    if(!startFrameProcessingCommon(timestamp))
        return;

    // some cleanup from last iteration, can't do before because of visualization
    core_kfs_.clear();
    overlap_kfs_.clear();

    // create new frame
    HSO_START_TIMER("pyramid_creation");

    new_frame_.reset(new Frame(cam_, img.clone(), timestamp));

    if(map_.size() == 0) 
        new_frame_->keyFrameId_ = 0;
    else
        new_frame_->keyFrameId_ = map_.lastKeyframe()->keyFrameId_;

    if(timestamp_s != NULL) new_frame_->m_timestamp_s = *timestamp_s;

    
    
    HSO_STOP_TIMER("pyramid_creation");

    // process frame
    UpdateResult res = RESULT_FAILURE;
    if(stage_ == STAGE_DEFAULT_FRAME)
        res = processFrame();
    else if(stage_ == STAGE_SECOND_FRAME)
        res = processSecondFrame();
    else if(stage_ == STAGE_FIRST_FRAME)
        res = processFirstFrame();
    else if(stage_ == STAGE_RELOCALIZING)
        res = relocalizeFrame(SE3(Matrix3d::Identity(), Vector3d::Zero()), map_.getClosestKeyframe(last_frame_));


    // set last frame
    last_frame_ = new_frame_;
    new_frame_.reset();

    // finish processing
    finishFrameProcessingCommon(last_frame_->id_, res, last_frame_->m_n_inliers);
}

FrameHandlerMono::UpdateResult FrameHandlerMono::processFirstFrame()
{
    new_frame_->T_f_w_ = SE3(Matrix3d::Identity(), Vector3d::Zero());



    if(klt_homography_init_.addFirstFrame(new_frame_) == initialization::FAILURE)
        return RESULT_NO_KEYFRAME;
    new_frame_->setKeyframe();
    map_.addKeyframe(new_frame_);
    stage_ = STAGE_SECOND_FRAME;
    HSO_INFO_STREAM("Init: Selected first frame.");

    firstFrame_ = new_frame_;

    firstFrame_->m_exposure_time = 1.0;
    firstFrame_->m_exposure_finish = true;





    return RESULT_IS_KEYFRAME;
}


FrameHandlerBase::UpdateResult FrameHandlerMono::processSecondFrame()
{
    initialization::InitResult res = klt_homography_init_.addSecondFrame(new_frame_);





    if(res == initialization::FAILURE)
        return RESULT_FAILURE;
    else if(res == initialization::NO_KEYFRAME)
        return RESULT_NO_KEYFRAME;

    stage_ = STAGE_DEFAULT_FRAME;
    klt_homography_init_.reset();
    HSO_INFO_STREAM("Init: Selected second frame, triangulated initial map.");

    afterInit_ = true;
    firstFrame_->setKeyPoints();
    return RESULT_IS_KEYFRAME;
}

FrameHandlerBase::UpdateResult FrameHandlerMono::processFrame()
{
    // Set initial pose 
    new_frame_->T_f_w_ = motionModel_ * last_frame_->T_f_w_;
    // calcMotionModel();

    if(afterInit_) last_frame_ = firstFrame_;

    new_frame_->m_last_frame = last_frame_;


    if(new_frame_->gradMean_ > last_frame_->gradMean_ + 0.5)
    {
        //HSO_DEBUG_STREAM("Img Align:\t Using The Lucas-Kanade Algorithm.");
        boost::timer align;
        HSO_START_TIMER("sparse_img_align");

        CoarseTracker Tracker(false, Config::kltMaxLevel(), Config::kltMinLevel()+1, 50, false);
        size_t img_align_n_tracked = Tracker.run(last_frame_, new_frame_);

        HSO_STOP_TIMER("sparse_img_align");
        HSO_LOG(img_align_n_tracked);
        HSO_DEBUG_STREAM("Img Align:\t Tracked = " << img_align_n_tracked <<"\t \t Cost = "<<align.elapsed()<<"s");
    }
    else
    {
        // HSO_DEBUG_STREAM("Img Align:\t Using The Inverse Compositional Algorithm.");
        boost::timer align;
        HSO_START_TIMER("sparse_img_align");

        CoarseTracker Tracker(true, Config::kltMaxLevel(), Config::kltMinLevel()+1, 50, false);
        size_t img_align_n_tracked = Tracker.run(last_frame_, new_frame_);

        HSO_STOP_TIMER("sparse_img_align");
        HSO_LOG(img_align_n_tracked);
        HSO_DEBUG_STREAM("Img Align:\t Tracked = " << img_align_n_tracked << "\t \t Cost = "<<align.elapsed()<<"s");
    }



    // map reprojection & feature alignment
    boost::timer reprojector;
    HSO_START_TIMER("reproject");

    reprojector_.reprojectMap(new_frame_, overlap_kfs_);

    HSO_STOP_TIMER("reproject");
    const size_t repr_n_new_references = reprojector_.n_matches_;
    const size_t repr_n_mps = reprojector_.n_trials_;
    const size_t repr_n_sds = reprojector_.n_seeds_;
    const size_t repr_n_fis = reprojector_.n_filters_;
    HSO_LOG2(repr_n_mps, repr_n_new_references);
    HSO_DEBUG_STREAM("Reprojection:\t nPoints = "<<repr_n_mps<<"\t \t nMatches = "<<repr_n_new_references <<"\t \t nSeeds = "<<repr_n_sds<<"\t \t Cost = "<<reprojector.elapsed()<<"s");


    if(repr_n_new_references < Config::qualityMinFts())
    {
        HSO_WARN_STREAM_THROTTLE(1.0, "Not enough matched features.");
        new_frame_->T_f_w_ = last_frame_->T_f_w_; // reset to avoid crazy pose jumps
        tracking_quality_ = TRACKING_INSUFFICIENT;
        return RESULT_FAILURE;
    }

    // pose optimization
    HSO_START_TIMER("pose_optimizer");
    size_t sfba_n_edges_final = 0;
    double sfba_thresh = 0, sfba_error_init = 0, sfba_error_final = 0;

    pose_optimizer::optimizeLevenbergMarquardt3rd(
        Config::poseOptimThresh(), 12, false,
        new_frame_, sfba_thresh, sfba_error_init, sfba_error_final, sfba_n_edges_final);


    new_frame_->m_n_inliers = sfba_n_edges_final;



    HSO_STOP_TIMER("pose_optimizer");
    HSO_LOG4(sfba_thresh, sfba_error_init, sfba_error_final, sfba_n_edges_final);
    HSO_DEBUG_STREAM("PoseOptimizer:\t ErrInit = "<<sfba_error_init<<"px\t thresh = "<<sfba_thresh);
    HSO_DEBUG_STREAM("PoseOptimizer:\t ErrFin. = "<<sfba_error_final<<"px\t nObsFin. = "<<sfba_n_edges_final);
    if(sfba_n_edges_final < Config::qualityMinFts())
        return RESULT_FAILURE;
      


    
    // select keyframe
    core_kfs_.insert(new_frame_);
    setTrackingQuality(sfba_n_edges_final);
    if(tracking_quality_ == TRACKING_INSUFFICIENT)
    {
        new_frame_->T_f_w_ = last_frame_->T_f_w_; // reset to avoid crazy pose jumps
        return RESULT_FAILURE;
    }

    double depth_mean, depth_min;
    frame_utils::getSceneDepth(*new_frame_, depth_mean, depth_min);
    double distance_mean;
    frame_utils::getSceneDistance(*new_frame_, distance_mean);


    if(!needNewKf(distance_mean, sfba_n_edges_final) && !afterInit_)
    {
        createCovisibilityGraph(new_frame_, Config::coreNKfs(), false);


        // m_photomatric_calib->addFrame(new_frame_, pc_step);
            

        depth_filter_->addFrame(new_frame_);

        regular_counter_++;

        motionModel_ = new_frame_->T_f_w_ * last_frame_->T_f_w_.inverse();
        

        
        return RESULT_NO_KEYFRAME;
    }

    if(afterInit_) afterInit_ = false;

    // changeFrameEdgeLetNormal();

    regular_counter_ = 0;
    new_frame_->setKeyframe();


    // new keyframe selected
    for(Features::iterator it=new_frame_->fts_.begin(); it!=new_frame_->fts_.end(); ++it)
        if((*it)->point != NULL)
            (*it)->point->addFrameRef(*it);

    map_.point_candidates_.addCandidatePointToFrame(new_frame_);



    createCovisibilityGraph(new_frame_, Config::coreNKfs(), true);

    // bundle adjustment
    if(Config::lobaNumIter() > 0)
    {
        HSO_START_TIMER("local_ba");

        size_t loba_n_erredges_init, loba_n_erredges_fin;
        double loba_err_init, loba_err_fin;

        ba::LocalBundleAdjustment(new_frame_.get(), &LocalMap_, &map_, loba_n_erredges_init, loba_n_erredges_fin, loba_err_init, loba_err_fin);

        HSO_STOP_TIMER("local_ba");
        HSO_LOG4(loba_n_erredges_init, loba_n_erredges_fin, loba_err_init, loba_err_fin);
        HSO_DEBUG_STREAM("Local BA:\t RemovedEdges {"<<loba_n_erredges_init<<", "<<loba_n_erredges_fin<<"} \t "
                        "Error {"<<loba_err_init<<", "<<loba_err_fin<<"}" << " \t" << "Local Map Size {" << LocalMap_.size() << "}");
    }

    // m_photomatric_calib->addKeyFrame(new_frame_, pc_step);
    
        

    for(auto& kf: overlap_kfs_) kf.first->setKeyPoints();

    if(sfba_n_edges_final <= 70)
        depth_filter_->addKeyframe(new_frame_, distance_mean, 0.5*depth_min, 100);  // robust
    else
        depth_filter_->addKeyframe(new_frame_, distance_mean, 0.5*depth_min, 200);  








    
    // add keyframe to map
    map_.addKeyframe(new_frame_);


    motionModel_ = new_frame_->T_f_w_ * last_frame_->T_f_w_.inverse();

    return RESULT_IS_KEYFRAME;
}

FrameHandlerMono::UpdateResult FrameHandlerMono::relocalizeFrame(const SE3& T_cur_ref, FramePtr ref_keyframe)
{
    HSO_WARN_STREAM_THROTTLE(1.0, "Relocalizing frame");
    if(ref_keyframe == nullptr)
    {
        HSO_INFO_STREAM("No reference keyframe.");
        return RESULT_FAILURE;
    }

    CoarseTracker Tracker(true, Config::kltMaxLevel(), Config::kltMinLevel(), 15, false);
    size_t img_align_n_tracked = Tracker.run(ref_keyframe, last_frame_);


    if(img_align_n_tracked > 30)
    {
        SE3 T_f_w_last = last_frame_->T_f_w_;
        last_frame_ = ref_keyframe;
        FrameHandlerMono::UpdateResult res = processFrame();
        if(res != RESULT_FAILURE)
        {
            stage_ = STAGE_DEFAULT_FRAME;
            HSO_INFO_STREAM("Relocalization successful.");
        }
        else
            new_frame_->T_f_w_ = T_f_w_last; // reset to last well localized pose

        return res;
    }
    return RESULT_FAILURE;
}

bool FrameHandlerMono::relocalizeFrameAtPose(
    const int keyframe_id,
    const SE3& T_f_kf,
    const cv::Mat& img,
    const double timestamp)
{
    FramePtr ref_keyframe;
    if(!map_.getKeyframeById(keyframe_id, ref_keyframe))
        return false;

    new_frame_.reset(new Frame(cam_, img.clone(), timestamp));
    UpdateResult res = relocalizeFrame(T_f_kf, ref_keyframe);

    if(res != RESULT_FAILURE) 
    {
        last_frame_ = new_frame_;
        return true;
    }
    return false;
}

void FrameHandlerMono::resetAll()
{
    resetCommon();
    last_frame_.reset();
    new_frame_.reset();
    core_kfs_.clear();
    overlap_kfs_.clear();
    depth_filter_->reset();
}

void FrameHandlerMono::setFirstFrame(const FramePtr& first_frame)
{
    resetAll();
    last_frame_ = first_frame;
    last_frame_->setKeyframe();
    map_.addKeyframe(last_frame_);
    stage_ = STAGE_DEFAULT_FRAME;
}

bool FrameHandlerMono::needNewKf(const double& scene_depth_mean, const size_t& num_observations)
{   
    if(regular_counter_ < 3) return false;


    size_t n_mean_converge_frame;
    {
        boost::unique_lock<boost::mutex> lock(depth_filter_->mean_mutex_);
        n_mean_converge_frame = depth_filter_->nMeanConvergeFrame_;
    }

    // if(num_observations <= 70 && regular_counter_ > 0.7*n_mean_converge_frame) return true;

    // if(num_observations > 120 && map_.size() > 5)
    // {
    //     float nLockFrame = n_mean_converge_frame*0.8;
    //     if(nLockFrame > 5) nLockFrame = 5;
    //     if(regular_counter_ < nLockFrame) return false;
    // }

    if(regular_counter_ < std::min(3, int(n_mean_converge_frame*0.8))) return false;
    


    const FramePtr last_kf = map_.lastKeyframe();
    const SE3 T_c_r_full(new_frame_->T_f_w_ * last_kf->T_f_w_.inverse());
    const SE3 T_c_r_nR(Matrix3d::Identity(), T_c_r_full.translation());
    // const SE3 T_c_r_nt(T_c_r_full.rotation_matrix(), Vector3d(0,0,0));

    float optical_flow_full = 0, optical_flow_nR = 0, optical_flow_nt = 0;
    size_t optical_flow_num = 0;

    for(auto& ft_kf: last_kf->fts_)
    {
        if(ft_kf->point == NULL) continue;

        Vector3d p_ref(ft_kf->f * (ft_kf->point->pos_ - last_kf->pos()).norm());
        Vector3d p_cur_full(T_c_r_full * p_ref);
        Vector3d p_cur_nR(T_c_r_nR * p_ref);
        // Vector3d p_cur_nt(T_c_r_nt * p_ref);

        Vector2d uv_cur_full(new_frame_->cam_->world2cam(p_cur_full));
        Vector2d uv_cur_nR(new_frame_->cam_->world2cam(p_cur_nR));
        // Vector2d uv_cur_nt(new_frame_->cam_->world2cam(p_cur_nt));

        optical_flow_full += (uv_cur_full - ft_kf->px).squaredNorm();
        optical_flow_nR += (uv_cur_nR - ft_kf->px).squaredNorm();
        // optical_flow_nt += (uv_cur_nt - ft_kf->px).squaredNorm();

        optical_flow_num++;
    }
    optical_flow_full /= optical_flow_num; if(optical_flow_full < 133) return false;
    optical_flow_full = sqrtf(optical_flow_full);

    optical_flow_nR /= optical_flow_num; 
    optical_flow_nR = sqrtf(optical_flow_nR);

    // optical_flow_nt /= optical_flow_num; 
    // optical_flow_nt = sqrtf(optical_flow_nt);
    



    const int defult_resolution = 752+480;
    const float setting_maxShiftWeightT = 0.04*defult_resolution; 
    // const float setting_maxShiftWeightR = 0.01f*defult_resolution;
    const float setting_maxShiftWeightRT = 0.02*defult_resolution;
    const float setting_kfGlobalWeight = 0.75;

    const int wh = new_frame_->cam_->width() + new_frame_->cam_->height();

    float DSO_judgement = setting_kfGlobalWeight*setting_maxShiftWeightT* optical_flow_nR   / wh+
                       // setting_kfGlobalWeight*setting_maxShiftWeightR* optical_flow_nt   / wh+
                          setting_kfGlobalWeight*setting_maxShiftWeightRT*optical_flow_full / wh;

    return DSO_judgement > 1;



}

void FrameHandlerMono::setCoreKfs(size_t n_closest)
{
    size_t n = min(n_closest, overlap_kfs_.size()-1);
    std::partial_sort(overlap_kfs_.begin(), overlap_kfs_.begin()+n, overlap_kfs_.end(),
                    boost::bind(&pair<FramePtr, size_t>::second, _1) >
                    boost::bind(&pair<FramePtr, size_t>::second, _2));
    std::for_each(overlap_kfs_.begin(), overlap_kfs_.end(), [&](pair<FramePtr,size_t>& i){ core_kfs_.insert(i.first); });
}

bool FrameHandlerMono::kfOverView()
{
    size_t n_overLap = 0;
    for(set<Frame*>::iterator ite = LocalMap_.begin(); ite != LocalMap_.end(); ++ite)
    {
        size_t nPoints = 0;
        for(auto& keypoint: (*ite)->key_pts_)
        {
            if(keypoint == nullptr) continue;

            if(new_frame_->isVisible(keypoint->point->pos_))
                nPoints++;
        }
        if(nPoints < 5) n_overLap++;
    }

    if(n_overLap == LocalMap_.size()) return true;

    return false;
}




bool FrameHandlerMono::frameCovisibilityComparator(pair<int, Frame*>& lhs, pair<int, Frame*>& rhs)
{
    if(lhs.first != rhs.first)
        return (lhs.first > rhs.first);
    else
        return (lhs.second->id_ < rhs.second->id_);

    // return (lhs.first-0.1*lhs.second->keyFrameId_) > (rhs.first-0.1*rhs.second->keyFrameId_);
}

bool FrameHandlerMono::frameCovisibilityComparatorF(pair<float, Frame*>& lhs, pair<float, Frame*>& rhs)
{
    return lhs.first > rhs.first;
}


// modified code in ORB-SLAM (https://github.com/raulmur/ORB_SLAM)
void FrameHandlerMono::createCovisibilityGraph(FramePtr currentFrame, size_t n_closest, bool is_keyframe)
{
    std::map<Frame*, int> KFcounter;
    int n_linliers = 0;
    for(Features::iterator it = currentFrame->fts_.begin(); it != currentFrame->fts_.end(); ++it)
    {
        if((*it)->point == NULL) continue;

        n_linliers++;

        for(auto ite = (*it)->point->obs_.begin(); ite != (*it)->point->obs_.end(); ++ite)
        {
            if((*ite)->frame->id_== currentFrame->id_) continue;
            KFcounter[(*ite)->frame]++;
        }
    }

    



    // This should not happen
    if(KFcounter.empty()) return;

    int nmax=0;
    Frame* pKFmax=NULL;
    const int th = n_linliers > 30? 5 : 3;

    vector< pair<int, Frame*> > vPairs;
    vPairs.reserve(KFcounter.size());  


    vpLocalKeyFrames.clear();

    for(std::map<Frame*,int>::iterator mit=KFcounter.begin(), mend=KFcounter.end(); mit!=mend; mit++)
    {
        vpLocalKeyFrames.push_back(mit->first); 

        if(mit->second>nmax) 
        {
            nmax=mit->second;
            pKFmax=mit->first;
        }

        if(mit->second>=th)
            vPairs.push_back(make_pair(mit->second,mit->first));


        if(mit->first->keyFrameId_+5 < currentFrame->keyFrameId_)
            if(!mit->first->sobelX_.empty())
            {
                mit->first->sobelX_.clear();
                mit->first->sobelY_.clear();
            }
    }
  

    if(vPairs.empty())
        vPairs.push_back(make_pair(nmax,pKFmax));

    std::partial_sort(vPairs.begin(), vPairs.begin()+vPairs.size(), vPairs.end(), boost::bind(&FrameHandlerMono::frameCovisibilityComparator, _1, _2));

    const size_t nCovisibility = 5;
    size_t k = min(nCovisibility, vPairs.size());
    for(size_t i = 0; i < k; ++i)
        currentFrame->connectedKeyFrames.push_back(vPairs[i].second);


    if(is_keyframe)
    {
        LocalMap_.clear();

        size_t n = min(n_closest, vPairs.size());
        std::for_each(vPairs.begin(), vPairs.begin()+n, [&](pair<int, Frame*>& i){ LocalMap_.insert(i.second); });






        FramePtr LastKF = map_.lastKeyframe();
        if(LocalMap_.find(LastKF.get()) == LocalMap_.end())
            LocalMap_.insert(LastKF.get()); 

        assert(LocalMap_.find(currentFrame.get()) == LocalMap_.end());
        LocalMap_.insert(currentFrame.get());

    }
}



} // namespace hso
