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


#include <set>
#include <hso/map.h>
#include <hso/point.h>
#include <hso/frame.h>
#include <hso/feature.h>
#include <boost/bind.hpp>

namespace hso {

Map::Map() {}

Map::~Map()
{
  reset();
  HSO_INFO_STREAM("Map destructed");
}

void Map::reset()
{
  keyframes_.clear();
  point_candidates_.reset();
  emptyTrash();
}

bool Map::safeDeleteFrame(FramePtr frame)
{
  bool found = false;
  for(auto it=keyframes_.begin(), ite=keyframes_.end(); it!=ite; ++it)
  {
    if(*it == frame)
    {
      std::for_each((*it)->fts_.begin(), (*it)->fts_.end(), [&](Feature* ftr){
        removePtFrameRef(it->get(), ftr);
      });
      keyframes_.erase(it);
      found = true;
      break;
    }
  }

  point_candidates_.removeFrameCandidates(frame);

  if(found)
    return true;

  HSO_ERROR_STREAM("Tried to delete Keyframe in map which was not there.");
  return false;
}

bool Map::safeDeleteFrameID(int id)
{
  bool found = false;
  FramePtr delete_frame = NULL;
  for(auto it=keyframes_.begin(), ite=keyframes_.end(); it!=ite; ++it)
  {
    if((*it)->id_ == id)
    {
      delete_frame = *it;

      std::for_each((*it)->fts_.begin(), (*it)->fts_.end(), [&](Feature* ftr){
        removePtFrameRef(it->get(), ftr);
      });
      keyframes_.erase(it);
      found = true;
      break;
    }
  }

  point_candidates_.removeFrameCandidates(delete_frame);

  if(found)
    return true;

  HSO_ERROR_STREAM("Tried to delete Keyframe in map which was not there.");
  return false;
}

void Map::removePtFrameRef(Frame* frame, Feature* ftr)
{
  if(ftr->point == NULL)
    return; // mappoint may have been deleted in a previous ref. removal
  Point* pt = ftr->point;
  ftr->point = NULL;
  if(pt->obs_.size() <= 2)
  {
    // If the references list of mappoint has only size=2, delete mappoint
    safeDeletePoint(pt);
    return;
  }
  pt->deleteFrameRef(frame);  // Remove reference from map_point
  frame->removeKeyPoint(ftr); // Check if mp was keyMp in keyframe
}

void Map::safeDeletePoint(Point* pt)
{
  // Delete references to mappoints in all keyframes
  std::for_each(pt->obs_.begin(), pt->obs_.end(), [&](Feature* ftr){
    ftr->point=NULL;
    ftr->frame->removeKeyPoint(ftr);
  });
  pt->obs_.clear();

  // Delete mappoint
  deletePoint(pt);
}

void Map::safeDeleteTempPoint(pair<Point*, Feature*>& p)
{
    if(p.first->seedStates_ == -1)
    {
        if(p.first->isBad_)
            safeDeletePoint(p.first);
        else
        {
            assert(p.first->hostFeature_ == p.second);

            // update pos
            p.first->pos_ = p.first->hostFeature_->frame->T_f_w_.inverse() * (p.first->hostFeature_->f*(1.0/p.first->idist_));

            if(p.first->obs_.size() == 1)
            {
                // Add to candidates
                p.first->type_ = Point::TYPE_CANDIDATE;
                p.first->n_failed_reproj_ = 0;
                p.first->n_succeeded_reproj_ = 0;

                // thread safe
                point_candidates_.candidates_.push_back(MapPointCandidates::PointCandidate(p.first, p.first->obs_.front()));
            }
            else
            {
                // update to unknow
                p.first->type_ = Point::TYPE_UNKNOWN;
                p.first->n_failed_reproj_ = 0;
                p.first->n_succeeded_reproj_ = 0;
                p.second->frame->addFeature(p.second);
            }
        }
    }
    else 
    {   
        assert(p.first->seedStates_ == 1 && 
               p.first->obs_.back()->point->id_ == p.second->point->id_);

        for(auto it = p.first->obs_.begin(); it != p.first->obs_.end(); ++it)
            if((*it)->point->id_ != p.second->point->id_)  // delete target frame obs
            {
                (*it)->point=NULL;
                (*it)->frame->removeKeyPoint(*it);
            }

        p.first->obs_.clear();
        deletePoint(p.first);
    }
}


void Map::deletePoint(Point* pt)
{
  pt->type_ = Point::TYPE_DELETED;
  trash_points_.push_back(pt);
}

void Map::addKeyframe(FramePtr new_keyframe)
{
  keyframes_.push_back(new_keyframe);
}

void Map::getCloseKeyframes(
    const FramePtr& frame,
    std::list< std::pair<FramePtr,double> >& close_kfs) const
{
    for(auto kf : keyframes_)
    {   
        // check if kf has overlaping field of view with frame, use therefore KeyPoints
        for(auto keypoint : kf->key_pts_)
        {   
            if(keypoint == nullptr) continue;
  
            assert(keypoint->point != NULL);
            if(frame->isVisible(keypoint->point->pos_))
            {

                close_kfs.push_back( std::make_pair( kf, (frame->T_f_w_.translation()-kf->T_f_w_.translation()).norm() ) );
                break; // this keyframe has an overlapping field of view -> add to close_kfs
            }
        }
    }
}

FramePtr Map::getClosestKeyframe(const FramePtr& frame) const
{
  list< pair<FramePtr,double> > close_kfs;
  getCloseKeyframes(frame, close_kfs);
  if(close_kfs.empty())
  {
    return nullptr;
  }


  // Sort KFs with overlap according to their closeness
  close_kfs.sort(boost::bind(&std::pair<FramePtr, double>::second, _1) <
                 boost::bind(&std::pair<FramePtr, double>::second, _2));

  if(close_kfs.front().first != frame)
    return close_kfs.front().first;
  close_kfs.pop_front();
  return close_kfs.front().first;
}

FramePtr Map::getFurthestKeyframe(const Vector3d& pos) const
{
  FramePtr furthest_kf;
  double maxdist = 0.0;
  for(auto it=keyframes_.begin(), ite=keyframes_.end(); it!=ite; ++it)
  {
    double dist = ((*it)->pos()-pos).norm();
    if(dist > maxdist) {
      maxdist = dist;
      furthest_kf = *it;
    }
  }
  return furthest_kf;
}

bool Map::getKeyframeById(const int id, FramePtr& frame) const
{
    bool found = false;
    for(auto it=keyframes_.begin(), ite=keyframes_.end(); it!=ite; ++it)
        if((*it)->id_ == id) 
        {
            found = true;
            frame = *it;
            break;
        }
    return found;
}

void Map::transform(const Matrix3d& R, const Vector3d& t, const double& s)
{
  for(auto it=keyframes_.begin(), ite=keyframes_.end(); it!=ite; ++it)
  {
    Vector3d pos = s*R*(*it)->pos() + t;
    Matrix3d rot = R*(*it)->T_f_w_.rotation_matrix().inverse();
    (*it)->T_f_w_ = SE3(rot, pos).inverse();
    for(auto ftr=(*it)->fts_.begin(); ftr!=(*it)->fts_.end(); ++ftr)
    {
      if((*ftr)->point == NULL)
        continue;
      if((*ftr)->point->last_published_ts_ == -1000)
        continue;
      (*ftr)->point->last_published_ts_ = -1000;
      (*ftr)->point->pos_ = s*R*(*ftr)->point->pos_ + t;
    }
  }
}

void Map::emptyTrash()
{
  std::for_each(trash_points_.begin(), trash_points_.end(), [&](Point*& pt){
    delete pt;
    pt=NULL;
  });
  trash_points_.clear();
  point_candidates_.emptyTrash();
}

MapPointCandidates::MapPointCandidates()
{}

MapPointCandidates::~MapPointCandidates()
{
  reset();
}

void MapPointCandidates::newCandidatePoint(Point* point, double depth_sigma2)
{
  point->type_ = Point::TYPE_CANDIDATE;
  boost::unique_lock<boost::mutex> lock(mut_);

  candidates_.push_back(PointCandidate(point, point->obs_.front()));
}

void MapPointCandidates::addPauseSeedPoint(Point* point)
{
  assert(point->type_ == Point::TYPE_TEMPORARY);
  boost::unique_lock<boost::mutex> lock(mut_);

  assert(point->hostFeature_ == point->obs_.front());

  temporaryPoints_.push_back(make_pair(point, point->obs_.front()));
}

void MapPointCandidates::addCandidatePointToFrame(FramePtr frame)
{
    boost::unique_lock<boost::mutex> lock(mut_);
    PointCandidateList::iterator it=candidates_.begin();
    while(it != candidates_.end())
    {
        if(it->first->obs_.front()->frame == frame.get())
        {
            assert(it->first->obs_.size() == 2);

            // insert feature in the frame
            it->first->type_ = Point::TYPE_UNKNOWN;
            it->first->n_failed_reproj_ = 0;

            it->second->frame->addFeature(it->second);


            it = candidates_.erase(it);
        }
        // else if(it->first->n_succeeded_reproj_ > 30)
        // {
        //     assert(it->first->obs_.size() == 1);

        //     it->first->type_ = Point::TYPE_UNKNOWN;
        //     it->first->n_failed_reproj_ = 0;

        //     it->first->n_succeeded_reproj_ = 0; 

        //     // Run away from list!
        //     // if(!it->second->haveAdded)
        //     it->second->frame->addFeature(it->second);
        //     // else
        //     //   cv::waitKey(0);
        //     // for(auto ite = it->second.begin(); ite != it->second.end(); ++ite)
        //     //   (*ite)->frame->addFeature(*ite);

        //     it = candidates_.erase(it);
        // }
        else
            ++it;
    }
}

bool MapPointCandidates::deleteCandidatePoint(Point* point)
{
    boost::unique_lock<boost::mutex> lock(mut_);
    for(auto it=candidates_.begin(), ite=candidates_.end(); it!=ite; ++it)
    {
        if(it->first == point)
        {

            // Frame* frame = it->second->frame;
            // assert(frame->deleteOneChild(it->second));



            deleteCandidate(*it);
            candidates_.erase(it);
            return true;
        }
    }
    return false;
}

void MapPointCandidates::changeCandidatePosition(Frame* frame)
{
    boost::unique_lock<boost::mutex> lock(mut_);
    
    for(PointCandidateList::iterator it = candidates_.begin(); it != candidates_.end(); ++it)
    {
        Point* point = it->first;
        Feature* ft = it->second;

        assert(point != NULL && 
               point->type_ == Point::TYPE_CANDIDATE &&
               point->obs_.size() == 1 &&
               point->vPoint_ == NULL);

        if(ft->frame->id_ == frame->id_)
            point->pos_ = frame->T_f_w_.inverse() * (ft->f * (1.0/point->idist_));
    }
}

void MapPointCandidates::removeFrameCandidates(FramePtr frame)
{
  boost::unique_lock<boost::mutex> lock(mut_);
  auto it=candidates_.begin();
  // bool haveFound = false;
  while(it!=candidates_.end())
  {
    if(it->second->frame == frame.get())
    {
      deleteCandidate(*it);
      it = candidates_.erase(it);
    }
    else
      ++it;
    // haveFound = false;
    // for(auto ite = it->second.begin(); ite != it->second.end(); ++ite)
    //   if((*ite)->frame == frame.get())
    //   {
    //     deleteCandidate(*it);
    //     it = candidates_.erase(it);
    //     haveFound = true;
    //     break;
    //   }
    
    // if(!haveFound)
    //   ++it;
  }
}

void MapPointCandidates::reset()
{
  boost::unique_lock<boost::mutex> lock(mut_);
  std::for_each(candidates_.begin(), candidates_.end(), [&](PointCandidate& c){
    delete c.first;
    delete c.second;
    // for(auto it = c.second.begin(); it != c.second.end(); ++it)
    // {
    //   delete *it;
    //   *it = NULL;
    // }
    // c.second.clear();
  });
  candidates_.clear();
}

void MapPointCandidates::deleteCandidate(PointCandidate& c)
{
  // camera-rig: another frame might still be pointing to the candidate point
  // therefore, we can't delete it right now.

  delete c.second; c.second=NULL;
  // for(auto it = c.second.begin(); it != c.second.end(); ++it)
  // {
  //   delete *it;
  //   *it = NULL;
  // }
  // c.second.clear();

  c.first->type_ = Point::TYPE_DELETED;
  trash_points_.push_back(c.first);
}

void MapPointCandidates::emptyTrash()
{
  std::for_each(trash_points_.begin(), trash_points_.end(), [&](Point*& p){
    delete p; p=NULL;
  });
  trash_points_.clear();
}

namespace map_debug {

void mapValidation(Map* map, int id)
{
  for(auto it=map->keyframes_.begin(); it!=map->keyframes_.end(); ++it)
    frameValidation(it->get(), id);
}

void frameValidation(Frame* frame, int id)
{
  for(auto it = frame->fts_.begin(); it!=frame->fts_.end(); ++it)
  {
    if((*it)->point==NULL)
      continue;

    if((*it)->point->type_ == Point::TYPE_DELETED)
      printf("ERROR DataValidation %i: Referenced point was deleted.\n", id);

    if(!(*it)->point->findFrameRef(frame))
      printf("ERROR DataValidation %i: Frame has reference but point does not have a reference back.\n", id);

    pointValidation((*it)->point, id);
  }
  for(auto it=frame->key_pts_.begin(); it!=frame->key_pts_.end(); ++it)
    if(*it != NULL)
      if((*it)->point == NULL)
        printf("ERROR DataValidation %i: KeyPoints not correct!\n", id);
}

void pointValidation(Point* point, int id)
{
  for(auto it=point->obs_.begin(); it!=point->obs_.end(); ++it)
  {
    bool found=false;
    for(auto it_ftr=(*it)->frame->fts_.begin(); it_ftr!=(*it)->frame->fts_.end(); ++it_ftr)
     if((*it_ftr)->point == point) {
       found=true; break;
     }
    if(!found)
      printf("ERROR DataValidation %i: Point %i has inconsistent reference in frame %i, is candidate = %i\n", id, point->id_, (*it)->frame->id_, (int) point->type_);
  }
}

void mapStatistics(Map* map)
{
  // compute average number of features which each frame observes
  size_t n_pt_obs(0);
  for(auto it=map->keyframes_.begin(); it!=map->keyframes_.end(); ++it)
    n_pt_obs += (*it)->nObs();
  printf("\n\nMap Statistics: Frame avg. point obs = %f\n", (float) n_pt_obs/map->size());

  // compute average number of observations that each point has
  size_t n_frame_obs(0);
  size_t n_pts(0);
  std::set<Point*> points;
  for(auto it=map->keyframes_.begin(); it!=map->keyframes_.end(); ++it)
  {
    for(auto ftr=(*it)->fts_.begin(); ftr!=(*it)->fts_.end(); ++ftr)
    {
      if((*ftr)->point == NULL)
        continue;
      if(points.insert((*ftr)->point).second) {
        ++n_pts;
        n_frame_obs += (*ftr)->point->nRefs();
      }
    }
  }
  printf("Map Statistics: Point avg. frame obs = %f\n\n", (float) n_frame_obs/n_pts);
}

} // namespace map_debug
} // namespace hso
