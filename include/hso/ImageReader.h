//
//  ImageReader.h
//  OnlinePhotometricCalibration
//
//  Created by Paul on 16.11.17.
//  Copyright (c) 2017-2018 Paul Bergmann and co-authors. All rights reserved.
//
//  See LICENSE.txt
//

#pragma once

#include <stdio.h>
#include <iostream>
#include <dirent.h>
#include <opencv2/opencv.hpp>

namespace hso {

/**
 * Read input images from image files
 * Resizes the images if requested
 */

class ImageReader
{
public:
    
    /** 
     * Initialize the image reader
     * @param image_folder Image folder
     * @param new_size Resize input images to new_size
     */
    ImageReader(std::string image_folder, cv::Size new_size, std::string time_folder="None");

    /**
     * Read a new input image from the hard drive and return it
     *
     * @param Input image index to read
     * @return Read input image
     */
    cv::Mat readImage(int image_index);
    std::string readStamp(int image_index);

    int getNumImages() { return (int)m_files.size(); }

    int getDir(std::string dir, std::vector<std::string> &files);

    inline bool stampValid() { return m_stamp_valid; }
    
private:
    
    /**
     * Resize images to this size
     */
    cv::Size m_img_new_size;

    bool m_stamp_valid;

    std::vector<std::string> m_files;
    std::vector<std::string> m_times;
};
}
