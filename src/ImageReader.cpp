//
//  ImageReader.cpp
//  OnlinePhotometricCalibration
//
//  Created by Paul on 16.11.17.
//  Copyright (c) 2017-2018 Paul Bergmann and co-authors. All rights reserved.
//
//  See LICENSE.txt
//

#include "hso/ImageReader.h"
#include <fstream>

namespace hso {

ImageReader::ImageReader(std::string image_folder, cv::Size new_size, std::string time_folder)
{
    getDir(image_folder, m_files);
    printf("ImageReader: got %d files in %s!\n", (int)m_files.size(), image_folder.c_str());

    m_img_new_size = new_size;

    m_stamp_valid = false;
    if(time_folder != "None")
    {   
        std::ifstream tr;
        tr.open(time_folder.c_str());
        while(!tr.eof() && tr.good())
        {
            char buf[1000];
            tr.getline(buf, 1000);

            int id;
            char stamp[100];
            float x,y,z,a,b,c,d;
            float exposure = 0;
  

            if(8 == sscanf(buf, "%s %f %f %f %f %f %f %f", stamp, &x, &y, &z, &a, &b, &c, &d))
            {
                std::string time_stamp = stamp;
                m_times.push_back(time_stamp);
            }
            else if(3 == sscanf(buf, "%d %s %f", &id, stamp, &exposure)) 
            {
                std::string time_stamp = stamp;
                m_times.push_back(time_stamp);
            }
            else if(2 == sscanf(buf, "%d %s", &id, stamp))  
            {
                std::string time_stamp = stamp;
                m_times.push_back(time_stamp);
            }
            else if(1 == sscanf(buf, "%s", stamp))
            {
                std::string time_stamp = stamp;
                m_times.push_back(time_stamp);
            }

        }
        tr.close();

        assert(m_times.size() == m_files.size());
        m_stamp_valid = true;
    }
}

cv::Mat ImageReader::readImage(int image_index)
{
    // Read image from disk
    cv::Mat image = cv::imread(m_files.at(image_index), cv::IMREAD_GRAYSCALE);
        
    if(!image.data)
    {
        std::cout << "ERROR READING IMAGE " << m_files.at(image_index) << std::endl;
        return cv::Mat();
    }
    
    // Resize input image
    cv::resize(image, image, m_img_new_size);
    
    return image;
}

std::string ImageReader::readStamp(int image_index)
{
    return m_times[image_index];
}

int ImageReader::getDir(std::string dir, std::vector<std::string> &files)
{
    DIR *dp;
    struct dirent *dirp;
    if((dp = opendir(dir.c_str())) == NULL)
    {
        return -1;
    }

    while ((dirp = readdir(dp)) != NULL)
    {
        std::string name = std::string(dirp->d_name);

        if(name != "." && name != "..")
        {
            if(strstr(name.c_str(),".png")!=NULL || strstr(name.c_str(),".jpg")!=NULL)
                files.push_back(name);
        }
    }

    closedir(dp);
    std::sort(files.begin(), files.end());


    if(dir.at(dir.length() - 1) != '/')
        dir = dir+"/";

    for(unsigned int i = 0; i < files.size(); i++)
    {
        if(files[i].at(0) != '/')
            files[i] = dir + files[i];
    }
    

    return (int)files.size();
}

}
