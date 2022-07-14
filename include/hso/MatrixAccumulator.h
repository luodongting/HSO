/**
* This file is part of DSO.
* 
* Copyright 2016 Technical University of Munich and Intel.
* Developed by Jakob Engel <engelj at in dot tum dot de>,
* for more information see <http://vision.in.tum.de/dso>.
* If you use this code, please cite the respective publications as
* listed on the above website.
*
* DSO is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* DSO is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with DSO. If not, see <http://www.gnu.org/licenses/>.
*/

#pragma once

#include "hso/global.h"
#include <emmintrin.h>

class Accumulator7
{
public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW

	Eigen::Matrix<float,7,7> H;
	Eigen::Matrix<float,7,1> b;
	size_t num;

	inline void initialize()
	{
		H.setZero();
		b.setZero();
		std::memset(SSEData,   0, sizeof(float)*4*28);
		std::memset(SSEData1k, 0, sizeof(float)*4*28);
		std::memset(SSEData1m, 0, sizeof(float)*4*28);
		num = numIn1 = numIn1k = numIn1m = 0;
	}

	inline void finish()
	{
		shiftUp(true);
		assert(numIn1==0);
		assert(numIn1k==0);

		int idx=0;
		for(int r=0;r<7;r++)
			for(int c=r;c<7;c++)
			{
				float d = SSEData1m[idx+0] + SSEData1m[idx+1] + SSEData1m[idx+2] + SSEData1m[idx+3];
				H(r,c) = H(c,r) = d;
				idx+=4;
			}
		assert(idx==4*28);
	}

	inline void updateSingleWeighted(
				  float J0, float J1,
				  float J2, float J3,
				  float J4, float J5,
				  float J6,
				  float w, int off=0)
	{
		float* pt=SSEData+off;
		*pt += J0*J0*w; pt+=4; J0*=w;
		*pt += J1*J0; pt+=4;
		*pt += J2*J0; pt+=4;
		*pt += J3*J0; pt+=4;
		*pt += J4*J0; pt+=4;
		*pt += J5*J0; pt+=4;
		*pt += J6*J0; pt+=4;

		*pt += J1*J1*w; pt+=4; J1*=w;
		*pt += J2*J1; pt+=4;
		*pt += J3*J1; pt+=4;
		*pt += J4*J1; pt+=4;
		*pt += J5*J1; pt+=4;
		*pt += J6*J1; pt+=4;

		*pt += J2*J2*w; pt+=4; J2*=w;
		*pt += J3*J2; pt+=4;
		*pt += J4*J2; pt+=4;
		*pt += J5*J2; pt+=4;
		*pt += J6*J2; pt+=4;

		*pt += J3*J3*w; pt+=4; J3*=w;
		*pt += J4*J3; pt+=4;
		*pt += J5*J3; pt+=4;
		*pt += J6*J3; pt+=4;

		*pt += J4*J4*w; pt+=4; J4*=w;
		*pt += J5*J4; pt+=4;
		*pt += J6*J4; pt+=4;

		*pt += J5*J5*w; pt+=4; J5*=w;
		*pt += J6*J5; pt+=4;

		*pt += J6*J6*w; pt+=4;

		num++;
		numIn1++;
		shiftUp(false);
	}

private:
	EIGEN_ALIGN16 float SSEData[4*28];
	EIGEN_ALIGN16 float SSEData1k[4*28];
	EIGEN_ALIGN16 float SSEData1m[4*28];
	float numIn1, numIn1k, numIn1m;

	void shiftUp(bool force)
	{
		if(numIn1 > 1000 || force)
		{
			for(int i=0;i<28;i++)
				_mm_store_ps(SSEData1k+4*i, _mm_add_ps(_mm_load_ps(SSEData+4*i),_mm_load_ps(SSEData1k+4*i)));

			numIn1k+=numIn1;
			numIn1=0;
			std::memset(SSEData,0, sizeof(float)*4*28);
		}

		if(numIn1k > 1000 || force)
		{
			for(int i=0;i<28;i++)
				_mm_store_ps(SSEData1m+4*i, _mm_add_ps(_mm_load_ps(SSEData1k+4*i),_mm_load_ps(SSEData1m+4*i)));

			numIn1m+=numIn1k;
			numIn1k=0;
			std::memset(SSEData1k,0, sizeof(float)*4*28);
		}
	}
};

