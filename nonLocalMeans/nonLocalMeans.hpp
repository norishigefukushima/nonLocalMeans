#pragma once


#include "opencv/cv.h"

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
using namespace std;
using namespace cv;

void nonLocalMeansFilterBase(Mat& src, Mat& dest, int templeteWindowSize, int searchWindowSize, double h, double sigma=0.0);
void nonLocalMeansFilter(Mat& src, Mat& dest, int templeteWindowSize, int searchWindowSize, double h, double sigma=0.0);

void cvtColorBGR2PLANE(const Mat& src, Mat& dest);

double PSNR(Mat& src, Mat& ref, int color_code=CV_BGR2YUV, int channel=0);
double PSNR_32f28u(Mat& src, Mat& ref, int color_code=CV_BGR2YUV, int channel=0);
void addNoise(Mat&src, Mat& dest, double sigma);