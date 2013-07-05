#include "nonLocalMeans.hpp"


template <class T>
void cvtColorBGR2PLANE_(const Mat& src, Mat& dest, int depth)
{
	vector<Mat> v(3);
	split(src,v);
	dest.create(Size(src.cols, src.rows*3),depth);

	memcpy(dest.data,                    v[0].data,src.size().area()*sizeof(T));
	memcpy(dest.data+src.size().area()*sizeof(T),  v[1].data,src.size().area()*sizeof(T));
	memcpy(dest.data+2*src.size().area()*sizeof(T),v[2].data,src.size().area()*sizeof(T));
}

void cvtColorBGR2PLANE(const Mat& src, Mat& dest)
{
	if(src.channels()!=3)printf("input image must have 3 channels\n");

	if(src.depth()==CV_8U)
	{
		cvtColorBGR2PLANE_<uchar>(src, dest, CV_8U);
	}
	else if(src.depth()==CV_16U)
	{
		cvtColorBGR2PLANE_<ushort>(src, dest, CV_16U);
	}
	if(src.depth()==CV_16S)
	{
		cvtColorBGR2PLANE_<short>(src, dest, CV_16S);
	}
	if(src.depth()==CV_32S)
	{
		cvtColorBGR2PLANE_<int>(src, dest, CV_32S);
	}
	if(src.depth()==CV_32F)
	{
		cvtColorBGR2PLANE_<float>(src, dest, CV_32F);
	}
	if(src.depth()==CV_64F)
	{
		cvtColorBGR2PLANE_<double>(src, dest, CV_64F);
	}
}

double psnr(Mat& src, Mat& ref, Mat& mask)
{
	double mse = norm(src,ref,NORM_L2SQR,mask)/(double)src.size().area();
	double ret = 10.0*log10((255.0*255.0)/mse);
	return ret;
}
double PSNR(Mat& src, Mat& ref, int color_code, int channel)
{
	Mat mask = Mat::ones(src.size(),CV_8U);
	CV_Assert(src.channels()==ref.channels());

	if(src.channels()==1)
	{
		return psnr(src,ref,mask);
	}
	else
	{
		Mat s1,r1;
		cvtColor(src,s1,color_code);
		cvtColor(ref,r1,color_code);
		vector<Mat> sv;
		vector<Mat> rv;
		split(src,sv);
		split(ref,rv);
		return psnr(sv[channel],rv[channel],mask);
	}
}

double PSNR_32f28u(Mat& src, Mat& ref, int color_code, int channel)
{
	Mat mask = Mat::ones(src.size(),CV_8U);
	CV_Assert(src.channels()==ref.channels());
	Mat src8u,ref8u;
	src.convertTo(src8u,CV_8U);
	ref.convertTo(ref8u,CV_8U);
	if(src.channels()==1)
	{
		return psnr(src8u,ref8u,mask);
	}
	else
	{
		Mat s1,r1;
		cvtColor(src8u,s1,color_code);
		cvtColor(ref8u,r1,color_code);
		vector<Mat> sv;
		vector<Mat> rv;
		split(src,sv);
		split(ref,rv);
		return psnr(sv[channel],rv[channel],mask);
	}
}

void addNoiseMono_nf(Mat& src, Mat& dest,double sigma)
{
    Mat s;
	src.convertTo(s,CV_32S);
    Mat n(s.size(),CV_32S);
    randn(n,0,sigma);
    Mat temp = s+n;
	temp.convertTo(dest,src.type());
}
void addNoiseMono_f(Mat& src, Mat& dest,double sigma)
{
    Mat s;
	src.convertTo(s,CV_64F);
    Mat n(s.size(),CV_64F);
    randn(n,0,sigma);
    Mat temp = s+n;
	temp.convertTo(dest,src.type());
}
void addNoiseMono(Mat& src, Mat& dest,double sigma)
{
	if(src.type()==CV_32F || src.type()==CV_64F)
	{
		addNoiseMono_f(src,dest,sigma);
	}
	else
	{
		addNoiseMono_nf(src,dest,sigma);
	}
}
void addNoise(Mat&src, Mat& dest, double sigma)
{
    if(src.channels()==1)
    {
        addNoiseMono(src,dest,sigma);
        
        return;
    }
    else
    {
        vector<Mat> s(src.channels());
        vector<Mat> d(src.channels());
        split(src,s);
        for(int i=0;i<src.channels();i++)
        {
            addNoiseMono(s[i],d[i],sigma);
        }
        cv::merge(d,dest);
    }
}