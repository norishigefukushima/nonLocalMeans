#include "nonLocalMeans.hpp"

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/photo/photo.hpp>

#define CV_VERSION_NUMBER CVAUX_STR(CV_MAJOR_VERSION) CVAUX_STR(CV_MINOR_VERSION) CVAUX_STR(CV_SUBMINOR_VERSION)

#ifdef _DEBUG
#pragma comment(lib, "opencv_imgproc"CV_VERSION_NUMBER"d.lib")
#pragma comment(lib, "opencv_highgui"CV_VERSION_NUMBER"d.lib")
#pragma comment(lib, "opencv_core"CV_VERSION_NUMBER"d.lib")
#pragma comment(lib, "opencv_photo"CV_VERSION_NUMBER"d.lib")
#else
#pragma comment(lib, "opencv_photo"CV_VERSION_NUMBER".lib")
#pragma comment(lib, "opencv_imgproc"CV_VERSION_NUMBER".lib")
#pragma comment(lib, "opencv_highgui"CV_VERSION_NUMBER".lib")
#pragma comment(lib, "opencv_core"CV_VERSION_NUMBER".lib")
#endif

void test8uc1(bool show=false)
{
	Mat src = imread("image.png",0);
	Mat noise;
	Mat dest;
	int64 pre;

	float sigma = 15;
	cout<<"gray"<<endl;
	cout<<"sigma: "<<sigma<<endl;

	cout<<"RAW: "<<endl;
	addNoise(src,noise,sigma);
	cout<<PSNR(src,noise)<<"dB"<<endl;
	cout<<endl;

	cout<<"NML opencv: "<<endl;
	pre = getTickCount();
	fastNlMeansDenoising(noise,dest,sigma,3,7);
	cout<<(getTickCount()-pre)/(getTickFrequency())*1000<<"ms"<<endl;
	cout<<PSNR(src,dest)<<"dB"<<endl;
	cout<<endl;

	cout<<"NML base: "<<endl;
	pre = getTickCount();
	nonLocalMeansFilterBase(noise,dest,3,7,sigma);
	cout<<(getTickCount()-pre)/(getTickFrequency())*1000<<"ms"<<endl;
	cout<<PSNR(src,dest)<<"dB"<<endl;
	cout<<endl;

	cout<<"NML sse4: "<<endl;
	pre = getTickCount();
	nonLocalMeansFilter(noise,dest,3,7,sigma);
	cout<<(getTickCount()-pre)/(getTickFrequency())*1000<<"ms"<<endl;
	cout<<PSNR(src,dest)<<"dB"<<endl;
	cout<<endl;

	if(show)
	{
		imshow("noise",noise);
		imshow("NLM",dest);
		waitKey();
	}
}

void test8uc3(bool show=false)
{
	Mat src = imread("image.png");
	Mat noise;
	Mat dest;
	int64 pre;

	float sigma = 15;
	cout<<"color"<<endl;
	cout<<"sigma: "<<sigma<<endl;

	cout<<"RAW: "<<endl;
	addNoise(src,noise,sigma);
	cout<<PSNR(src,noise)<<"dB"<<endl;
	cout<<endl;

	cout<<"NML opencv: "<<endl;
	pre = getTickCount();
	fastNlMeansDenoisingColored(noise,dest,sigma,sigma,3,7);
	cout<<(getTickCount()-pre)/(getTickFrequency())*1000<<"ms"<<endl;
	cout<<PSNR(src,dest)<<"dB"<<endl;
	cout<<endl;

	cout<<"NML base: "<<endl;
	pre = getTickCount();
	nonLocalMeansFilterBase(noise,dest,3,7,2*sigma);
	cout<<(getTickCount()-pre)/(getTickFrequency())*1000<<"ms"<<endl;
	cout<<PSNR(src,dest)<<"dB"<<endl;
	cout<<endl;

	cout<<"NML sse4: "<<endl;
	pre = getTickCount();
	nonLocalMeansFilter(noise,dest,3,7,2*sigma);
	cout<<(getTickCount()-pre)/(getTickFrequency())*1000<<"ms"<<endl;
	cout<<PSNR(src,dest)<<"dB"<<endl;
	cout<<endl;

	if(show)
	{
		imshow("noise",noise);
		imshow("NLM",dest);
		waitKey();
	}
}


void test32fc1(bool show=false)
{
	Mat src_ = imread("image.png",0);
	Mat noise_;
	Mat dest;
	int64 pre;

	float sigma = 15;
	cout<<"gray"<<endl;
	cout<<"sigma: "<<sigma<<endl;

	cout<<"RAW: "<<endl;
	addNoise(src_,noise_,sigma);
	Mat src,noise;
	src_.convertTo(src,CV_32F);
	noise_.convertTo(noise,CV_32F);

	cout<<PSNR_32f28u(src,noise)<<"dB"<<endl;
	cout<<endl;

	cout<<"NML base: "<<endl;
	pre = getTickCount();
	nonLocalMeansFilterBase(noise,dest,3,7,sigma);
	cout<<(getTickCount()-pre)/(getTickFrequency())*1000<<"ms"<<endl;
	cout<<PSNR_32f28u(src,dest)<<"dB"<<endl;
	cout<<endl;

	cout<<"NML sse4: "<<endl;
	pre = getTickCount();
	nonLocalMeansFilter(noise,dest,3,7,sigma);
	cout<<(getTickCount()-pre)/(getTickFrequency())*1000<<"ms"<<endl;
	cout<<PSNR_32f28u(src,dest)<<"dB"<<endl;
	cout<<endl;

	if(show)
	{
		imshow("noise",noise);
		imshow("NLM",dest);
		waitKey();
	}
}

void test32fc3(bool show=false)
{
	Mat src_ = imread("image.png");
	Mat noise_;
	Mat dest;
	int64 pre;

	float sigma = 15;
	cout<<"color"<<endl;
	cout<<"sigma: "<<sigma<<endl;

	cout<<"RAW: "<<endl;
	addNoise(src_,noise_,sigma);
	Mat src,noise;
	src_.convertTo(src,CV_32F);
	noise_.convertTo(noise,CV_32F);

	cout<<PSNR_32f28u(src,noise)<<"dB"<<endl;
	cout<<endl;

	cout<<"NML base: "<<endl;
	pre = getTickCount();
	nonLocalMeansFilterBase(noise,dest,3,7,2*sigma);
	cout<<(getTickCount()-pre)/(getTickFrequency())*1000<<"ms"<<endl;
	cout<<PSNR_32f28u(src,dest)<<"dB"<<endl;
	cout<<endl;

	cout<<"NML sse4: "<<endl;
	pre = getTickCount();
	nonLocalMeansFilter(noise,dest,3,7,2*sigma);
	cout<<(getTickCount()-pre)/(getTickFrequency())*1000<<"ms"<<endl;
	cout<<PSNR_32f28u(src,dest)<<"dB"<<endl;
	cout<<endl;

	if(show)
	{
		imshow("noise",noise);
		imshow("NLM",dest);
		waitKey();
	}
}
int main()
{
	test8uc1();
	test8uc3();

	test32fc1();//"NML opencv is not support 32F "
	test32fc3();//"NML opencv is not support 32F "

	return 0;
}