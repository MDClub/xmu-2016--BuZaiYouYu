
#define USE_CAMERA		0						/*使用摄像头宏开关*/
#define DETECT_FRAME	1						/*测试帧率宏开关*/

#include <opencv2/opencv.hpp>
#include "opencv2/highgui/highgui.hpp"
#include <opencv2/nonfree/nonfree.hpp>
#include "opencv2/features2d/features2d.hpp"
#include <opencv.hpp>
#include <time.h>
#include "afx.h"
#include "cv.h"
#include "cvAux.h"
#include "highgui.h"
#include "cxcore.h"
#include <stdio.h>
#include <ctype.h>

#include "FaceTest.h"
#pragma comment(lib,"libfacedetect.lib")

using namespace cv;
using namespace std;

// 从图片的x、y坐标处返回相应的色调、饱和度和亮度
int getpixel(IplImage *image, int x, int y, int *h, int *s, int *v){
	*h =(uchar) image->imageData[y *image->widthStep+x * image->nChannels];
	*s =(uchar) image->imageData[y *image->widthStep+ x * image->nChannels + 1];
	*v =(uchar) image->imageData[y *image->widthStep+ x * image->nChannels + 2];
	return 0; 
}

void ShowCurTime(Mat& curFrame);

//声明追踪函数
void TrackFace( int x,int y,int w,int h,CvCapture* capture,string strName);

int main( )
{  
	cout<<"请输入要识别的照片和名字"<<endl;
	//读入一张正面照
	string srcPath,strName;
	while(cin>>srcPath>>strName)
	{
		Mat srcImg,srcImg_gray,srcFace;
		srcImg=imread(srcPath);
		if(!srcImg.data)
		{
			cout<<"输入有错，请重新输入！"<<endl;
		}
		else
		{
			//被匹配的图片为视屏文件的每一帧
			CvCapture* capture;
#if USE_CAMERA
			//如果想要视频实时检测，用下面这条代码
			capture = cvCaptureFromCAM(0);
#else
			//如果是检测已经录好的视频文件，请用下面这条代码
			string strMovie;
			cout<<"请输入视频路径"<<endl;
			while(cin>>strMovie)
			{
				if(!(capture = cvCaptureFromAVI(strMovie.c_str())))
				{
					cout<<"Error:请输入正确视频路径"<<endl;
				}	
				else
					break;
			}
#endif
			cout<<"实时识别... 视频窗口中输入q结束识别"<<endl;

			//将正面照转化为灰度图像
			cvtColor(srcImg,srcImg_gray,COLOR_BGR2GRAY);
			//检测要匹配图像中的人脸，调用头文件“renlian.h”中的接口函数
			int * pResults = NULL; 
			pResults = facedetect_multiview((unsigned char*)(srcImg_gray.ptr(0)),srcImg_gray.cols, srcImg_gray.rows,srcImg_gray.step,1.2f, 5, 24);
			//将正面照中的人脸截取出来
			short * p = ((short*)(pResults+1));
			int x = p[0];
			int y = p[1];
			int w = p[2];
			int h = p[3];
			//int neighbors = p[4];
			//int angle = p[5];
			//缩小脸部范围，同时也就缩小了匹配时间
			Rect rect(x+(int)(w/8),y+(int)(w/8),w-(int)(w/4),h-(int)(w/4));
			srcFace=srcImg_gray(rect);
			//显示原始头像
			imshow("原始头像",srcFace);

			/*提取人脸特征*/
			//检测Surf关键点、提取训练图像的描述符
			vector<KeyPoint>  srcImgkeyPoint;
			Mat  srcImgDescriptor;
			SurfFeatureDetector featureDetector(80);
			featureDetector.detect(srcFace,srcImgkeyPoint);
			SurfDescriptorExtractor featureExtractor;
			featureExtractor.compute(srcFace,srcImgkeyPoint,srcImgDescriptor);
			//创建基于Flann的描述符匹配对象
			FlannBasedMatcher matcher;
			vector<Mat> srcImg_desc_collection(1,srcImgDescriptor);
			matcher.add(srcImg_desc_collection);
			matcher.train();

			Mat curFrame,curFrame_gray;		//当前视屏帧
			Mat curFace;									//检测到的face
			while(char(waitKey(1)) !='q')
			{
#if DETECT_FRAME
				//检测帧率
				int64 time0=getTickCount();
#endif
				//读入每一帧图像
				curFrame= cvQueryFrame(capture);
				if(!curFrame.data) break;

				//显示当前时间
				ShowCurTime(curFrame);

				//将每帧图片转化为灰度图像
				cvtColor(curFrame,curFrame_gray,COLOR_BGR2GRAY);

				//检测图像中所有的人脸
				pResults = facedetect_multiview((unsigned char*)(curFrame_gray.ptr(0)),curFrame_gray.cols, curFrame_gray.rows,curFrame_gray.step,1.2f, 5, 24);

				//识别人脸图像并与原人脸图像匹配，若达到要求则显示出来
				bool hasFound = false;
				for(int i = 0; i < (pResults ? *pResults : 0); i++)
				{
					short * p = (short*)(pResults+1)+6*i;
					int x = p[0];
					int y = p[1];
					int w = p[2];
					int h = p[3];
					//int neighbors = p[4];
					//int angle = p[5];

					//缩小脸部范围，同时也就缩小了匹配时间
					Rect rect(x+(int)(w/8),y+(int)(w/8),w-(int)(w/4),h-(int)(w/4));
					curFace=curFrame_gray(rect);

					//检测Surf关键点、提取训练图像的描述符
					vector<KeyPoint> keyPoint;
					Mat Descriptor;
					featureDetector.detect(curFace,keyPoint);
					featureExtractor.compute(curFace,keyPoint,Descriptor);

					//匹配训练和测试描述符
					vector<vector<DMatch>> matches;
					matcher.knnMatch(Descriptor,matches,2);

					//根据劳式算法，得到优秀的匹配点
					vector<DMatch> goodMatches;
					for(unsigned int k=0;k<matches.size();k++)
					{
						if(matches[k][0].distance<0.6*matches[k][1].distance)
							goodMatches.push_back(matches[k][0]);
					}

					//如果优秀特征点个数达到要求则显示出来，然后进行追踪
					/*注意：优秀匹配的点的阀值选取越大，检测人脸的正确率越高，但是，检测范围会减小；
					        反之，优秀匹配点的阀值选取的越小，检测范围会变大，检测的正确率会降低。
					        这里默认的是1，是因为用项目的包含的视频检测会有很好的效果，如果用摄像头
					        实时检测请调高阀值，以便提高真确的识别率*/
					if(goodMatches.size()>1)
					{
						hasFound = true;
						//把正确的头像用方框框起来
						rectangle( curFrame, Point(x,y), Point(x+w,y+h), Scalar( 255, 0, 255 ), 3, 8, 0 );

						//插入文字  
						//参数为：承载的图片，插入的文字，文字的位置（文本框左下角），字体，大小，颜色  
						putText( curFrame, strName, Point( x,y-5),CV_FONT_HERSHEY_COMPLEX, 0.8, Scalar(0, 250, 0)); 

						//显示框出真确头像的图像
						imshow("识别窗口",curFrame);

						//调用追踪函数，进行追踪
						TrackFace(x,y,w,h,capture,strName);

						break;
					}
				}
				if(!hasFound)
				{
					//显示没有识别正确头像时的图像
					imshow("识别窗口",curFrame);
				}
			}

			//释放内存
			cvDestroyWindow("识别窗口");
			cvDestroyWindow("原始头像");
			cvReleaseCapture(&capture);

			cout<<"识别结束，请输入要识别的照片和名字"<<endl;
		}
	}

	return 0;
}
//显示时间函数
void ShowCurTime(Mat& curFrame)
{
	//显示时间
	SYSTEMTIME systime;
	GetLocalTime(&systime); 
	unsigned short year = systime.wYear;
	unsigned short month = systime.wMonth;
	unsigned short day = systime.wDay;
	unsigned short hour = systime.wHour;
	unsigned short minute = systime.wMinute;
	unsigned short second = systime.wSecond;
	String time=format("%u/%u/%u--%u:%u:%u",year,month,day,hour,minute,second);
	putText(curFrame, time, Point(15,25),CV_FONT_HERSHEY_COMPLEX, 0.8, Scalar(0, 250, 0)); 
}

//追踪函数
void TrackFace( int x,int y,int w,int h,CvCapture* capture,string strName)
{
	IplImage* image = 0;
	IplImage* HSV = 0;

	//创建追踪窗口
	cvNamedWindow("识别窗口", CV_WINDOW_AUTOSIZE );

	//Condensation结构体初始化
	int DP=2; // 状态向量的维数
	int MP=2; // 观测向量的维数
	int SamplesNum=300; // 样本粒子的数量
	CvConDensation* ConDens=cvCreateConDensation( DP, MP, SamplesNum );

	//Condensation结构体中一些参数的初始化
	CvMat* lowerBound; // 下界
	CvMat* upperBound; // 上界
	lowerBound = cvCreateMat(2, 1, CV_32F);
	upperBound = cvCreateMat(2, 1, CV_32F);
	//设置粒子坐标的上下界为窗口大小
	cvmSet( lowerBound, 0, 0, x); cvmSet( upperBound, 0, 0, x+w);
	cvmSet( lowerBound, 1, 0, y); cvmSet( upperBound, 1, 0,y+h);
	//初始化
	cvConDensInitSampleSet(ConDens, lowerBound, upperBound);


	//迁移矩阵的初始化
	ConDens->DynamMatr[0]=1.0;ConDens->DynamMatr[1]=0.0;
	ConDens->DynamMatr[2]=0.0;ConDens->DynamMatr[3]=1.0;

	while(char(waitKey(1)) !='q')
	{
		IplImage* frame = 0;

		int X,Y;
		int H,S,V;

		frame = cvQueryFrame( capture );//读入每一帧

		if( !frame )
		{
			break;
		}

		if( !image )
		{
			image = cvCreateImage( cvGetSize(frame), 8, 3 );
			image->origin = frame->origin;
			HSV = cvCreateImage( cvGetSize(frame), 8, 3 );
			HSV->origin = frame->origin;
		}

		cvCopy( frame, image, 0 );
		cvCvtColor(image ,HSV , CV_BGR2HSV);

		//粒子的置信度计算，置信度需要自己建模
		int sumX=0;
		int sumY=0;
		int minX=10000;
		int minY=10000;
		for(int i=0; i < SamplesNum; i++)
		{
			X=(int)ConDens->flSamples[i][0];
			Y=(int)ConDens->flSamples[i][1];
			sumX+=X;//sumX和sumY用于计算追踪窗口的中心点
			sumY+=Y;
			if(X>=0 && X<=1280 && Y>=0 && Y<=720) //粒子的坐标在窗口范围之内
			{
				if(minX>X)
				{
					minX=X;
				}
				if(minY>Y)
				{
					minY=Y;
				}
				//获取HSV空间值
				getpixel(HSV, X, Y, &H, &S, &V);
				if(H<=5 && S>=70) // 肤色的判定
				{ 
					//cvCircle(image, cvPoint(X,Y), 4, CV_RGB(255,0,0), 1);//显示追踪粒子点
					ConDens->flConfidence[i]=1.0;
				}
				else
				{
					ConDens->flConfidence[i]=0.0;
				}
			}
			else
			{
				ConDens->flConfidence[i]=0.0;
			}
		}

		//把追踪的头像用方框框起来
		cvRectangle(image, cvPoint((int)(sumX/SamplesNum)-(int)(0.5*w),(int)(sumY/SamplesNum)-(int)(0.5*h)), 
			cvPoint((int)(sumX/SamplesNum)+(int)(0.5*w),(int)(sumY/SamplesNum)+(int)(0.5*h)), Scalar( 255, 0, 255 ), 3, 8, 0 );
		//cvRectangle(image, cvPoint((int)(sumX/SamplesNum)-(int)(sumX/SamplesNum-minX),(int)(sumY/SamplesNum)-(int)(sumY/SamplesNum-minY)), 
		//	cvPoint((int)(sumX/SamplesNum)+(int)(sumX/SamplesNum-minX),(int)(sumY/SamplesNum)+(int)(sumY/SamplesNum-minY)), Scalar( 255, 0, 255 ), 3, 8, 0 );
		//插入文字  
		//参数为：承载的图片，插入的文字，文字的位置（文本框左下角），字体，大小，颜色  
		CvFont font; 
		cvInitFont(&font,CV_FONT_HERSHEY_COMPLEX, 0.5, 1, 1, 1, 8); 
		cvPutText( image,strName.c_str(),cvPoint((int)(sumX/SamplesNum)-(int)(0.5*w),(int)(sumY/SamplesNum)-(int)(0.5*h)-10),&font, Scalar(0, 250, 0));
		//cvPutText( image,strName.c_str(),cvPoint((int)(sumX/SamplesNum)-(int)(sumX/SamplesNum-minX),(int)(sumY/SamplesNum)-(int)(sumY/SamplesNum-minY)-10),&font, Scalar(0, 250, 0)); 

		//更新滤波器状态
		cvConDensUpdateByTime(ConDens);

		//显示时间
		ShowCurTime((Mat)image);

		//显示追踪窗口
		cvShowImage( "识别窗口", image );
		waitKey(1);
	}

	//释放内存
	cvReleaseImage(&image);
	cvReleaseImage(&HSV);
	cvReleaseConDensation(&ConDens);
	cvReleaseMat( &lowerBound );
	cvReleaseMat( &upperBound );
}
