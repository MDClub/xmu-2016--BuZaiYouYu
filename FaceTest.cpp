
#define USE_CAMERA		0						/*ʹ������ͷ�꿪��*/
#define DETECT_FRAME	1						/*����֡�ʺ꿪��*/

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

// ��ͼƬ��x��y���괦������Ӧ��ɫ�������ͶȺ�����
int getpixel(IplImage *image, int x, int y, int *h, int *s, int *v){
	*h =(uchar) image->imageData[y *image->widthStep+x * image->nChannels];
	*s =(uchar) image->imageData[y *image->widthStep+ x * image->nChannels + 1];
	*v =(uchar) image->imageData[y *image->widthStep+ x * image->nChannels + 2];
	return 0; 
}

void ShowCurTime(Mat& curFrame);

//����׷�ٺ���
void TrackFace( int x,int y,int w,int h,CvCapture* capture,string strName);

int main( )
{  
	cout<<"������Ҫʶ�����Ƭ������"<<endl;
	//����һ��������
	string srcPath,strName;
	while(cin>>srcPath>>strName)
	{
		Mat srcImg,srcImg_gray,srcFace;
		srcImg=imread(srcPath);
		if(!srcImg.data)
		{
			cout<<"�����д����������룡"<<endl;
		}
		else
		{
			//��ƥ���ͼƬΪ�����ļ���ÿһ֡
			CvCapture* capture;
#if USE_CAMERA
			//�����Ҫ��Ƶʵʱ��⣬��������������
			capture = cvCaptureFromCAM(0);
#else
			//����Ǽ���Ѿ�¼�õ���Ƶ�ļ�������������������
			string strMovie;
			cout<<"��������Ƶ·��"<<endl;
			while(cin>>strMovie)
			{
				if(!(capture = cvCaptureFromAVI(strMovie.c_str())))
				{
					cout<<"Error:��������ȷ��Ƶ·��"<<endl;
				}	
				else
					break;
			}
#endif
			cout<<"ʵʱʶ��... ��Ƶ����������q����ʶ��"<<endl;

			//��������ת��Ϊ�Ҷ�ͼ��
			cvtColor(srcImg,srcImg_gray,COLOR_BGR2GRAY);
			//���Ҫƥ��ͼ���е�����������ͷ�ļ���renlian.h���еĽӿں���
			int * pResults = NULL; 
			pResults = facedetect_multiview((unsigned char*)(srcImg_gray.ptr(0)),srcImg_gray.cols, srcImg_gray.rows,srcImg_gray.step,1.2f, 5, 24);
			//���������е�������ȡ����
			short * p = ((short*)(pResults+1));
			int x = p[0];
			int y = p[1];
			int w = p[2];
			int h = p[3];
			//int neighbors = p[4];
			//int angle = p[5];
			//��С������Χ��ͬʱҲ����С��ƥ��ʱ��
			Rect rect(x+(int)(w/8),y+(int)(w/8),w-(int)(w/4),h-(int)(w/4));
			srcFace=srcImg_gray(rect);
			//��ʾԭʼͷ��
			imshow("ԭʼͷ��",srcFace);

			/*��ȡ��������*/
			//���Surf�ؼ��㡢��ȡѵ��ͼ���������
			vector<KeyPoint>  srcImgkeyPoint;
			Mat  srcImgDescriptor;
			SurfFeatureDetector featureDetector(80);
			featureDetector.detect(srcFace,srcImgkeyPoint);
			SurfDescriptorExtractor featureExtractor;
			featureExtractor.compute(srcFace,srcImgkeyPoint,srcImgDescriptor);
			//��������Flann��������ƥ�����
			FlannBasedMatcher matcher;
			vector<Mat> srcImg_desc_collection(1,srcImgDescriptor);
			matcher.add(srcImg_desc_collection);
			matcher.train();

			Mat curFrame,curFrame_gray;		//��ǰ����֡
			Mat curFace;									//��⵽��face
			while(char(waitKey(1)) !='q')
			{
#if DETECT_FRAME
				//���֡��
				int64 time0=getTickCount();
#endif
				//����ÿһ֡ͼ��
				curFrame= cvQueryFrame(capture);
				if(!curFrame.data) break;

				//��ʾ��ǰʱ��
				ShowCurTime(curFrame);

				//��ÿ֡ͼƬת��Ϊ�Ҷ�ͼ��
				cvtColor(curFrame,curFrame_gray,COLOR_BGR2GRAY);

				//���ͼ�������е�����
				pResults = facedetect_multiview((unsigned char*)(curFrame_gray.ptr(0)),curFrame_gray.cols, curFrame_gray.rows,curFrame_gray.step,1.2f, 5, 24);

				//ʶ������ͼ����ԭ����ͼ��ƥ�䣬���ﵽҪ������ʾ����
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

					//��С������Χ��ͬʱҲ����С��ƥ��ʱ��
					Rect rect(x+(int)(w/8),y+(int)(w/8),w-(int)(w/4),h-(int)(w/4));
					curFace=curFrame_gray(rect);

					//���Surf�ؼ��㡢��ȡѵ��ͼ���������
					vector<KeyPoint> keyPoint;
					Mat Descriptor;
					featureDetector.detect(curFace,keyPoint);
					featureExtractor.compute(curFace,keyPoint,Descriptor);

					//ƥ��ѵ���Ͳ���������
					vector<vector<DMatch>> matches;
					matcher.knnMatch(Descriptor,matches,2);

					//������ʽ�㷨���õ������ƥ���
					vector<DMatch> goodMatches;
					for(unsigned int k=0;k<matches.size();k++)
					{
						if(matches[k][0].distance<0.6*matches[k][1].distance)
							goodMatches.push_back(matches[k][0]);
					}

					//�����������������ﵽҪ������ʾ������Ȼ�����׷��
					if(goodMatches.size()>1)
					{
						hasFound = true;
						//����ȷ��ͷ���÷��������
						rectangle( curFrame, Point(x,y), Point(x+w,y+h), Scalar( 255, 0, 255 ), 3, 8, 0 );

						//��������  
						//����Ϊ�����ص�ͼƬ����������֣����ֵ�λ�ã��ı������½ǣ������壬��С����ɫ  
						putText( curFrame, strName, Point( x,y-5),CV_FONT_HERSHEY_COMPLEX, 0.8, Scalar(0, 250, 0)); 

						//��ʾ�����ȷͷ���ͼ��
						imshow("ʶ�𴰿�",curFrame);

						//����׷�ٺ���������׷��
						TrackFace(x,y,w,h,capture,strName);

						break;
					}
				}
				if(!hasFound)
				{
					//��ʾû��ʶ����ȷͷ��ʱ��ͼ��
					imshow("ʶ�𴰿�",curFrame);
				}
			}

			//�ͷ��ڴ�
			cvDestroyWindow("ʶ�𴰿�");
			cvDestroyWindow("ԭʼͷ��");
			cvReleaseCapture(&capture);

			cout<<"ʶ�������������Ҫʶ�����Ƭ������"<<endl;
		}
	}

	return 0;
}
//��ʾʱ�亯��
void ShowCurTime(Mat& curFrame)
{
	//��ʾʱ��
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

//׷�ٺ���
void TrackFace( int x,int y,int w,int h,CvCapture* capture,string strName)
{
	IplImage* image = 0;
	IplImage* HSV = 0;

	//����׷�ٴ���
	cvNamedWindow("ʶ�𴰿�", CV_WINDOW_AUTOSIZE );

	//Condensation�ṹ���ʼ��
	int DP=2; // ״̬������ά��
	int MP=2; // �۲�������ά��
	int SamplesNum=300; // �������ӵ�����
	CvConDensation* ConDens=cvCreateConDensation( DP, MP, SamplesNum );

	//Condensation�ṹ����һЩ�����ĳ�ʼ��
	CvMat* lowerBound; // �½�
	CvMat* upperBound; // �Ͻ�
	lowerBound = cvCreateMat(2, 1, CV_32F);
	upperBound = cvCreateMat(2, 1, CV_32F);
	//����������������½�Ϊ���ڴ�С
	cvmSet( lowerBound, 0, 0, x); cvmSet( upperBound, 0, 0, x+w);
	cvmSet( lowerBound, 1, 0, y); cvmSet( upperBound, 1, 0,y+h);
	//��ʼ��
	cvConDensInitSampleSet(ConDens, lowerBound, upperBound);


	//Ǩ�ƾ���ĳ�ʼ��
	ConDens->DynamMatr[0]=1.0;ConDens->DynamMatr[1]=0.0;
	ConDens->DynamMatr[2]=0.0;ConDens->DynamMatr[3]=1.0;

	while(char(waitKey(1)) !='q')
	{
		IplImage* frame = 0;

		int X,Y;
		int H,S,V;

		frame = cvQueryFrame( capture );//����ÿһ֡

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

		//���ӵ����Ŷȼ��㣬���Ŷ���Ҫ�Լ���ģ
		int sumX=0;
		int sumY=0;
		int minX=10000;
		int minY=10000;
		for(int i=0; i < SamplesNum; i++)
		{
			X=(int)ConDens->flSamples[i][0];
			Y=(int)ConDens->flSamples[i][1];
			sumX+=X;//sumX��sumY���ڼ���׷�ٴ��ڵ����ĵ�
			sumY+=Y;
			if(X>=0 && X<=1280 && Y>=0 && Y<=720) //���ӵ������ڴ��ڷ�Χ֮��
			{
				if(minX>X)
				{
					minX=X;
				}
				if(minY>Y)
				{
					minY=Y;
				}
				//��ȡHSV�ռ�ֵ
				getpixel(HSV, X, Y, &H, &S, &V);
				if(H<=5 && S>=70) // ��ɫ���ж�
				{ 
					//cvCircle(image, cvPoint(X,Y), 4, CV_RGB(255,0,0), 1);//��ʾ׷�����ӵ�
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

		//��׷�ٵ�ͷ���÷��������
		cvRectangle(image, cvPoint((int)(sumX/SamplesNum)-(int)(0.5*w),(int)(sumY/SamplesNum)-(int)(0.5*h)), 
			cvPoint((int)(sumX/SamplesNum)+(int)(0.5*w),(int)(sumY/SamplesNum)+(int)(0.5*h)), Scalar( 255, 0, 255 ), 3, 8, 0 );
		//cvRectangle(image, cvPoint((int)(sumX/SamplesNum)-(int)(sumX/SamplesNum-minX),(int)(sumY/SamplesNum)-(int)(sumY/SamplesNum-minY)), 
		//	cvPoint((int)(sumX/SamplesNum)+(int)(sumX/SamplesNum-minX),(int)(sumY/SamplesNum)+(int)(sumY/SamplesNum-minY)), Scalar( 255, 0, 255 ), 3, 8, 0 );
		//��������  
		//����Ϊ�����ص�ͼƬ����������֣����ֵ�λ�ã��ı������½ǣ������壬��С����ɫ  
		CvFont font; 
		cvInitFont(&font,CV_FONT_HERSHEY_COMPLEX, 0.5, 1, 1, 1, 8); 
		cvPutText( image,strName.c_str(),cvPoint((int)(sumX/SamplesNum)-(int)(0.5*w),(int)(sumY/SamplesNum)-(int)(0.5*h)-10),&font, Scalar(0, 250, 0));
		//cvPutText( image,strName.c_str(),cvPoint((int)(sumX/SamplesNum)-(int)(sumX/SamplesNum-minX),(int)(sumY/SamplesNum)-(int)(sumY/SamplesNum-minY)-10),&font, Scalar(0, 250, 0)); 

		//�����˲���״̬
		cvConDensUpdateByTime(ConDens);

		//��ʾʱ��
		ShowCurTime((Mat)image);

		//��ʾ׷�ٴ���
		cvShowImage( "ʶ�𴰿�", image );
		waitKey(1);
	}

	//�ͷ��ڴ�
	cvReleaseImage(&image);
	cvReleaseImage(&HSV);
	cvReleaseConDensation(&ConDens);
	cvReleaseMat( &lowerBound );
	cvReleaseMat( &upperBound );
}
