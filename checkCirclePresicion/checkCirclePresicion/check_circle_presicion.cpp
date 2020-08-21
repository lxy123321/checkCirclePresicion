#include<iostream>
#include<vector>
#include<string>
#include<fstream>
#include <opencv2/opencv.hpp>

#include"SharedHead.h"
#include"TEllipse.h"
#include"TEllipseCollection.h"


using namespace std;
using namespace cv;

const int width = 1280;
const int height = 1024;
Size imageSize(width, height);
Mat colorImage;

void rectify(const Mat& imgL, const Mat& imgR, Mat& rectifyL, Mat& rectifyR, Mat& Q)   
{
	FileStorage fCali("stereo_CaliResult.xml", FileStorage::READ);
	Mat M1, D1, M2, D2, R, T;
	fCali["M1"] >> M1;
	fCali["D1"] >> D1;
	fCali["M2"] >> M2;
	fCali["D2"] >> D2;
	fCali["R"] >> R;
	fCali["T"] >> T;
	fCali.release();

	Mat P1, P2, R1, R2;
	stereoRectify(M1, D1, M2, D2, imageSize, R, T, R1, R2, P1, P2, Q);  //Q是输出参数
	Mat map11, map12, map21, map22;
	initUndistortRectifyMap(M1, D1, R1, P1, imageSize, CV_32FC1, map11, map12);
	initUndistortRectifyMap(M2, D2, R2, P2, imageSize, CV_32FC1, map21, map22);
	remap(imgL, rectifyL, map11, map12, INTER_LINEAR);
	remap(imgR, rectifyR, map21, map22, INTER_LINEAR);

	//for test检验校正结果
	/*Mat part;
	Mat pair(height, width * 2, CV_8UC3, Scalar::all(0));

	part = pair.colRange(0, width);
	cvtColor(rectifyL, part, CV_GRAY2BGR);
	part = pair.colRange(width, width * 2);
	cvtColor(rectifyR, part, CV_GRAY2BGR);

	for (int i = 0; i < height; i += 64)
	{
	line(pair, Point(0, i), Point(width * 2, i), Scalar(0, 0, 255));
	}
	imshow("stereoRectify", pair);
	waitKey(0);*/
}

void convertParam(const Mat& Mk1, const Mat& Mk2, double M1[][4], double M2[][4])
{
	for (int i = 0; i < 3; i++)
	{
		for (int j = 0; j <= 3; j++)
		{
			M1[i][j] = Mk1.at<double>(i, j);
		}
	}

	for (int i = 0; i < 3; i++)
	{
		for (int j = 0; j <= 3; j++)
		{
			M2[i][j] = Mk2.at<double>(i, j);
		}
	}
}

Mat getTarget(const Mat& img,const Mat& model)
{
	Mat temp;
	matchTemplate(img, model, temp, TM_CCOEFF_NORMED);

	Point MaxLoc;
	minMaxLoc(temp, NULL, NULL, NULL, &MaxLoc);
	Mat zero = Mat::zeros(img.rows, img.cols, CV_8U);
	Mat result(zero, Rect(MaxLoc.x, MaxLoc.y, model.cols, model.rows));
	result.setTo(255);
	/*imshow("zero", zero);
	waitKey();*/
	return zero;
}

int Find_counter2(cv::Mat img1, TEllipseCollection &ellpco)
{
	TEllipse myellipse;
	vector<vector<Point>> contours;    //轮廓的点
	findContours(img1, contours, RETR_LIST, CHAIN_APPROX_NONE);

	int num = contours.size();
	Mat colorImage;
	cvtColor(img1, colorImage, CV_GRAY2BGR);
	vector<vector<cv::Point>> contourDst = contours;

	Point pointTemp;
	vector<Point> pointVector;
	double area;
	vector<float> radius;
	radius.resize(contourDst.size());
	vector<cv::Point2f> ptCenter;
	ptCenter.resize(contourDst.size());
	for (int j = 0; j<contourDst.size(); j++)
	{
		int count = contourDst[j].size(); // number of points
		pointVector.clear();
		area = fabs(cv::contourArea(contourDst[j]));//获得椭圆的面积
		cv::minEnclosingCircle(contourDst[j], ptCenter[j], radius[j]);//计算轮廓的最小外接圆
		float cd = area / (PI * radius[j] * radius[j]);

		if (count>10 && (count<2000) && area<2000 && area>10 && radius[j]>5 && radius[j]<100 && cd>0.5)//判断标准根据图像大小做调整 
		{
			cv::drawContours(colorImage, contourDst, j, cv::Scalar(0, 0, 255));//R:绘制轮廓（包括椭圆和噪声的轮廓）
			cv::RotatedRect box = cv::fitEllipse(contourDst[j]);//R:拟合当前轮廓
			cv::Point center;
			cv::Size size;
			center.x = int(box.center.x + 0.5);
			center.y = int(box.center.y + 0.5);
			size.width = int(box.size.width / 2 + 0.5);
			size.height = int(box.size.height / 2 + 0.5);
			myellipse.m_dCenterpx = box.center.x;//像面坐标
			myellipse.m_dCenterpy = box.center.y;//像面坐标
			myellipse.m_dArea = area;
			myellipse.nid = j;//第j个
			ellpco.ellipses.push_back(myellipse);//R:存储了椭圆中心的坐标，长轴，短轴（或者认为是宽和高的一半），面积
			int x = (int)myellipse.m_dCenterpx;
			int y = (int)myellipse.m_dCenterpy;
			cv::line(colorImage, Point(x - 5, y), Point(x + 5, y), CV_RGB(0, 0, 255));//R:在中心画十字叉 对于小图的标定
			cv::line(colorImage, Point(x, y - 5), Point(x, y + 5), CV_RGB(0, 0, 255));
			//imshow("contours.bmp", colorImage);
			//waitKey();
		}
	}
	num = ellpco.Size();
	return num;//R:返回值是符合要求的椭圆的数量
}

void Find_Topology(TEllipseCollection &ellpco, double DX, double DY, vector<C4DPointD>& m_vecFeaturePts,const Mat &img)
{
	vector<TEllipse> FourBigCircle;
	vector<TEllipse> SmallCircle;
	vector<TEllipse> ellipses(ellpco.ellipses);//括号前的是括号里的一个副本
	ellpco.Clear();

	// 2. 对图像边长和面积进行排序，找到调变点，查找调变点的信息 进行噪声剔出
	{
		//////////////////////////////////////////////////////////////////////////
		set<OrderRuler> CircleLengthOrder;
		OrderRuler temp;
		std::vector<TEllipse>::iterator iter = ellipses.begin();
		double area;
		int nsize = ellipses.size();

		while (iter != ellipses.end())
		{

			temp.area = iter->m_dArea;
			temp.nid = iter->nid;
			CircleLengthOrder.insert(temp);
			iter++;
		}
		//////////////////////////////////////////////////////////////////////////

		std::set<OrderRuler>::iterator iterj = CircleLengthOrder.begin();
		for (int i = 0; i < nsize - 4; i++)
		{
			iter = ellipses.begin();
			while (iter != ellipses.end())
			{
				if (iter->nid == iterj->nid)
				{
					SmallCircle.push_back(*iter);
					break;
				}
				iter++;
			}
			iterj++;
		}

		for (int i = 0; i<4; i++)
		{
			iter = ellipses.begin();
			while (iter != ellipses.end())
			{
				if (iter->nid == iterj->nid)
				{
					FourBigCircle.push_back(*iter);
					break;
				}
				iter++;
			}
			iterj++;
		}
	}

	if (FourBigCircle.size() != 4)
	{
		cout << "大圆标记没有找到，请仔细检查图像" << endl;
	}

	// 3. 为四个大圆排序，编号
	{
		int norigin = 0, npid1 = 1, npid2 = 2, npid3 = 3;

		std::vector<TEllipse> FourBigCircleTemp;
		FourBigCircleTemp.reserve(100);

		std::vector<TEllipse>::iterator iter = FourBigCircle.begin();
		while (iter != FourBigCircle.end())
		{
			FourBigCircleTemp.push_back(*iter);
			iter++;
		}

		CVect v1(FourBigCircleTemp[0], FourBigCircleTemp[1]);
		CVect v2(FourBigCircleTemp[0], FourBigCircleTemp[2]);
		CVect v3(FourBigCircleTemp[0], FourBigCircleTemp[3]);

		// 大圆0是斜线以外的点
		if (sinanglefab(v1, v2) > 0.052 && sinanglefab(v1, v3) > 0.052 && sinanglefab(v2, v3) > 0.052)		// 3度
		{
			npid1 = 0;
			set<DistCompare> distvector;
			DistCompare disttemp;
			disttemp.SetValue(CVect(FourBigCircleTemp[1], FourBigCircleTemp[2]), -1,
				CVect(FourBigCircleTemp[0], FourBigCircleTemp[1]), 1);
			distvector.insert(disttemp);
			disttemp.SetValue(CVect(FourBigCircleTemp[1], FourBigCircleTemp[2]), -1,
				CVect(FourBigCircleTemp[0], FourBigCircleTemp[2]), 2);
			distvector.insert(disttemp);
			disttemp.SetValue(CVect(FourBigCircleTemp[1], FourBigCircleTemp[2]), -1,
				CVect(FourBigCircleTemp[0], FourBigCircleTemp[3]), 3);
			distvector.insert(disttemp);
			std::set<DistCompare>::iterator iterd1 = distvector.begin();
			npid3 = iterd1->nid2;
			iterd1++;
			norigin = iterd1->nid2;
			iterd1++;
			npid2 = iterd1->nid2;
		}
		// 大圆0是斜线上的点
		else
		{
			if (sinanglefab(v1, v2) < 0.052)
			{
				npid1 = 3;
				set<DistCompare> distvector;
				DistCompare disttemp;
				disttemp.SetValue(CVect(FourBigCircleTemp[1], FourBigCircleTemp[2]), -1,
					CVect(FourBigCircleTemp[3], FourBigCircleTemp[1]), 1);
				distvector.insert(disttemp);
				disttemp.SetValue(CVect(FourBigCircleTemp[1], FourBigCircleTemp[2]), -1,
					CVect(FourBigCircleTemp[3], FourBigCircleTemp[2]), 2);
				distvector.insert(disttemp);
				disttemp.SetValue(CVect(FourBigCircleTemp[1], FourBigCircleTemp[2]), -1,
					CVect(FourBigCircleTemp[3], FourBigCircleTemp[0]), 0);
				distvector.insert(disttemp);
				std::set<DistCompare>::iterator iterd1 = distvector.begin();
				npid3 = iterd1->nid2;
				iterd1++;
				norigin = iterd1->nid2;
				iterd1++;
				npid2 = iterd1->nid2;
			}
			else if (sinanglefab(v1, v3) < 0.052)
			{
				npid1 = 2;
				set<DistCompare> distvector;
				DistCompare disttemp;
				disttemp.SetValue(CVect(FourBigCircleTemp[1], FourBigCircleTemp[3]), -1,
					CVect(FourBigCircleTemp[2], FourBigCircleTemp[0]), 0);
				distvector.insert(disttemp);
				disttemp.SetValue(CVect(FourBigCircleTemp[1], FourBigCircleTemp[3]), -1,
					CVect(FourBigCircleTemp[2], FourBigCircleTemp[1]), 1);
				distvector.insert(disttemp);
				disttemp.SetValue(CVect(FourBigCircleTemp[1], FourBigCircleTemp[3]), -1,
					CVect(FourBigCircleTemp[2], FourBigCircleTemp[3]), 3);
				distvector.insert(disttemp);
				std::set<DistCompare>::iterator iterd1 = distvector.begin();
				npid3 = iterd1->nid2;
				iterd1++;
				norigin = iterd1->nid2;
				iterd1++;
				npid2 = iterd1->nid2;
			}

			else if (sinanglefab(v2, v3) < 0.052)
			{
				npid1 = 1;
				set<DistCompare> distvector;
				DistCompare disttemp;
				disttemp.SetValue(CVect(FourBigCircleTemp[2], FourBigCircleTemp[3]), -1,
					CVect(FourBigCircleTemp[1], FourBigCircleTemp[0]), 0);
				distvector.insert(disttemp);
				disttemp.SetValue(CVect(FourBigCircleTemp[2], FourBigCircleTemp[3]), -1,
					CVect(FourBigCircleTemp[1], FourBigCircleTemp[2]), 2);
				distvector.insert(disttemp);
				disttemp.SetValue(CVect(FourBigCircleTemp[2], FourBigCircleTemp[3]), -1,
					CVect(FourBigCircleTemp[1], FourBigCircleTemp[3]), 3);
				distvector.insert(disttemp);
				std::set<DistCompare>::iterator iterd1 = distvector.begin();
				npid3 = iterd1->nid2;
				iterd1++;
				norigin = iterd1->nid2;
				iterd1++;
				npid2 = iterd1->nid2;
			}
		}
		// 四个点按照原点，最远点，两近点（离远点最近在前）的顺序排列好
		FourBigCircleTemp[norigin].nxid = 0;
		FourBigCircleTemp[norigin].nyid = 0;
		FourBigCircleTemp[npid1].nxid = 2;
		FourBigCircleTemp[npid1].nyid = 0;
		FourBigCircleTemp[npid2].nxid = 1;
		FourBigCircleTemp[npid2].nyid = 1;
		FourBigCircleTemp[npid3].nxid = -1;
		FourBigCircleTemp[npid3].nyid = -1;

		FourBigCircle.clear();

		FourBigCircle.push_back(FourBigCircleTemp[norigin]);
		FourBigCircle.push_back(FourBigCircleTemp[npid1]);
		FourBigCircle.push_back(FourBigCircleTemp[npid2]);
		FourBigCircle.push_back(FourBigCircleTemp[npid3]);
	}
	// 4. 查找双轴
	CVect vxy45(FourBigCircle[0], FourBigCircle[2]);
	TEllipse xAxisHelpPoint, yAxisHelpPoint, yAxisPoint;
	vector<TEllipse> xAxisPointVect;
	vector<TEllipse> yAxisHelpPointVect;
	vector<TEllipse> xAxisHelpPointVect;
	vector<TEllipse> yAxisPointVect;

	{
		//////////////////////////////////////////////////////////////////////////
		{
			CVect vx(FourBigCircle[0], FourBigCircle[1]);
			vector<TEllipse> nxAxisPointVect;
			double sinangleh = 0.08;		// 3度
			std::vector<TEllipse>::iterator iter = SmallCircle.begin();


			while (iter != SmallCircle.end())
			{
				if (sinanglefab(vx, CVect(FourBigCircle[0], *iter)) < sinangleh)
				{
					xAxisPointVect.push_back(*iter);
					iter = SmallCircle.erase(iter);
					continue;
				}
				iter++;
			}

			iter = xAxisPointVect.begin();
			while (iter != xAxisPointVect.end())
			{
				if (acos(cosangle(vxy45, CVect(FourBigCircle[0], *iter))) > 1.57)
				{
					nxAxisPointVect.push_back(*iter);
					iter = xAxisPointVect.erase(iter);
					continue;
				}
				iter++;
			}
			//
			xAxisPointVect.push_back(FourBigCircle[1]);
			// order the axis point
			set<DistCompare> distset;
			std::set<DistCompare>::iterator iterd;
			iter = xAxisPointVect.begin();
			int ncount = 0;
			DistCompare disttemp;
			while (iter != xAxisPointVect.end())
			{
				disttemp.SetValue(FourBigCircle[0], -1, *iter, ncount);
				distset.insert(disttemp);
				iter++; ncount++;
			}
			iterd = distset.begin();
			ncount = 1;
			yAxisHelpPoint = (xAxisPointVect[iterd->nid2]);//(1，0)
			while (iterd != distset.end())
			{
				xAxisPointVect[iterd->nid2].nxid = ncount;
				xAxisPointVect[iterd->nid2].nyid = 0;
				iterd++; ncount++;
			}

			iter = nxAxisPointVect.begin();
			distset.clear();
			ncount = 0;
			while (iter != nxAxisPointVect.end())
			{
				disttemp.SetValue(FourBigCircle[0], -1, *iter, ncount);
				distset.insert(disttemp);
				iter++; ncount++;
			}
			iterd = distset.begin();
			ncount = 1;
			while (iterd != distset.end())
			{
				nxAxisPointVect[iterd->nid2].nxid = -ncount;
				nxAxisPointVect[iterd->nid2].nyid = 0;
				iterd++; ncount++;
			}
			iter = nxAxisPointVect.begin();
			while (iter != nxAxisPointVect.end())
			{
				xAxisPointVect.push_back(*iter);
				iter++;
			}
		}
		//////////////////////////////////////////////////////////////////////////
		{
			vector<TEllipse> nyAxisHelpPointVect;

			CVect vy(yAxisHelpPoint, FourBigCircle[2]);
			double sinangleh = 0.08;		// 3度
			std::vector<TEllipse>::iterator iter = SmallCircle.begin();
			while (iter != SmallCircle.end())
			{
				if (sinanglefab(vy, CVect(FourBigCircle[2], *iter)) < sinangleh)
				{
					yAxisHelpPointVect.push_back(*iter);
					iter = SmallCircle.erase(iter);
					continue;
				}
				iter++;
			}
			iter = yAxisHelpPointVect.begin();
			while (iter != yAxisHelpPointVect.end())
			{
				if (acos(cosangle(vxy45, CVect(FourBigCircle[2], *iter))) > 1.57)
				{
					nyAxisHelpPointVect.push_back(*iter);
					iter = yAxisHelpPointVect.erase(iter);
					continue;
				}
				iter++;
			}
			yAxisHelpPointVect.push_back(FourBigCircle[2]);
			set<DistCompare> distset;
			std::set<DistCompare>::iterator iterd;
			iter = yAxisHelpPointVect.begin();
			int ncount = 0;
			DistCompare disttemp;
			while (iter != yAxisHelpPointVect.end())
			{
				disttemp.SetValue(FourBigCircle[1], -1, *iter, ncount);
				distset.insert(disttemp);
				iter++; ncount++;
			}
			iterd = distset.begin();
			ncount = 1;
			while (iterd != distset.end())
			{
				yAxisHelpPointVect[iterd->nid2].nxid = 1;
				yAxisHelpPointVect[iterd->nid2].nyid = ncount;
				iterd++; ncount++;
			}
			iter = nyAxisHelpPointVect.begin();
			distset.clear();
			ncount = 0;
			while (iter != nyAxisHelpPointVect.end())
			{
				disttemp.SetValue(FourBigCircle[2], -1, *iter, ncount);
				distset.insert(disttemp);
				iter++; ncount++;
			}
			iterd = distset.begin();
			ncount = 1;
			xAxisHelpPoint = (nyAxisHelpPointVect[iterd->nid2]);
			while (iterd != distset.end())
			{
				nyAxisHelpPointVect[iterd->nid2].nxid = 1;
				nyAxisHelpPointVect[iterd->nid2].nyid = -ncount;
				iterd++; ncount++;
			}
			iter = nyAxisHelpPointVect.begin();
			while (iter != nyAxisHelpPointVect.end())
			{
				yAxisHelpPointVect.push_back(*iter);
				iter++;
			}
		}
		//////////////////////////////////////////////////////////////////////////
		{
			vector<TEllipse> nxAxisHelpPointVect;
			CVect vx(FourBigCircle[3], xAxisHelpPoint);
			double sinangleh = 0.08;		// 3度
			std::vector<TEllipse>::iterator iter = SmallCircle.begin();
			while (iter != SmallCircle.end())
			{
				if (sinanglefab(vx, CVect(FourBigCircle[3], *iter)) < sinangleh)
				{
					xAxisHelpPointVect.push_back(*iter);
					iter = SmallCircle.erase(iter);
					continue;
				}
				iter++;
			}
			iter = xAxisHelpPointVect.begin();
			while (iter != xAxisHelpPointVect.end())
			{
				if (acos(cosangle(vxy45, CVect(FourBigCircle[3], *iter))) > 1.57)
				{
					nxAxisHelpPointVect.push_back(*iter);
					iter = xAxisHelpPointVect.erase(iter);
					continue;
				}
				iter++;
			}
			xAxisHelpPointVect.push_back(xAxisHelpPoint);
			// order the axis point
			set<DistCompare> distset;
			std::set<DistCompare>::iterator iterd;
			iter = xAxisHelpPointVect.begin();
			int ncount = 0;
			DistCompare disttemp;
			while (iter != xAxisHelpPointVect.end())
			{
				disttemp.SetValue(FourBigCircle[3], -1, *iter, ncount);
				distset.insert(disttemp);
				iter++; ncount++;
			}
			iterd = distset.begin();
			ncount = 0;
			yAxisPoint = (xAxisHelpPointVect[iterd->nid2]);//(0,-1)
			while (iterd != distset.end())
			{
				xAxisHelpPointVect[iterd->nid2].nxid = ncount;
				xAxisHelpPointVect[iterd->nid2].nyid = -1;
				iterd++; ncount++;
			}
			//
			nxAxisHelpPointVect.push_back(FourBigCircle[3]);
			iter = nxAxisHelpPointVect.begin();
			distset.clear();
			ncount = 0;
			while (iter != nxAxisHelpPointVect.end())
			{
				disttemp.SetValue(FourBigCircle[3], -1, *iter, ncount);
				distset.insert(disttemp);
				iter++; ncount++;
			}
			iterd = distset.begin();
			ncount = 1;
			while (iterd != distset.end())
			{
				nxAxisHelpPointVect[iterd->nid2].nxid = -ncount;
				nxAxisHelpPointVect[iterd->nid2].nyid = -1;
				iterd++; ncount++;
			}
			iter = nxAxisHelpPointVect.begin();
			while (iter != nxAxisHelpPointVect.end())
			{
				xAxisHelpPointVect.push_back(*iter);
				iter++;
			}
		}
		//////////////////////////////////////////////////////////////////////////
		{
			vector<TEllipse> nyAxisPointVect;

			CVect vy(yAxisPoint, FourBigCircle[0]);
			double sinangleh = 0.08;		// 3度
			std::vector<TEllipse>::iterator iter = SmallCircle.begin();
			while (iter != SmallCircle.end())
			{
				if (sinanglefab(vy, CVect(FourBigCircle[0], *iter)) < sinangleh)
				{
					yAxisPointVect.push_back(*iter);
					iter = SmallCircle.erase(iter);
					continue;
				}
				iter++;
			}
			iter = yAxisPointVect.begin();
			while (iter != yAxisPointVect.end())
			{
				if (acos(cosangle(vxy45, CVect(FourBigCircle[0], *iter))) > 1.57)
				{
					nyAxisPointVect.push_back(*iter);
					iter = yAxisPointVect.erase(iter);
					continue;
				}
				iter++;
			}
			set<DistCompare> distset;
			std::set<DistCompare>::iterator iterd;
			iter = yAxisPointVect.begin();
			int ncount = 0;
			DistCompare disttemp;
			while (iter != yAxisPointVect.end())
			{
				disttemp.SetValue(FourBigCircle[0], -1, *iter, ncount);
				distset.insert(disttemp);
				iter++; ncount++;
			}
			iterd = distset.begin();
			ncount = 1;
			while (iterd != distset.end())
			{
				yAxisPointVect[iterd->nid2].nxid = 0;
				yAxisPointVect[iterd->nid2].nyid = ncount;
				iterd++; ncount++;
			}
			nyAxisPointVect.push_back(yAxisPoint);
			iter = nyAxisPointVect.begin();
			distset.clear();
			ncount = 0;
			while (iter != nyAxisPointVect.end())
			{
				disttemp.SetValue(FourBigCircle[0], -1, *iter, ncount);
				distset.insert(disttemp);
				iter++; ncount++;
			}
			iterd = distset.begin();
			ncount = 1;
			xAxisHelpPoint = (nyAxisPointVect[iterd->nid2]);
			while (iterd != distset.end())
			{
				nyAxisPointVect[iterd->nid2].nxid = 0;
				nyAxisPointVect[iterd->nid2].nyid = -ncount;
				iterd++; ncount++;
			}
			iter = nyAxisPointVect.begin();
			while (iter != nyAxisPointVect.end())
			{
				yAxisPointVect.push_back(*iter);
				iter++;
			}
		}
	}


	// 5.确定双轴的顺序
	vector<TEllipse> FinalCircle;
	{
		//////////////////////////////////////////////////////////////////////////
		set<SmallRuler> orderruler;
		SmallRuler temporder;
		std::vector<TEllipse>::iterator iter = xAxisPointVect.begin();
		while (iter != xAxisPointVect.end())
		{
			temporder.SetValue(&(*iter), iter->nxid);
			orderruler.insert(temporder);
			iter++;
		}
		vector<TEllipse> tempbc;
		std::set<SmallRuler>::iterator iterruler = orderruler.begin();
		while (iterruler != orderruler.end())
		{
			tempbc.push_back(*(iterruler->bc));
			iterruler++;
		}
		xAxisPointVect.clear();
		iter = tempbc.begin();
		while (iter != tempbc.end())
		{
			if (iter->nxid == 0 || iter->nxid == 1)
				FinalCircle.push_back(*iter);
			else
				xAxisPointVect.push_back(*iter);
			iter++;
		}
		//////////////////////////////////////////////////////////////////////////
		tempbc.clear();
		orderruler.clear();
		iter = xAxisHelpPointVect.begin();
		while (iter != xAxisHelpPointVect.end())
		{
			temporder.SetValue(&(*iter), iter->nxid);
			orderruler.insert(temporder);
			iter++;
		}
		iterruler = orderruler.begin();
		while (iterruler != orderruler.end())
		{
			tempbc.push_back(*(iterruler->bc));
			iterruler++;
		}
		xAxisHelpPointVect.clear();
		iter = tempbc.begin();
		while (iter != tempbc.end())
		{
			if (iter->nxid == 0 || iter->nxid == 1)
				FinalCircle.push_back(*iter);
			else
				xAxisHelpPointVect.push_back(*iter);
			iter++;
		}
		//////////////////////////////////////////////////////////////////////////
		tempbc.clear();
		orderruler.clear();
		iter = yAxisPointVect.begin();
		while (iter != yAxisPointVect.end())
		{
			temporder.SetValue(&(*iter), iter->nyid);
			orderruler.insert(temporder);
			iter++;
		}
		iterruler = orderruler.begin();
		while (iterruler != orderruler.end())
		{
			tempbc.push_back(*(iterruler->bc));
			iterruler++;
		}
		yAxisPointVect.clear();
		iter = tempbc.begin();
		while (iter != tempbc.end())
		{
			if (iter->nyid != 0 && iter->nyid != -1)
				yAxisPointVect.push_back(*iter);
			iter++;
		}
		//////////////////////////////////////////////////////////////////////////
		tempbc.clear();
		orderruler.clear();
		iter = yAxisHelpPointVect.begin();
		while (iter != yAxisHelpPointVect.end())
		{
			temporder.SetValue(&(*iter), iter->nyid);
			orderruler.insert(temporder);
			iter++;
		}
		iterruler = orderruler.begin();
		while (iterruler != orderruler.end())
		{
			tempbc.push_back(*(iterruler->bc));
			iterruler++;
		}
		yAxisHelpPointVect.clear();
		iter = tempbc.begin();
		while (iter != tempbc.end())
		{
			if (iter->nyid != 0 && iter->nyid != -1)
				yAxisHelpPointVect.push_back(*iter);
			iter++;
		}
		//////////////////////////////////////////////////////////////////////////
	}


	// 5.确定其他小圆的坐标
	{
		std::vector<TEllipse>::iterator iter = xAxisPointVect.begin();
		std::vector<TEllipse>::iterator iter2;
		std::vector<TEllipse>::iterator iters = SmallCircle.begin();
		double sinangleh = 0.05;		// 3度
										//////////////////////////////////////////////////////////////////////////
		while (iter != xAxisPointVect.end())
		{
			iter2 = xAxisHelpPointVect.begin();
			bool bFind = false;
			while (iter2 != xAxisHelpPointVect.end())
			{
				if (iter->nxid == iter2->nxid)
				{
					bFind = true;
					break;
				}
				iter2++;
			}
			if (bFind)
			{
				iters = SmallCircle.begin();
				while (iters != SmallCircle.end())
				{
					if (sinanglefab(CVect(*iter, *iter2), CVect(*iter, *iters)) < sinangleh)
						iters->nxid = iter->nxid;
					iters++;
				}
			}
			iter++;
		}
		//////////////////////////////////////////////////////////////////////////
		iter = yAxisPointVect.begin();
		while (iter != yAxisPointVect.end())
		{
			iter2 = yAxisHelpPointVect.begin();
			bool bFind = false;
			while (iter2 != yAxisHelpPointVect.end())
			{
				if (iter->nyid == iter2->nyid)
				{
					bFind = true;
					break;
				}
				iter2++;
			}
			if (bFind)
			{
				iters = SmallCircle.begin();
				while (iters != SmallCircle.end())
				{
					if (sinanglefab(CVect(*iter, *iter2), CVect(*iter, *iters)) < sinangleh)
						iters->nyid = iter->nyid;
					iters++;
				}
			}
			iter++;
		}
	}

	// 6.总结最后圆
	{
		std::vector<TEllipse>::iterator iter = FourBigCircle.begin();
		FinalCircle.push_back(FourBigCircle[0]);
		iter = xAxisPointVect.begin();
		while (iter != xAxisPointVect.end())
		{
			FinalCircle.push_back(*iter);
			iter++;
		}
		iter = xAxisHelpPointVect.begin();
		while (iter != xAxisHelpPointVect.end())
		{
			FinalCircle.push_back(*iter);
			iter++;
		}
		iter = yAxisPointVect.begin();
		while (iter != yAxisPointVect.end())
		{
			FinalCircle.push_back(*iter);
			iter++;
		}
		iter = yAxisHelpPointVect.begin();
		while (iter != yAxisHelpPointVect.end())
		{
			FinalCircle.push_back(*iter);
			iter++;
		}
		iter = SmallCircle.begin();
		while (iter != SmallCircle.end())
		{
			FinalCircle.push_back(*iter);
			iter++;
		}
	}

	// 7. 修正多余的圆
	{
		CVect vx(FourBigCircle[0], FourBigCircle[1]);
		CVect vy(FourBigCircle[0], yAxisPoint);
		double sinangleh = 0.05;		// 3度
		std::vector<TEllipse>::iterator iter = FinalCircle.begin();
		while (iter != FinalCircle.end())
		{
			if (iter->nxid == 0)
			{
				if (sinanglefab(vy, CVect(FourBigCircle[0], *iter)) > sinangleh)
				{
					iter = FinalCircle.erase(iter);
					continue;
				}
			}
			if (iter->nyid == 0)
			{
				if (sinanglefab(vx, CVect(FourBigCircle[0], *iter)) > sinangleh)
				{
					iter = FinalCircle.erase(iter);
					continue;
				}
			}
			iter++;
		}
	}

	// out put the data
	std::vector<TEllipse>::iterator iter = ellipses.begin();
	iter = FinalCircle.begin();
	while (iter != FinalCircle.end())
	{
		iter->m_dXw = (iter->nxid)*DX;
		iter->m_dYw = (iter->nyid)*DY;
		ellpco.Push(*iter);
		iter++;
	}
	iter = FinalCircle.begin();
	if (!m_vecFeaturePts.empty())
	{
		m_vecFeaturePts.clear();
		vector<C4DPointD>().swap(m_vecFeaturePts);
	}
	int circle_num = FinalCircle.size();
	C4DPointD tempPts;
	tempPts.xf = 0;
	tempPts.yf = 0;
	tempPts.xw = 0;
	tempPts.yw = 0;
	tempPts.zw = 0;
	for (int i = 0; i<circle_num; i++)
	{
		m_vecFeaturePts.push_back(tempPts);
	}
	double xf, yf, xw, yw, zw;
	vector<C4DPointD>::iterator iter1 = m_vecFeaturePts.begin();
	while (iter != FinalCircle.end())
	{
		iter1->xf = iter->m_dCenterpx;
		iter1->yf = iter->m_dCenterpy;
		iter1->xw = iter->nxid * DX;
		iter1->yw = iter->nyid * DY;
		iter1->zw = 0;
		iter++;
		iter1++;
	}

	// draw the element
	iter1 = m_vecFeaturePts.begin();
	String str;
	String str2;
	cvtColor(img, colorImage, CV_GRAY2BGR);
	while (iter1 != m_vecFeaturePts.end())
	{
		xf = (int)iter1->xf;
		yf = (int)iter1->yf;
		xw = iter1->xw;
		yw = iter1->yw;
		cv::line(colorImage, Point(xf - 5, yf), Point(xf + 5, yf), cv::Scalar(0, 0, 255));
		cv::line(colorImage, Point(xf, yf - 5), Point(xf, yf + 5), cv::Scalar(0, 0, 255));

		str = to_string(xw);
		int pos = str.find(".");
		str = str.substr(0, pos + 2);
		cv::putText(colorImage, str, Point(xf + 4, yf + 4), cv::FONT_HERSHEY_PLAIN, 0.2, cv::Scalar(0, 255, 255));
		str2 = to_string(yw);
		pos = str2.find(".");
		str2 = str2.substr(0, pos + 2);
		cv::putText(colorImage, str2, Point(xf + 4, yf + 12), cv::FONT_HERSHEY_PLAIN, 0.2, cv::Scalar(0, 255, 255));
		iter1++;
		imshow("sortPointImage", colorImage);
		waitKey();
	}
}

void sortcoordinate(vector<C4DPointD> temvecFeaturePtsL, vector<Point2f>& imagePoints, vector<Point3f>& objectPoints)
{

		Point3f obpoints;
		float xw = 0;
		float yw = 0;
		int num = 0;
		while (num != NFEATUREPOINTS)
		{
			for (int j = 0; j < NFEATUREPOINTS; j++)
			{
				int subscript = j;
				int errorxw = fabs(xw - temvecFeaturePtsL[subscript].xw);
				int erroryw = fabs(yw - temvecFeaturePtsL[subscript].yw);
				if ((errorxw < 1e-6) && (erroryw < 1e-6))
				{
					float x = temvecFeaturePtsL[subscript].xf;
					float y = temvecFeaturePtsL[subscript].yf;
					imagePoints.push_back(Point2f(x, y));
					objectPoints.push_back(Point3f(xw, yw, 0));
					//表示又找到一个点
					num++;
					break; //跳出本次循环，开始寻找下一个点
				}
			}

			float edge = (sqrt(NFEATUREPOINTS) - 1)*DRONE_X;
			if (yw == edge)
			{
				yw = 0;
				xw = xw + DRONE_X;
			}
			else
			{
				yw = yw + DRONE_Y;
			}
		}
}


int main()
{
	Mat imgl = imread("checkImages\\test_l3.bmp", IMREAD_GRAYSCALE);
	Mat imgr = imread("checkImages\\test_r3.bmp", IMREAD_GRAYSCALE);
	Mat model = imread("checkImages\\model.bmp", IMREAD_GRAYSCALE);
	Mat rectifyl, rectifyr, Q;
	rectify(imgl, imgr, rectifyl, rectifyr, Q);

	//获取靶标区域
	Mat mask1, mask2;
	mask1 = getTarget(imgl, model);
	mask2 = getTarget(imgr, model);
	bitwise_and(imgl, mask1, imgl);
	bitwise_and(imgr, mask2, imgr);
	

	//二值化，寻找轮廓,拟合椭圆
	Mat threImgL, threImgR;
	adaptiveThreshold(imgl, threImgL, 255, ADAPTIVE_THRESH_MEAN_C,THRESH_BINARY,25,0);
	adaptiveThreshold(imgr, threImgR, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 25,0);
	TEllipseCollection ellpl, ellpr;
	
	Find_counter2(threImgL, ellpl);
	Find_counter2(threImgR, ellpr);

	vector<C4DPointD> m_vecFeaturePtsl;
	vector<C4DPointD> m_vecFeaturePtsr;
	Find_Topology(ellpl, DRONE_X, DRONE_Y, m_vecFeaturePtsl, imgl );
	Find_Topology(ellpr, DRONE_X, DRONE_Y, m_vecFeaturePtsr,imgr);
	
	
	//排序
	double offsetX = 0;
	double offsetY = 0;
	for (auto b = m_vecFeaturePtsl.begin(); b != m_vecFeaturePtsl.end(); ++b)
	{
		if (b->xw < offsetX) { offsetX = b->xw; }
		if (b->yw < offsetY) { offsetY = b->yw; }
	}
	for (auto b = m_vecFeaturePtsr.begin(); b != m_vecFeaturePtsr.end(); ++b)
	{
		b->xw = b->xw - offsetX;
		b->yw = -b->yw - offsetY;
	}
	for (auto b = m_vecFeaturePtsl.begin(); b != m_vecFeaturePtsl.end(); ++b)
	{
		b->xw = b->xw - offsetX;
		b->yw = -b->yw - offsetY;
	}

	vector<Point2f> imagePoints[2];
	vector<Point3f>objectPointsl;
	vector<Point3f>objectPointsr;
	//坐标点排序（两幅图像特征点之间可能出现不完全对应）
	sortcoordinate(m_vecFeaturePtsl, imagePoints[0], objectPointsl);
	sortcoordinate(m_vecFeaturePtsr, imagePoints[1], objectPointsr);

	fstream fout("imagel_points.txt", ios::out);
	for (int i = 0; i < imagePoints[0].size(); i++)
	{
		fout << imagePoints[0][i].x << " " << imagePoints[0][i].y << " " << objectPointsl[i].x << " " << objectPointsl[i].y << " " << objectPointsl[i].z << endl;
	}
	fout.close();


	fout.open("imager_points.txt", ios::out);
	for (int i = 0; i < imagePoints[1].size(); i++)
	{
		fout << imagePoints[1][i].x << " " << imagePoints[1][i].y << " " << objectPointsr[i].x << " " << objectPointsr[i].y << " " << objectPointsr[i].z << endl;
	}
	fout.close();

	//计算投影矩阵，重建三维点
	Mat Mk1, Mk2;
	FileStorage fMk("ProjectMatrix_LR.xml", FileStorage::READ);
	fMk["Mk1"] >> Mk1;
	fMk["Mk2"] >> Mk2;
	fMk.release();

	double M1[3][4];
	double M2[3][4];//转换参数
	convertParam(Mk1, Mk2, M1, M2);

	double *u1 = new double[81];
	double *v1 = new double[81];
	double *u2 = new double[81];
	double *v2 = new double[81];
	vector<Point3f> worldPoint;
	for (int i = 0; i < 81; i++)
	{
		u1[i] = imagePoints[0][i].x;
		v1[i] = imagePoints[0][i].y;
		u2[i] = imagePoints[1][i].x;
		v2[i] = imagePoints[1][i].y;
	}

	for (int i = 0; i < 81; i++)
	{
		Mat A = (Mat_<double>(4, 3) << u1[i] * M1[2][0] - M1[0][0], u1[i] * M1[2][1] - M1[0][1], u1[i] * M1[2][2] - M1[0][2],
			v1[i] * M1[2][0] - M1[1][0], v1[i] * M1[2][1] - M1[1][1], v1[i] * M1[2][2] - M1[1][2],
			u2[i] * M2[2][0] - M2[0][0], u2[i] * M2[2][1] - M2[0][1], u2[i] * M2[2][2] - M2[0][2],
			v2[i] * M2[2][0] - M2[1][0], v2[i] * M2[2][1] - M2[1][1], v2[i] * M2[2][2] - M2[1][2]);

		Mat M = (Mat_<double>(4, 1) << M1[0][3] - u1[i] * M1[2][3], M1[1][3] - v1[i] * M1[2][3], M2[0][3] - u2[i] * M2[2][3], M2[1][3] - v2[i] * M2[2][3]);

		Mat X = Mat::zeros(3, 1, CV_64FC1);

		solve(A, M, X, DECOMP_SVD);  //SVD分解求解，如果报错，改为DECOMP_NORMAL

		Point3f point;
		point.x = *X.ptr<double>(0, 0);
		point.y = *X.ptr<double>(1, 0);
		point.z = *X.ptr<double>(2, 0);
		worldPoint.push_back(point);
	}
	//数据处理
	cout << "重建81个圆心的坐标：" << endl;
	for (auto p : worldPoint)
	{
		cout << p << endl;
	}

	vector<double> vDist;
	for (int i = 1; i < worldPoint.size(); i++)
	{
		if (i % 9 != 0)
		{
			double dist = sqrt(pow(worldPoint[i].x - worldPoint[i - 1].x, 2) + pow(worldPoint[i].y - worldPoint[i - 1].y, 2) + pow(worldPoint[i].z - worldPoint[i - 1].z, 2));
			vDist.push_back(dist);
			cout << i - 1 << "-" << i << "距离：" << dist;
			cout << " \t 与真值的偏差=" << dist - 12.0 << endl;
		}
		
	}
	Mat D(vDist);
	Mat mean, stddev;
	meanStdDev(D, mean, stddev);
	cout << "测量数据的均值、标准差：" << endl;
	cout << "mean=" << mean << endl;
	cout << "stddev=" << stddev << endl;
	cout << "最大误差：" << *(std::max_element(std::begin(vDist), std::end(vDist)))-12.0 << endl;
	cout << "与真值相比的标准差：" << endl;
	D = D - 12.0;
	Mat P= D.mul(D);
	Scalar s = sum(P);
	double d = s[0];
	d /= 72.0;
	cout <<sqrt(d)<< endl;
	system("pause");


	return 0;
}