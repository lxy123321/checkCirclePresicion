#pragma once
#include<iostream>
#include"TEllipse.h"

#define PI 3.141592654		        //圆周率π
#define NFEATUREPOINTS   81         //每张图片的特征点数量
#define IMAGE_WIDTH 1280	        //图像宽
#define IMAGE_HEIGHT 1024	        //图像高度
#define DRONE_X		12		        //靶标x方向相邻圆心距离
#define DRONE_Y		12		        //靶标y方向相邻圆心距离

#define IMAGE_NAME_LEN          64
#define CALIIMG_PATH        "CaliImage"      //标定图像路径
#define TESTIMG_PATH        "TestImage"      //测量图像路径
#define CALIRES_PATH        "CaliResult"     //标定结果路径
#define CHECKPRO_PATH       "CheckProcess"   //中间过程的检查


struct C4DPointD
{
	double xf, yf, xw, yw, zw;

	C4DPointD(double xfv = 0.0, double yfv = 0.0, 
		double xwv = 0.0, double ywv = 0.0, double zwv = 0.0) : xf(xfv), yf(yfv), xw(xwv), yw(ywv), zw(zwv) {};

	C4DPointD(const C4DPointD &rhs)
	{
		memcpy(this, &rhs, sizeof(C4DPointD));
	}

	C4DPointD& C4DPointD::operator = (const C4DPointD& rhs)
	{
		if (this == &rhs)
			return *this;

		memcpy(this, &rhs, sizeof(C4DPointD));
		return *this;
	}
};

struct CBinoPointD
{
	double ul, vl, ur, vr, xw, yw, zw;  //ul 左相机的像素坐标 ur 右相机的像素坐标 xw yx zw 对应的三维坐标
	int red, green, blue;
	CBinoPointD(double xfv = 0.0, double yfv = 0.0, 
		double xwv = 0.0, double ywv = 0.0, double zwv = 0.0) : ul(xfv), vl(yfv), ur(xfv), vr(yfv), xw(xwv), yw(ywv), zw(zwv), red(0), green(0), blue(0) {};

	bool operator<(const CBinoPointD &cor2) const {

		if (ur<cor2.ur)
			return true;
		if (ur == cor2.ur)
		{
			if (vr< cor2.vr)
				return true;
		}
		return false;
	}
};

class OrderRuler
{
public:
	int nid;
	int ncount;
	double area;

public:
	OrderRuler() :nid(0), ncount(0), area(0.0) {};
	OrderRuler(const OrderRuler& theOrderRuler) { memcpy(this, &theOrderRuler, sizeof(OrderRuler)); }

	OrderRuler& operator=(const OrderRuler& theOrderRuler)
	{
		memcpy(this, &theOrderRuler, sizeof(OrderRuler));
		return *this;
	}

	bool operator<(const OrderRuler& v) const
	{
		if (area == v.area && ncount == v.ncount)
			return nid < v.nid;
		else if (area == v.area)
			return ncount < v.ncount;
		return area < v.area;
	}

	bool operator==(const OrderRuler& v) const
	{
		return ncount == v.ncount && area == v.area && nid == v.nid;
	}
};

struct CVect
{
	double x, y;

	CVect(double x1, double y1, double x2, double y2)
	{
		x = x2 - x1;
		y = y2 - y1;
	}

	CVect(TEllipse& p1, TEllipse& p2)
	{
		x = p2.m_dCenterpx - p1.m_dCenterpx;
		y = p2.m_dCenterpy - p1.m_dCenterpy;
	}
};

inline double cosangle(const CVect & v1, const CVect & v2)
{
	return (v1.x*v2.x + v1.y*v2.y) / (sqrt(v1.x * v1.x + v1.y * v1.y) * sqrt(v2.x* v2.x + v2.y * v2.y));
}

inline double sinanglefab(const CVect & v1, const CVect & v2)
{
	double cosanglev = (v1.x*v2.x + v1.y*v2.y) / (sqrt(v1.x * v1.x + v1.y * v1.y) * sqrt(v2.x* v2.x + v2.y * v2.y));
	return sqrt(1.0 - cosanglev * cosanglev);
}

class DistCompare
{

public:
	int nid1, nid2;
	double dist;

public:
	DistCompare() :nid1(0), nid2(0), dist(0.0) {};
	DistCompare(const DistCompare& theDistCompare) { memcpy(this, &theDistCompare, sizeof(DistCompare)); }

	DistCompare& operator=(const DistCompare& theOrderRuler)
	{
		memcpy(this, &theOrderRuler, sizeof(DistCompare));
		return *this;
	}

	bool operator<(const DistCompare& v) const
	{
		return dist < v.dist;
	}

	bool operator>(const DistCompare& v) const
	{
		return dist > v.dist;
	}

	bool operator==(const DistCompare& v) const
	{
		return dist == v.dist;
	}

	void SetValue(TEllipse& p1, int n1, TEllipse& p2, int n2)
	{
		dist = (p1.m_dCenterpx - p2.m_dCenterpx) * (p1.m_dCenterpx - p2.m_dCenterpx) + (p1.m_dCenterpy - p2.m_dCenterpy) * (p1.m_dCenterpy - p2.m_dCenterpy);
		nid1 = n1; nid2 = n2;
	}

	void SetValue(double x, double y, int npos, TEllipse& p1, TEllipse& p2)
	{
		dist = (x - p1.m_dCenterpx) * (x - p1.m_dCenterpx) + (y - p1.m_dCenterpy) * (y - p1.m_dCenterpy) +
			(x - p2.m_dCenterpx) * (x - p2.m_dCenterpx) + (y - p2.m_dCenterpy) * (y - p2.m_dCenterpy);
		nid1 = npos;
	}

	void SetValue(TEllipse& p1, double x, double y, int npos)
	{
		dist = (x - p1.m_dCenterpx) * (x - p1.m_dCenterpx) + (y - p1.m_dCenterpy) * (y - p1.m_dCenterpy);
		nid1 = npos;
	}

	void SetValue(CVect& v1, int n1, CVect& v2, int n2)
	{
		dist = sinanglefab(v1, v2);
		nid1 = n1; nid2 = n2;
	}
};

class SmallRuler
{
public:
	int ncount;
	TEllipse* bc;

public:
	SmallRuler() :ncount(0), bc(NULL) {};
	SmallRuler(const SmallRuler& theOrderRuler) { memcpy(this, &theOrderRuler, sizeof(SmallRuler)); }

	SmallRuler& operator=(const SmallRuler& theOrderRuler)
	{
		memcpy(this, &theOrderRuler, sizeof(SmallRuler));
		return *this;
	}

	bool operator<(const SmallRuler& v) const
	{
		return ncount < v.ncount;
	}

	bool operator==(const OrderRuler& v) const
	{
		return ncount == v.ncount;
	}

	void SetValue(TEllipse* p1, int n1)
	{
		ncount = n1; bc = p1;
	}
};

