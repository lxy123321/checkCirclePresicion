#pragma once
#include <vector>
#include <opencv2\opencv.hpp>

class TEllipse
{
public:
	TEllipse();
	~TEllipse();
public:
	int nid;
	//图像处理后加上的
	double m_dCenterpx, m_dCenterpy;//像面坐标***
	double m_dArea;
	double m_dHuM[7];//HU矩组
	double m_dSimilarity;//相似系数
						 /*	double aa,bb;*/
	double m_dXw, m_dYw, m_dZw; //世界坐标****	
	int nxid, nyid;
	CvSeq* contour;
	TEllipse* nTel[4];
	/*TEllipse()
	{
		Clear();
	}*/
	TEllipse(const TEllipse& tell)
	{
		memcpy(this, &tell, sizeof(TEllipse));
	}

	TEllipse(double m[7],
		double cpx = 0.0, double cpy = 0.0,
		double area = 0.0, double similarity = 0.0,
		double xw = 0.0, double yw = 0.0, double zw = 0.0,
		int xid = 0, int yid = 0, int id = 0
	) :m_dCenterpx(cpx), m_dCenterpy(cpy), m_dArea(area), m_dSimilarity(similarity),
		m_dXw(xw), m_dYw(yw), m_dZw(zw), nxid(xid), nyid(yid), nid(id)/*,nTel(NULL)*/
	{
		if (m != NULL)
		{
			size_t size = sizeof(double) * 7;
			memcpy(m_dHuM, m, size);
		}

		for (int i = 0; i<4; i++)
		{
			nTel[i] = NULL;
		}
	}

	TEllipse& operator=(const TEllipse& rhs)
	{
		memcpy(this, &rhs, sizeof(TEllipse));
		return *this;
	};

	void Clear();
//	BOOL CaculateM();
	bool CaculateSimilarity(const TEllipse& ellipseMax);
};

