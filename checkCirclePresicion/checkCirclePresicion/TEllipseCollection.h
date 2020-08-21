#pragma once
#include<vector>
#include"TEllipse.h"

using namespace std;

class TEllipseCollection
{
public:
	TEllipseCollection();
	~TEllipseCollection();

public:
	vector<TEllipse> ellipses;
	double m_dz;
	/*TEllipseCollection()
	{
		ellipses.clear();
		m_dz = 0.0;
	}*/
	void Clear();
	int Size();
	void Push(const TEllipse& rhs);
	void Append(const TEllipseCollection& rhs);
	void Pop();
};

