#include "TEllipse.h"


TEllipse::TEllipse()
{
	Clear();
}


TEllipse::~TEllipse()
{
}


void TEllipse::Clear()
{
	m_dCenterpx = m_dCenterpy = 0.0;
	m_dXw = m_dYw = m_dZw = 0.0;
	m_dArea = 0.0;
	for (int i = 0; i < 7; i++) m_dHuM[i] = 0.0;
	m_dSimilarity = 0.0;

}

bool TEllipse::CaculateSimilarity(const TEllipse& ellipse)
{
	for (int i = 0; i<7; i++)
	{
		if (m_dHuM[i] == 0 || ellipse.m_dHuM[i] == 0)
		{
			m_dSimilarity = 1.0;
			break;
		}

		m_dSimilarity += 1 / m_dHuM[i] - 1 / ellipse.m_dHuM[i];
	}
	return true;
}