#include "TEllipseCollection.h"


TEllipseCollection::TEllipseCollection()
{
	ellipses.clear();
	m_dz = 0.0;
}


TEllipseCollection::~TEllipseCollection()

{
}

void TEllipseCollection::Clear()
{
	this->ellipses.clear();
	m_dz = 0.0;
}

void TEllipseCollection::Push(const TEllipse& rhs)
{
	this->ellipses.push_back(rhs);
}

void TEllipseCollection::Append(const TEllipseCollection& rhs)
{
	this->ellipses.insert(this->ellipses.end(), rhs.ellipses.begin(), rhs.ellipses.end());
}

void TEllipseCollection::Pop()
{
	this->ellipses.pop_back();
}

int TEllipseCollection::Size()
{
	return int(this->ellipses.size());
}
