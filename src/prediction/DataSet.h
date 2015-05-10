//
//  DataSet.h
//  prediction
//
//  Created by Karol Kuna on 02/05/15.
//  Copyright (c) 2015 Karol Kuna. All rights reserved.
//

#ifndef __prediction__DataSet__
#define __prediction__DataSet__

#include <stdio.h>
#include <string>
#include <iostream>
#include <sstream>
#include <fstream>
#include "MemoryBlock.h"

using std::string;
using std::ifstream;

class DataSet {
	ifstream m_file;
	bool m_isInput;
	
public:
	MemoryBlock input;
	MemoryBlock output;
	
	DataSet(const string& fileName, bool isInput);
	~DataSet();
	
	bool Read();
};

#endif /* defined(__prediction__DataSet__) */
