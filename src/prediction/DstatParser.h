//
//  dstatParser.cpp
//  prediction
//
//  Created by Karol Kuna on 02/05/15.
//  Copyright (c) 2015 Karol Kuna. All rights reserved.
//

#ifndef __prediction__DstatParser__
#define __prediction__DstatParser__

#include <iostream>
#include <string>
#include <fstream>
#include "MemoryBlock.h"
#include "ActivationFunctions.h"

using std::cin;
using std::cout;
using std::endl;
using std::string;
using std::ifstream;
using std::vector;

class DstatParser {
	bool m_isFile;
	ifstream m_file;
	vector<string> m_labels;
	
	float GetValue(const string& word);
	void StoreValue(float value, int pos);
	
public:
	int recvIndex = -1;
	MemoryBlock output;
	
	DstatParser();
	DstatParser(const string& fileName);
	~DstatParser();
	
	bool Parse();
	void Preprocess();
	
};

#endif
