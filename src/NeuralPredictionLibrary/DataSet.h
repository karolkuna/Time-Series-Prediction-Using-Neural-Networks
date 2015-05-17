//
//  DataSet.h
//  NeuralPredictionLibrary
//
//  Copyright 2015 Karol Kuna
//
//  Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
//

#ifndef __prediction__DataSet__
#define __prediction__DataSet__

#include <stdio.h>
#include <string>
#include <iostream>
#include <sstream>
#include <fstream>
#include <stdexcept>
#include "MemoryBlock.h"

using std::string;
using std::ifstream;
using std::logic_error;

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
