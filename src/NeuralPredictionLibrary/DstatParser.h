//
//  DstatParser.cpp
//  NeuralPredictionLibrary
//
//  Copyright 2015 Karol Kuna
//
//  Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
//

#ifndef __prediction__DstatParser__
#define __prediction__DstatParser__

#include <iostream>
#include <string>
#include <fstream>
#include <stdexcept>
#include "MemoryBlock.h"
#include "ActivationFunctions.h"

using std::cin;
using std::cout;
using std::endl;
using std::string;
using std::ifstream;
using std::vector;
using std::logic_error;

class DstatParser {
	bool m_isFile;
	ifstream m_file;
	vector<string> m_labels;
	
	void PrepareOutput(istream& inputFile);
	float GetValue(const string& word);
	void StoreValue(float value, int pos);
	
public:
	int recvIndex;
	string line;
	MemoryBlock output;
	
	DstatParser();
	DstatParser(const string& fileName);
	~DstatParser();
	
	bool Parse();
	void Preprocess();
	
};

#endif
