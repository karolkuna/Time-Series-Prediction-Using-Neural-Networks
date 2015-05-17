//
//  DataSet.cpp
//  NeuralPredictionLibrary
//
//  Copyright 2015 Karol Kuna
//
//  Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
//

#include "DataSet.h"

DataSet::DataSet(const string& fileName, bool isInput) {
	m_file.open(fileName);
	if (!m_file.is_open()) {
		throw std::logic_error("Cannot open file!");
	}
	
	if (isInput) {
		string line;
		getline(m_file, line);
		int inputSize = 1;
		for (int i = 0; i < line.length(); i++) {
			if (line.at(i) == ' ')
				inputSize++;
		}
		input = MemoryBlock(inputSize);
		input.Fill(0);
	}

	string line;
	getline(m_file, line);
	int outputSize = 1;
	for (int i = 0; i < line.length(); i++) {
		if (line.at(i) == ' ')
			outputSize++;
	}
	output = MemoryBlock(outputSize);
	output.Fill(0);
	
	m_file.clear();
	m_file.seekg(0, std::ios::beg);
}

DataSet::~DataSet() {
	m_file.close();
}

bool DataSet::Read() {
	if (!m_file) {
		return false;
	}
	
	for (int i = 0; i < input.size; i++) {
		m_file >> input.data[i];
	}
	for (int i = 0; i < output.size; i++) {
		m_file >> output.data[i];
	}
	
	return true;
}
