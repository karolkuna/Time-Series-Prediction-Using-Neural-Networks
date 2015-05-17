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

#include "DstatParser.h"

DstatParser::DstatParser() {
	m_isFile = false;
	PrepareOutput(cin);
}

DstatParser::DstatParser(const string& fileName) {
	m_isFile = true;
	m_file.open(fileName);
	
	if (!m_file.is_open()) {
		throw std::logic_error("Cannot open file!");
	}
	
	PrepareOutput(m_file);
}

DstatParser::~DstatParser() {
	if (m_isFile) {
		m_file.close();
	}
}

void DstatParser::PrepareOutput(istream& inputFile) {
	getline(inputFile, line);
	
	if (line.compare("You did not select any stats, using -cdngy by default.") == 0) {
		//one extra line
		getline(inputFile, line);
	}
	getline(inputFile, line);
	
	recvIndex = 0;
	int outputSize = 0;
	
	std::size_t prev = 0, pos;
	while ((pos = line.find_first_of(" ';|", prev)) != string::npos) {
		if (pos > prev) {
			m_labels.push_back(line.substr(prev, pos-prev));
			if (m_labels.back().compare("recv") == 0) {
				recvIndex = outputSize;
			}
			outputSize++;
		}
		prev = pos + 1;
	}
	if (prev < line.length()) {
		m_labels.push_back(line.substr(prev, std::string::npos));
		outputSize++;
	}
	
	output = MemoryBlock(outputSize);
	output.Fill(0);
}

float DstatParser::GetValue(const string& word) {
	char lastChar = word.back();
	float value;
	
	if (lastChar == 'k' || lastChar == 'M' || lastChar == 'G' || lastChar == 'B') {
		value = stof(word.substr(0, word.size()-1));
		
		if (lastChar == 'k') {
			value *= 1024;
		} else if (lastChar == 'M') {
			value *= 1048576;
		} else if (lastChar == 'G') {
			value *= 1098907648;
		}
	} else {
		value = stof(word);
	}
	
	return value;
}

void DstatParser::StoreValue(float value, int pos) {
	output.data[pos] = value;
}

bool DstatParser::Parse() {
	int index = 0;
	
	if (getline(m_isFile ? m_file : cin, line)) {
		std::size_t prev = 0, pos;
		while ((pos = line.find_first_of(" ';|", prev)) != std::string::npos) {
			if (pos > prev) {
				StoreValue(GetValue(line.substr(prev, pos-prev)), index);
				index++;
			}
			prev = pos + 1;
		}
		if (prev < line.length())
			StoreValue(GetValue(line.substr(prev, std::string::npos)), index);
		return true;
	} else {
		return false;
	}
}

void DstatParser::Preprocess() {
	for (int i = 0; i < output.size; i++) {
		if (m_labels[i].compare("usr") == 0) {
			output.data[i] = output.data[i] * 0.01f;
		} else if (m_labels[i].compare("sys") == 0) {
			output.data[i] *= 0.01f;
		} else if (m_labels[i].compare("idl") == 0) {
			output.data[i] *= 0.01f;
		} else if (m_labels[i].compare("wai") == 0) {
			output.data[i] *= 0.01f;
		} else if (m_labels[i].compare("hiq") == 0) {
			output.data[i] *= 0.1f;
		} else if (m_labels[i].compare("siq") == 0) {
			output.data[i] *= 0.1f;
		} else if (m_labels[i].compare("read") == 0) {
			output.data[i] *= 0.00000001;
		} else if (m_labels[i].compare("writ") == 0) {
			output.data[i] *= 0.00000001;
		} else if (m_labels[i].compare("used") == 0) {
			output.data[i] *= 0.00000000001f;
		} else if (m_labels[i].compare("buff") == 0) {
			output.data[i] *= 0.0000000001f;
		} else if (m_labels[i].compare("cach") == 0) {
			output.data[i] *= 0.00000000001f;
		} else if (m_labels[i].compare("free") == 0) {
			output.data[i] *= 0.000000001f;
		} else if (m_labels[i].compare("recv") == 0) {
			output.data[i] *= 0.00000005;
		} else if (m_labels[i].compare("send") == 0) {
			output.data[i] *= 0.00000005;
		} else if (m_labels[i].compare("tot") == 0) {
			output.data[i] *= 0.001f;
		} else if (m_labels[i].compare("tcp") == 0) {
			output.data[i] *= 0.001f;
		} else if (m_labels[i].compare("udp") == 0) {
			output.data[i] *= 0.001f;
		} else if (m_labels[i].compare("raw") == 0) {
			output.data[i] *= 0.1f;
		} else if (m_labels[i].compare("frg") == 0) {
			output.data[i] *= 0.1f;
		} else if (m_labels[i].compare("lis") == 0) {
			output.data[i] *= 0.001f;
		} else if (m_labels[i].compare("act") == 0) {
			output.data[i] *= 0.001f;
		} else if (m_labels[i].compare("syn") == 0) {
			output.data[i] *= 0.01f;
		} else if (m_labels[i].compare("tim") == 0) {
			output.data[i] *= 0.01f;
		} else if (m_labels[i].compare("clo") == 0) {
			output.data[i] *= 0.001f;
		}
	}
	
	for (int i = 0; i < output.size; i++) {
		if (output.data[i] < 0) {
			output.data[i] = 0;
		} else if (output.data[i] > 1) {
			output.data[i] = 1;
		}
	}
}
