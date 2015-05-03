//
//  dstatParser.cpp
//  prediction
//
//  Created by Karol Kuna on 02/05/15.
//  Copyright (c) 2015 Karol Kuna. All rights reserved.
//

#include <iostream>
#include <string>
#include "MemoryBlock.h"

using std::cin;
using std::cout;
using std::endl;
using std::string;

class dstatParser {
	float GetValue(const string& word) {
		char lastChar = word.back();
		float value;
		
		if (lastChar == 'k' || lastChar == 'M' || lastChar == 'G') {
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
	
	void StoreValue(float value, std::size_t pos, const MemoryBlock& input, const MemoryBlock& output) {
		if (pos < 12)
			input.data[pos] = value;
		else if (pos < 14)
			output.data[pos - 13] = value;
		else
			input.data[pos - 2] = value;
	}
	
public:
	
	bool Parse(const MemoryBlock& input, const MemoryBlock& output) {
		string line;
		if (getline(cin, line)) {
			std::size_t prev = 0, pos;
			while ((pos = line.find_first_of(" ';|", prev)) != std::string::npos)
			{
				if (pos > prev) {
					if (pos < 12)
						StoreValue(GetValue(line.substr(prev, pos-prev)), pos, input, output);
					
				}
				prev = pos+1;
			}
			if (prev < line.length())
				StoreValue(GetValue(line.substr(prev, std::string::npos)), pos, input, output);
			return true;
		} else {
			return false;
		}
	}
};