//
//  MemoryBlock.h
//  NeuralPredictionLibrary
//
//  Copyright 2015 Karol Kuna
//
//  Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
//

#ifndef __prediction__MemoryBlock__
#define __prediction__MemoryBlock__

#include <iostream>
#include <cstring>
#include <random>
#include <stdexcept>

using namespace std;

class MemoryBlock {
public:
	float* data;
	int size;
	
	MemoryBlock();
	MemoryBlock(int size);
	MemoryBlock(const MemoryBlock& source);
	
	MemoryBlock& operator= (const MemoryBlock& source);
	~MemoryBlock();
	
	void Fill(float value);
	void GenerateNormal(float mean, float stdev);
	void GenerateUniform(float min, float max);
	
	float Sum() const;
	float SquareSum() const;
	
	void Add(const MemoryBlock& operand);
	void Add(float operand);
	void Subtract(const MemoryBlock& operand);
	void Subtract(float operand);
	void Multiply(const MemoryBlock& operand);
	void Multiply(float operand);
	
	void LeftShift(unsigned int shift);
	void RightShift(unsigned int shift);
	
	void CopyTo(MemoryBlock& target) const;
	void CopyTo(MemoryBlock& target, int sourceOffset, int targetOffset, int count) const;
	
	void Print() const;
};

#endif /* defined(__prediction__MemoryBlock__) */
