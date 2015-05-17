//
//  MemoryBlock.cpp
//  NeuralPredictionLibrary
//
//  Copyright 2015 Karol Kuna
//
//  Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
//

#include "MemoryBlock.h"

MemoryBlock::MemoryBlock() {
	data = NULL;
	size = 0;
}

MemoryBlock::MemoryBlock(int size) {
	if (size < 0) {
		throw std::logic_error("Invalid MemoryBlock size!");
	}
	this->data = new float[size];
	this->size = size;
}

MemoryBlock::MemoryBlock(const MemoryBlock& source) {
	if (source.size > 0)
	{
		data = new float[source.size];
		memcpy(data, source.data, source.size * sizeof(float));
	} else {
		data = NULL;
	}
		size = source.size;
}

MemoryBlock& MemoryBlock::operator= (const MemoryBlock& source) {
	if (this != &source) //avoid invalid self-assignment
	{
		if (data != NULL) {
			delete[] data;
		}
		
		data = new float[source.size];
		memcpy(data, source.data, source.size * sizeof(float));
		size = source.size;
	}
	// by convention, always return *this
	return *this;
}

MemoryBlock::~MemoryBlock() {
	if (data != NULL) {
		delete[] data;
	}
}

void MemoryBlock::Fill(float value) {
	for (int i = 0; i < size; i++) {
		data[i] = value;
	}
}

void MemoryBlock::GenerateNormal(float mean, float stdev) {
	std::random_device generator;
	std::normal_distribution<float> distribution(mean,stdev);
	
	for (int i = 0; i < size; i++) {
		data[i] = distribution(generator);
	}
}

void MemoryBlock::GenerateUniform(float min, float max) {
	std::random_device generator;
	std::uniform_real_distribution<float> distribution(min,max);
	
	for (int i = 0; i < size; i++) {
		data[i] = distribution(generator);
	}
}

float MemoryBlock::Sum() const {
	float sum = 0.0f;
	for (int i = 0; i < size; i++) {
		sum += data[i];
	}
	return sum;
}

float MemoryBlock::SquareSum() const {
	float sum = 0.0f;
	for (int i = 0; i < size; i++) {
		sum += data[i] * data[i];
	}
	return sum;
}

void MemoryBlock::Add(const MemoryBlock& operand) {
	if (this->size != operand.size) {
		throw std::logic_error("Operands must be of the same size!");
	}
	
	for (int i = 0; i < size; i++) {
		data[i] = data[i] + operand.data[i];
	}
}

void MemoryBlock::Add(float operand) {
	for (int i = 0; i < size; i++) {
		data[i] = data[i] + operand;
	}
}

void MemoryBlock::Subtract(const MemoryBlock& operand) {
	if (this->size != operand.size) {
		throw std::logic_error("Operands must be of the same size!");
	}
	
	for (int i = 0; i < size; i++) {
		data[i] = data[i] - operand.data[i];
	}
}

void MemoryBlock::Subtract(float operand) {
	for (int i = 0; i < size; i++) {
		data[i] = data[i] + operand;
	}
}

void MemoryBlock::Multiply(const MemoryBlock& operand) {
	if (this->size != operand.size) {
		throw std::logic_error("Operands must be the same size!");
	}
	
	for (int i = 0; i < size; i++) {
		data[i] = data[i] * operand.data[i];
	}
}

void MemoryBlock::Multiply(float operand) {
	for (int i = 0; i < size; i++) {
		data[i] = data[i] * operand;
	}
}

void MemoryBlock::LeftShift(unsigned int shift) {
	if (shift > size) {
		throw std::logic_error("Shift out of range");
	}
	
	for (int i = 0; i < size - shift; i++) {
		data[i] = data[i + shift];
	}
	
	for (int i = size - shift; i < size; i++) {
		data[i] = 0;
	}
}

void MemoryBlock::RightShift(unsigned int shift) {
	if (shift > size) {
		throw std::logic_error("Shift out of range");
	}
	
	for (int i = size - 1 - shift; i >= 0; i--) {
		data[i + shift] = data[i];
	}
	
	for (int i = 0; i < shift; i++) {
		data[i] = 0;
	}
}

void MemoryBlock::CopyTo(MemoryBlock& target) const{
	if (this->size != target.size) {
		throw std::logic_error("Source and target must be the same size!");
	}
	
	CopyTo(target, 0, 0, size);
}

void MemoryBlock::CopyTo(MemoryBlock& target, int sourceOffset, int targetOffset, int count) const {
	if (sourceOffset < 0 || targetOffset < 0 || count < 0) {
		throw std::logic_error("Invalid arguments provided");
	}
	if (sourceOffset + count > size || targetOffset + count > target.size) {
		throw std::logic_error("Elements out of range");
	}
	
	memcpy(target.data + targetOffset, data + sourceOffset, count * sizeof(float));
}

void MemoryBlock::Print() const {
	for (int i = 0; i < size; i++) {
		std::cout << data[i] << ' ';
	}
	std::cout << std::endl;
}

