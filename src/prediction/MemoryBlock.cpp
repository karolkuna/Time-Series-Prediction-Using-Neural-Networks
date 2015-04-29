//
//  MemoryBlock.cpp
//  prediction
//
//  Created by Karol Kuna on 09/02/15.
//  Copyright (c) 2015 Karol Kuna. All rights reserved.
//

#include "MemoryBlock.h"

void MemoryBlock::Validate() {
	for (int i = 0; i < size; i++) {
		if (data[i] == NAN || data[i] == INFINITY || data[i] == -INFINITY) {
			throw std::logic_error("Invalid value");
		}
		if (data[i] > 10000 || data[i] < -10000) {
			throw std::logic_error("Invalid value");
		}
	}
}

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
	std::default_random_engine generator(std::random_device{}());
	std::normal_distribution<float> distribution(mean,stdev);
	
	for (int i = 0; i < size; i++) {
		data[i] = distribution(generator);
	}
}

void MemoryBlock::GenerateUniform(float min, float max) {
	std::default_random_engine generator(std::random_device{}());
	std::uniform_real_distribution<float> distribution(min,max);
	
	for (int i = 0; i < size; i++) {
		data[i] = distribution(generator);
	}
}

float MemoryBlock::Sum() {
	float sum = 0.0f;
	for (int i = 0; i < size; i++) {
		sum += data[i];
	}
	return sum;
}

float MemoryBlock::SquareSum() {
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

void MemoryBlock::Subtract(const MemoryBlock& operand) {
	if (this->size != operand.size) {
		throw std::logic_error("Operands must be of the same size!");
	}
	
	for (int i = 0; i < size; i++) {
		data[i] = data[i] - operand.data[i];
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

const void MemoryBlock::CopyTo(MemoryBlock& target) {
	if (this->size != target.size) {
		throw std::logic_error("Source and target must be the same size!");
	}
	
	CopyTo(target, 0, 0, size);
}

const void MemoryBlock::CopyTo(MemoryBlock& target, int sourceOffset, int targetOffset, int count) {
	if (sourceOffset < 0 || targetOffset < 0 || count < 0) {
		throw std::logic_error("Invalid arguments provided");
	}
	if (sourceOffset + count > size || targetOffset + count > target.size) {
		throw std::logic_error("Elements out of range");
	}
	
	memcpy(target.data + targetOffset, data + sourceOffset, count * sizeof(float));
}

void MemoryBlock::Print() {
	for (int i = 0; i < size; i++) {
		std::cout << data[i] << ' ';
	}
	std::cout << std::endl;
}
