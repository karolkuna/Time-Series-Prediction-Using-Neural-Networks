//
//  MemoryBlock.h
//  prediction
//
//  Created by Karol Kuna on 09/02/15.
//  Copyright (c) 2015 Karol Kuna. All rights reserved.
//

#ifndef __prediction__MemoryBlock__
#define __prediction__MemoryBlock__

#include <iostream>
#include <random>

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
	void Validate() const;
};

#endif /* defined(__prediction__MemoryBlock__) */
