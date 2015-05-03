//
//  MemoryBlockView.cpp
//  prediction
//
//  Created by Karol Kuna on 29/04/15.
//  Copyright (c) 2015 Karol Kuna. All rights reserved.
//

#include "MemoryBlockView.h"

MemoryBlockView::MemoryBlockView(const MemoryBlock& block, int from, int size) {
	if (from + size > block.size) {
		throw std::logic_error("Invalid size specified!");
	}
	
	this->data = block.data + from;
	this->size = size;
}

MemoryBlockView::MemoryBlockView(const MemoryBlock& block) {
	this->data = block.data;
	this->size = block.size;
}

MemoryBlockView::~MemoryBlockView() {
	data = NULL;
}