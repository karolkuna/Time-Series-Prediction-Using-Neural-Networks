//
//  MemoryBlockView.cpp
//  NeuralPredictionLibrary
//
//  Copyright 2015 Karol Kuna
//
//  Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
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