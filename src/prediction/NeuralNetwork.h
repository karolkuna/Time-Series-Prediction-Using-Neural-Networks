//
//  NeuralNetwork.h
//  prediction
//
//  Created by Karol Kuna on 09/02/15.
//  Copyright (c) 2015 Karol Kuna. All rights reserved.
//

#ifndef __prediction__NeuralNetwork__
#define __prediction__NeuralNetwork__

#include <stdio.h>
#include "MemoryBlock.h"

class NeuralNetwork {
public:
	MemoryBlock output;
	virtual void Propagate(const MemoryBlock& input) = 0;
};

#endif /* defined(__prediction__NeuralNetwork__) */
