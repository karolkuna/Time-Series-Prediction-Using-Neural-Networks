//
//  LearningAlgorithm.h
//  prediction
//
//  Created by Karol Kuna on 20/04/15.
//  Copyright (c) 2015 Karol Kuna. All rights reserved.
//

#ifndef prediction_LearningAlgorithm_h
#define prediction_LearningAlgorithm_h

#include "MemoryBlock.h"

class LearningAlgorithm {
public:
	float error = 0;
	virtual void Train(MemoryBlock& target) = 0;
};

#endif