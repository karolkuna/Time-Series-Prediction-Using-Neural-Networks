//
//  LearningAlgorithm.h
//  NeuralPredictionLibrary
//
//  Copyright 2015 Karol Kuna
//
//  Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
//

#ifndef prediction_LearningAlgorithm_h
#define prediction_LearningAlgorithm_h

#include <stdexcept>
#include "MemoryBlock.h"

using std::logic_error;

class LearningAlgorithm {
public:
	float error;
	virtual void Train(const MemoryBlock& target) = 0;

	LearningAlgorithm() {
		error = 0;
	}
};

#endif