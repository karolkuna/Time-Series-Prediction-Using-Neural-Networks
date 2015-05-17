//
//  TBPTT.h
//  NeuralPredictionLibrary
//
//  Copyright 2015 Karol Kuna
//
//  Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
//

#ifndef __prediction__TBPTT__
#define __prediction__TBPTT__

#include <stdio.h>
#include <stdexcept>
#include "LearningAlgorithm.h"
#include "CWRecurrentNetwork.h"

using std::logic_error;

class TBPTT : public LearningAlgorithm {
private:
	NeuralLayer* m_unfoldedThresholdLayer;
	vector<NeuralLayer*> m_unfoldedInputLayers;
	vector< std::vector<NeuralLayer*> > m_unfoldedHiddenLayerModules;
	vector<NeuralLayer*> m_unfoldedOutputLayers;
	vector<MemoryBlock> m_unfoldedTargets;

	CWRecurrentNetwork* m_network;
	float m_learningRate, m_momentumRate;
	int m_depth;

	void CreateUnfoldedNetwork();
	void UpdateUnfoldedWeights();
	
public:
	
	TBPTT(CWRecurrentNetwork* network, float learningRate, float momentumRate, int depth);
	~TBPTT();
	
	void Train(const MemoryBlock& target);
};

#endif /* defined(__prediction__TBPTT__) */
