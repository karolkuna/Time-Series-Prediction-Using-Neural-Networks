//
//  TBPTT.h
//  prediction
//
//  Created by Karol Kuna on 20/04/15.
//  Copyright (c) 2015 Karol Kuna. All rights reserved.
//

#ifndef __prediction__TBPTT__
#define __prediction__TBPTT__

#include <stdio.h>
#include "LearningAlgorithm.h"
#include "CWRecurrentNetwork.h"

class CWRecurrentNetwork;

class TBPTT : public LearningAlgorithm {
private:
	NeuralLayer* m_unfoldedThresholdLayer;
	vector<NeuralLayer*> m_unfoldedInputLayers;
	vector< std::vector<NeuralLayer*> > m_unfoldedHiddenLayerModules;
	vector<NeuralLayer*> m_unfoldedOutputLayers;
	vector<MemoryBlock> m_unfoldedTargets;
	
	void UpdateUnfoldedWeights();
public:
	CWRecurrentNetwork* network;
	float learningRate, momentumRate;
	int depth;
	
	TBPTT(float learningRate, float momentumRate, int depth);
	~TBPTT();
	
	void Init(CWRecurrentNetwork* network);
	void Train(MemoryBlock& target);
};

#endif /* defined(__prediction__TBPTT__) */
