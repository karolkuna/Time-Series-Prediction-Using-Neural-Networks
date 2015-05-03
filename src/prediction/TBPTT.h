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
