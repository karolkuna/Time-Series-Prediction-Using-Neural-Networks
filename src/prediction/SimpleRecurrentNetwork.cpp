//
//  RecurrentNetwork.cpp
//  prediction
//
//  Created by Karol Kuna on 28/03/15.
//  Copyright (c) 2015 Karol Kuna. All rights reserved.
//

#include "SimpleRecurrentNetwork.h"
vector<int> oneModule = {1};

SimpleRecurrentNetwork::SimpleRecurrentNetwork(int inputUnits, int hiddenUnits, int outputUnits, ActivationFunction* activationFunction) : CWRecurrentNetwork(inputUnits, hiddenUnits, outputUnits, oneModule, activationFunction) {

	hiddenLayer = hiddenLayerModules[0];
	contextLayer = contextLayerModules[0];
}
