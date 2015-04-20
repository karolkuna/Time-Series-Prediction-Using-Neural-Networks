//
//  ActivationFunction.h
//  prediction
//
//  Created by Karol Kuna on 09/02/15.
//  Copyright (c) 2015 Karol Kuna. All rights reserved.
//

#ifndef __prediction__ActivationFunction__
#define __prediction__ActivationFunction__

#include <stdio.h>
#include <math.h>

class ActivationFunction {
public:
	virtual float operator()(float value) {return 0;};
	virtual float Derivative(float value) {return 0;};
};

class LogisticFunction : public ActivationFunction {
public:
	float operator()(float value) {
		float act = 1.0 / (1.0 + exp(-value));
		//NaN, Inf, -Inf fix
/*		if (value > 18)
			act = 1;
		if (value < -18)
			act = 0;
*/
		return act;
	}
	
	float Derivative(float value) {
		float act = (*this)(value);
		return act * (1 - act);
	}
};

#endif /* defined(__prediction__ActivationFunction__) */
