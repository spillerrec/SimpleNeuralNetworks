/*	This file is part of SimpleNeuralNetworks, which is free software and is licensed
 * under the terms of the GNU GPL v3.0. (see http://www.gnu.org/licenses/ ) */

#include "NeuralNetwork.hpp"

#include <cmath>

using namespace std;
using namespace SimpleNeuralNetworks;


NeuralNetwork::NeuralNetwork( const vector<int>& layer_sizes ){
	int previous = layer_sizes[0];
	for( unsigned i=1; i<layer_sizes.size(); i++ ){
		weights.emplace_back( initializeWeights( previous, layer_sizes[i] ) );
		previous = layer_sizes[i];
	}
}

Weights NeuralNetwork::initializeWeights( int previous, int next ){ //TODO: pass rnd generator
	auto w = Weights::Random( previous, next );
	//TODO: scale properly
	return w;
}

static float hyperbolicTangent( float x ){
	return tanh( x );
}

static void applyActivation( FeedForwardLayer& hidden ){
	for( unsigned i=0; i<hidden.cols(); i++ ) //TODO: do better...
		hidden(i) = hyperbolicTangent( hidden(i) );
}

static void applyDerivation( FeedForwardLayer& hidden ){
	for( unsigned i=0; i<hidden.cols(); i++ ){ //TODO: do better...
		double sech = 1.0 / std::cosh(hidden(i));
		hidden(i) = sech * sech;
	}
}

FeedForwardLayer NeuralNetwork::operator()( const FeedForwardLayer& input ) const{
	FeedForwardLayer hidden = input;
	for( auto& w : weights ){
		hidden *= w;
		applyActivation( hidden );
	}
	
	return hidden;
}
void NeuralNetwork::backpropagate( const FeedForwardLayer& input, const FeedForwardLayer& wanted ){
	struct Cache{
		FeedForwardLayer s; //Hidden value
		FeedForwardLayer z; //Activated value
		FeedForwardLayer d; //Delta
		Cache() { }
		Cache( FeedForwardLayer in ) : z(in) { }
	};
	
	//Forward
	vector<Cache> values{ input };
	auto hidden = input;
	for( auto& w : weights ){
		Cache cache;
		cache.s = (hidden *= w);
		applyActivation( hidden );
		cache.z = hidden;
		values.push_back( cache );
	}
	
	//Delta for output layer (MSE)
	values.back().d = values.back().z - wanted;
	
	float learning_rate = 0.01;
	for( int i=values.size()-2; i>0; i-- ){ //Skip first and last layer
		//Calculate delta D1 = f'( S1 ) . W1->2 * D2^T
		values[i].d = weights[i] * values[i+1].d.transpose();
		applyDerivation( values[i].s );
		values[i].d.array() *= values[i].s.array();
	}
	
	//Update weights W1 = W1 - n * Z^T * D2
	for( unsigned i=0; i<values.size()-1; i++ )
		weights[i] -= learning_rate * ( values[i].z.transpose() * values[i+1].d );
}
