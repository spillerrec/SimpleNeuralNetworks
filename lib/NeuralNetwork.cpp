#include "NeuralNetwork.hpp"

#include <iostream>
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

FeedForwardLayer NeuralNetwork::operator()( const FeedForwardLayer& input ) const{
	FeedForwardLayer hidden = input;
	for( auto& w : weights ){
		hidden *= w;
		applyActivation( hidden );
	}
	
	return hidden;
}
