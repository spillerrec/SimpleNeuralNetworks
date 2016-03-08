/*	This file is part of SimpleNeuralNetworks, which is free software and is licensed
 * under the terms of the GNU GPL v3.0. (see http://www.gnu.org/licenses/ ) */

#ifndef NEURAL_NETWORK_HPP
#define NEURAL_NETWORK_HPP

#include <eigen3/Eigen/Core>

#include <vector>

namespace SimpleNeuralNetworks{

using FeedForwardLayer = Eigen::RowVectorXf;
using OutputLayer = FeedForwardLayer;

using Weights = Eigen::MatrixXf;

class NeuralNetwork{
	private:
		struct Layer{
			Weights weight;
			FeedForwardLayer bias;
		};
		std::vector<Layer> weights;
		
		Layer initializeWeights( int previous, int next );
		
	public:
		NeuralNetwork( const std::vector<int>& layer_sizes );
		FeedForwardLayer operator()( const FeedForwardLayer& ) const;
		
		void backpropagate( const FeedForwardLayer&, const FeedForwardLayer& );
};

}

#endif
