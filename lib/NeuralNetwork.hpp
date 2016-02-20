/*
	This file is part of Overmix.

	Overmix is free software: you can redistribute it and/or modify
	it under the terms of the GNU General Public License as published by
	the Free Software Foundation, either version 3 of the License, or
	(at your option) any later version.

	Overmix is distributed in the hope that it will be useful,
	but WITHOUT ANY WARRANTY; without even the implied warranty of
	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
	GNU General Public License for more details.

	You should have received a copy of the GNU General Public License
	along with Overmix.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef NEURAL_NETWORK_HPP
#define NEURAL_NETWORK_HPP

#include <Eigen/Core>

#include <vector>

namespace SimpleNeuralNetworks{

using FeedForwardLayer = Eigen::RowVectorXf;
using OutputLayer = FeedForwardLayer;

using Weights = Eigen::MatrixXf;

class NeuralNetwork{
	private:
		std::vector<Weights> weights;
		
		Weights initializeWeights( int previous, int next );
		
	public:
		NeuralNetwork( const std::vector<int>& layer_sizes );
		FeedForwardLayer operator()( const FeedForwardLayer& ) const;
};

}

#endif