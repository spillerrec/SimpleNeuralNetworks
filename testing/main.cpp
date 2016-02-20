#include <NeuralNetwork.hpp>

#include <iostream>

using namespace std;
using namespace SimpleNeuralNetworks;

int main( int argc, char *argv[] ){
	int input_size = 3;
	int output_size = 2;
	NeuralNetwork net( {  input_size, 5, output_size } );
	FeedForwardLayer inputs( input_size );
	
	auto result = net( inputs );
	cout << result << endl;
	return 0;
}
