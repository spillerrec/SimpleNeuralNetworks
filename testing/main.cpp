#include <NeuralNetwork.hpp>

#include <iostream>

using namespace std;
using namespace SimpleNeuralNetworks;

int main( int argc, char *argv[] ){
	int input_size = 2;
	int output_size = 1;
	NeuralNetwork net( { input_size, 2, output_size } );
	
	FeedForwardLayer x1(2), x2(2), x3(2), x4(2);
	FeedForwardLayer y1(1), y2(1), y3(1), y4(1);
	x1 << -1, -1; y1 << -1;
	x2 << -1, +1; y2 << +1;
	x3 << +1, -1; y3 << +1;
	x4 << +1, +1; y4 << +1;
	
	for( int i=0; i<10000; i++ ){
		net.backpropagate( x1, y1 );
		net.backpropagate( x2, y2 );
		net.backpropagate( x3, y3 );
		net.backpropagate( x4, y4 );
	}
	
	cout << "Testing" << endl;
	cout << net( x1 ) << endl;
	cout << net( x2 ) << endl;
	cout << net( x3 ) << endl;
	cout << net( x4 ) << endl;
	
	return 0;
}
