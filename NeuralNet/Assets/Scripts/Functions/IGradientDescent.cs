using System;
using System.Collections.Generic;
using System.Linq;

using Mathf = UnityEngine.Mathf;


namespace NeuralNet
{
	/// <summary>
	/// Moves the network towards a lower-cost value,
	///     given the recommended directions to move in.
	/// </summary>
	public interface IGradientDescent
	{
		void ModifyNetwork(NeuronNetwork network,
						   uint miniBatchCount, uint epochCount,
		 			       List<Vector> biasDerivativesByLayer,
						   List<Matrix> weightDerivativesByLayer);
	}


	/// <summary>
	/// A simple gradient descent algorithm with a constant learning rate.
	/// </summary>
	public class GradientDescent_Constant : IGradientDescent
	{
		public float LearningRate;
		public GradientDescent_Constant(float learningRate) { LearningRate = learningRate; }
		public void ModifyNetwork(NeuronNetwork network,
								  uint miniBatchCount, uint epochCount,
		 						  List<Vector> biasDerivativesByLayer,
								  List<Matrix> weightDerivativesByLayer)
		{
			//Adjust the network to make the cost function smaller.
			float delta = -LearningRate;
			for (int layerI = 0; layerI < network.Layers.Count; ++layerI)
			{
				for (int nodeI = 0; nodeI < network.Layers[layerI].NNodes; ++nodeI)
				{
					network.Layers[layerI].Biases[nodeI] +=
						biasDerivativesByLayer[layerI][nodeI] * delta;

					int nPrevNodes = network.Layers[layerI].Weights.NColumns;
					for (int previousNodeI = 0; previousNodeI < nPrevNodes; ++previousNodeI)
					{
						network.Layers[layerI].Weights[nodeI, previousNodeI] +=
							weightDerivativesByLayer[layerI][nodeI, previousNodeI] * delta;
					}
				}
			}
		}
	}

	//TODO: A version of gradient descent that halves the learning rate when the dot product of the previous gradient and current gradient is negative.
}
