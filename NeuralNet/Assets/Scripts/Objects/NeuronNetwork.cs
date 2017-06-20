using System;
using System.Collections.Generic;

using Assert = UnityEngine.Assertions.Assert;
using Mathf = UnityEngine.Mathf;


namespace NeuralNet
{
	/// <summary>
	/// A set of layers of neurons.
	/// </summary>
	public class NeuronNetwork
	{
		/// <summary>
		/// All layers of the network in order, EXCEPT for the input layer.
		/// </summary>
		public List<NeuronLayer> Layers { get; private set; }

		public NeuronNetwork(System.Random rng,
							 IActivationFunc activationFunc, IValueInitializer valueInitalizer,
							 params int[] layerSizes)
		{
			Layers = new List<NeuronLayer>(layerSizes.Length);
			for (int i = 1; i < layerSizes.Length; ++i)
			{
				Matrix weights = new Matrix(layerSizes[i], layerSizes[i - 1]);
				Vector biases = new Vector(layerSizes[i]);
				valueInitalizer.Init(rng, weights, biases, i);

				Layers[i - 1] = new NeuronLayer(weights, biases, activationFunc);
			}
		}

		/// <summary>
		/// Gets the output of this network given the input.
		/// </summary>
		public Vector Evaluate(Vector inputs)
		{
			Vector previousLayerOutput = inputs;
			foreach (var layer in Layers)
			{
				Vector nextLayerOutput = null,
					   nextLayerOutputDerivatives = null;
				layer.Evaluate(previousLayerOutput,
							   ref nextLayerOutput,
							   ref nextLayerOutputDerivatives);

				previousLayerOutput = nextLayerOutput;
			}

			return previousLayerOutput;
		}
	}
}
