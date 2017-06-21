using System;
using System.Collections.Generic;

using RNG = System.Random;
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

		public NeuronNetwork(RNG rng,
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
		/// <param name="out_LayerValues">
		/// Each node layer's output, in order (ignoring the input layer).
		/// Assumes it already has the right number of elements.
		/// The vectors will be resized if they're the wrong size.
		/// </param>
		/// <param name="out_LayerDerivatives">
		/// The derivative of each node's activation function output,
		///     with respect to the node's raw output.
		/// Assumes it already has the right number of elements,
		///     and that each element has the right number of components.
		/// The vectors will be resized if they're the wrong size.
		/// </param>
		public void Evaluate(Vector inputs,
							 List<Vector> out_LayerValues, List<Vector> out_LayerDerivatives)
		{
			//Make sure everything is sized properly.
			Assert.AreEqual(out_LayerValues.Count, out_LayerDerivatives.Count,
							"Value list and Derivative list have different sizes");

			//Evaluate each layer in order.
			Vector previousLayerOutput = inputs;
			for (int i = 0; i < Layers.Count; ++i)
			{
				Vector value = out_LayerValues[i],
					   derivative = out_LayerDerivatives[i];
				Layers[i].Evaluate(previousLayerOutput, ref value, ref derivative);
				out_LayerValues[i] = value;
				out_LayerDerivatives[i] = derivative;

				previousLayerOutput = value;
			}
		}
	}
}
