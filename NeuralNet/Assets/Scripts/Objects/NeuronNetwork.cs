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

		/// <summary>
		/// The expected number of nodes in the first layer, a.k.a. the input to this network.
		/// </summary>
		public int NInputNodes { get; private set; }
		/// <summary>
		/// Gets the number of nodes in the layer before the given one.
		/// </summary>
		/// <param name="layerI">
		/// The layer. Note that passing a value of 0 yields the size of the input layer.
		/// </param>
		public int NNodesInPreviousLayer(int layerI)
		{
			return (layerI == 0 ?
						NInputNodes :
						Layers[layerI - 1].NNodes);
		}


		public NeuronNetwork(RNG rng,
							 IActivationFunc activationFunc, IValueInitializer valueInitalizer,
							 params int[] layerSizes)
		{
			NInputNodes = layerSizes[0];

			Layers = new List<NeuronLayer>(layerSizes.Length - 1);
			for (int i = 1; i < layerSizes.Length; ++i)
			{
				Matrix weights = new Matrix(layerSizes[i], layerSizes[i - 1]);
				Vector biases = new Vector(layerSizes[i]);
				valueInitalizer.Init(rng, weights, biases, i);

				Layers.Add(new NeuronLayer(weights, biases, activationFunc));
			}
		}


		/// <summary>
		/// Gets the output of this network given the input.
		/// </summary>
		/// <param name="out_LayerWeightedInputs">
		/// The weighted inputs into each node in each layer,
		///     in order (ignoring the input layer).
		/// Assumes it already has the right number of elements
		///     and each element has the right number of components.
		/// </param>
		/// <param name="out_LayerOutputs">
		/// Each node layer's output, in order (ignoring the input layer).
		/// Assumes it already has the right number of elements
		///     and each element has the right number of components.
		/// </param>
		/// <param name="out_LayerDerivatives">
		/// The derivative of each node layer's outputs,
		///     with respect to their weighted inputs.
		/// Assumes it already has the right number of elements
		///     and each element has the right number of components.
		/// </param>
		public void Evaluate(Vector inputs,
							 List<Vector> out_LayerWeightedInputs,
							 List<Vector> out_LayerOutputs,
							 List<Vector> out_LayerDerivatives)
		{
			//Make sure everything is sized properly.
			Assert.AreEqual(NInputNodes, inputs.Count, "Wrong number of inputs in sample");
			Assert.AreEqual(Layers.Count, out_LayerWeightedInputs.Count,
							"Layers list and Inputs list have different sizes");
			Assert.AreEqual(out_LayerWeightedInputs.Count, out_LayerOutputs.Count,
							"Inputs list and Ouputs list have different sizes");
			Assert.AreEqual(out_LayerOutputs.Count, out_LayerDerivatives.Count,
							"Outputs list and Derivatives list have different sizes");
			for (int i = 0; i < out_LayerWeightedInputs.Count; ++i)
			{
				Assert.AreEqual(Layers[i].NNodes, out_LayerWeightedInputs[i].Count,
								"WeightedInput list has wrong size at element " + i.ToString());
				Assert.AreEqual(Layers[i].NNodes, out_LayerOutputs[i].Count,
								"Output list has wrong size at element " + i.ToString());
				Assert.AreEqual(Layers[i].NNodes, out_LayerDerivatives[i].Count,
								"Derivatives list has wrong size at element " + i.ToString());
			}

			//Evaluate each layer in order.
			Vector previousLayerOutput = inputs;
			for (int i = 0; i < Layers.Count; ++i)
			{
				Layers[i].Evaluate(previousLayerOutput,
								   out_LayerWeightedInputs[i],
								   out_LayerOutputs[i],
								   out_LayerDerivatives[i]);

				previousLayerOutput = out_LayerOutputs[i];
			}
		}
	}
}
