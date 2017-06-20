using System;
using Assert = UnityEngine.Assertions.Assert;
using Mathf = UnityEngine.Mathf;


namespace NeuralNet
{
	/// <summary>
	/// A layer of neurons, with a set of weights and biases.
	/// </summary>
	public class NeuronLayer
	{
		/// <summary>
		/// The weights.
		/// Each row 'j' is the list of all weights for neuron 'j' in this layer.
		/// Each element 'k' in row 'j' is the weight from the previous layer's neuron 'k' to this layer's neuron j'.
		/// </summary>
		public Matrix Weights { get; private set; }
		/// <summary>
		/// The biases.
		/// Each element 'j' is the bias for neuron 'j' in this layer.
		/// </summary>
		public Vector Biases { get; private set; }
		/// <summary>
		/// The filter for this layer's outputs.
		/// </summary>
		public IActivationFunc ActivationFunc { get; private set; }

		public int NNodes { get { return Biases.Count; } }

		public NeuronLayer(Matrix weights, Vector biases, IActivationFunc activationFunc)
		{
			Assert.AreEqual(Weights.NColumns, Biases.Count);

			Weights = weights;
			Biases = biases;
			ActivationFunc = activationFunc;
		}

		/// <summary>
		/// Gets the outputs of this layer given the previous layer's outputs.
		/// Also gets the derivatives of the outputs with respect to the node's activation.
		/// </summary>
		public void Evaluate(Vector previousLayerOutputs,
						     ref Vector values, ref Vector activationFuncDerivatives)
		{
			Assert.AreEqual(Biases.Count, previousLayerOutputs.Count);

			//Make sure the output vectors are initialized.
			if (values == null || values.Count != NNodes)
				values = new Vector(NNodes);
			if (activationFuncDerivatives == null || activationFuncDerivatives.Count != NNodes)
				activationFuncDerivatives = new Vector(NNodes);

			//Get the activation, and evaluate it with the activation function.
			ActivationFunc.Evaluate(new Vector(Weights, previousLayerOutputs) + Biases,
									values, activationFuncDerivatives);
		}
	}
}
