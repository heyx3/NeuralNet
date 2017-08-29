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
		public IActivationFunc ActivationFunc { get; set; }

		public int NNodes { get { return Biases.Count; } }

		public NeuronLayer(Matrix weights, Vector biases, IActivationFunc activationFunc)
		{
			Assert.AreEqual(weights.NRows, biases.Count);

			Weights = weights;
			Biases = biases;
			ActivationFunc = activationFunc;
		}

		/// <summary>
		/// Gets the outputs of this layer given the previous layer's outputs.
		/// Also gets the derivatives of the outputs with respect to the weighted inputs.
		/// The "Weighted input" of a node is the combination of all input nodes plus the bias;
		///     it is the value that gets filtered by the ActivationFunc into an output value.
		/// </summary>
		public void Evaluate(Vector previousLayerOutputs,
						     Vector out_WeightedInputs, Vector out_Outputs,
							 Vector out_ActivationFuncDerivatives)
		{
			Assert.AreEqual(Weights.NColumns, previousLayerOutputs.Count);

			//Get the activation, and evaluate it with the activation function.
			out_WeightedInputs = new Vector(Weights, previousLayerOutputs) + Biases;
			ActivationFunc.Evaluate(out_WeightedInputs, out_Outputs,
									out_ActivationFuncDerivatives);
		}

		public void Resize(int size, int prevLayerSize)
		{
			Matrix oldWeights = Weights;
			Vector oldBiases = Biases;
			int oldNNodes = NNodes,
				oldNPrevNodes = Weights.NColumns;

			Weights = new Matrix(size, prevLayerSize);
			Biases = new Vector(size);

			for (int i = 0; i < oldNNodes & i < NNodes; ++i)
			{
				Biases[i] = oldBiases[i];
				for (int j = 0; j < Weights.NColumns; ++j)
					if (j < oldNPrevNodes)
						Weights[i, j] = oldWeights[i, j];
					else
						Weights[i, j] = 0.0f;
			}
			for (int i = oldNNodes; i < NNodes; ++i)
			{
				Biases[i] = 0.0f;
				for (int j = 0; j < Weights.NColumns; ++j)
					Weights[i, j] = 0.0f;
			}
		}
	}
}
