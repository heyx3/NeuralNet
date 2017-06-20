using System;
using Mathf = UnityEngine.Mathf;

namespace NeuralNet
{
	/// <summary>
	/// Initializes the weights and biases for a neural network.
	/// </summary>
	public interface IValueInitializer
	{
		/// <summary>
		/// Generates values into the given weight matrix and bias vector.
		/// </summary>
		/// <param name="layerIndex">
		/// The 0-based index of the current layer.
		/// Note that layer 0 is the input layer, which doesn't need weights/biases,
		///     so this parameter should always be greater than 0.
		/// </param>
		void Init(System.Random rng, Matrix weights, Vector biases, int layerIndex);
	}

	public class ValueInitializer_Gaussian : IValueInitializer
	{
		public float Mean, StandardDeviation;
		public ValueInitializer_Gaussian(float mean = 0.0f, float standardDeviation = 1.0f)
		{
			Mean = mean;
			StandardDeviation = standardDeviation;
		}
		public void Init(System.Random rng, Matrix weights, Vector biases, int layerIndex)
		{
			for (int i = 0; i < biases.Count; ++i)
				biases[i] = rng.NextGaussian(Mean, StandardDeviation);
			for (int row = 0; row < weights.NRows; ++row)
				for (int col = 0; col < weights.NColumns; ++col)
					weights[row, col] = rng.NextGaussian(Mean, StandardDeviation);
		}
	}
}
