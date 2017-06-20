using System;
using Mathf = UnityEngine.Mathf;

namespace NeuralNet
{
	/// <summary>
	/// A filter for the output of a node.
	/// More specifically, for a whole LAYER of nodes.
	/// </summary>
	public interface IActivationFunc
	{
		/// <summary>
		/// Gets the value for the given node outputs.
		/// </summary>
		void Evaluate(Vector nodeLayerOutput, Vector outValue);
		/// <summary>
		/// Gets the value and the derivative for the given node outputs.
		/// </summary>
		/// <param name="nodeLayerOutput">
		/// A vector where each component is the output of a node in the layer.
		/// </param>
		/// <param name="outValue">
		/// The filtered output of each node.
		/// </param>
		/// <param name="outDerivative">
		/// The derivative of the filtered output of each node.
		/// </param>
		void Evaluate(Vector nodeLayerOutput, Vector outValue, Vector outDerivative);
	}

	public class ActivationFunc_Logistic : IActivationFunc
	{
		public void Evaluate(Vector nodeOutputs, Vector outValue)
		{
			for (int i = 0; i < nodeOutputs.Count; ++i)
				outValue[i] = 1.0f / (1.0f + Mathf.Exp(-nodeOutputs[i]));
		}
		public void Evaluate(Vector nodeOutputs, Vector outValue, Vector outDerivative)
		{
			Evaluate(nodeOutputs, outValue);
			for (int i = 0; i < nodeOutputs.Count; ++i)
				outDerivative[i] = outValue[i] * (1.0f - outValue[i]);
		}
	}
}
