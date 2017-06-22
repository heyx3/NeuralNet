using System;
using Mathf = UnityEngine.Mathf;

namespace NeuralNet
{
	/// <summary>
	/// A filter for the weighted input into a node.
	/// Operates on a layer of nodes at a time.
	/// </summary>
	public interface IActivationFunc
	{
		/// <summary>
		/// Gets the output for the given weighted input.
		/// </summary>
		void Evaluate(Vector nodeLayerOutput, Vector out_Value);
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
		void Evaluate(Vector nodeLayerOutput, Vector out_Value, Vector out_Derivative);
	}

	public class ActivationFunc_Logistic : IActivationFunc
	{
		public void Evaluate(Vector nodeOutputs, Vector out_Value)
		{
			for (int i = 0; i < nodeOutputs.Count; ++i)
				out_Value[i] = 1.0f / (1.0f + Mathf.Exp(-nodeOutputs[i]));
		}
		public void Evaluate(Vector nodeOutputs, Vector out_Value, Vector out_Derivative)
		{
			Evaluate(nodeOutputs, out_Value);
			for (int i = 0; i < nodeOutputs.Count; ++i)
				out_Derivative[i] = out_Value[i] * (1.0f - out_Value[i]);
		}
	}
}
