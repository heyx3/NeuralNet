using System;
using System.Collections.Generic;
using System.Linq;

using Assert = UnityEngine.Assertions.Assert;
using Mathf = UnityEngine.Mathf;


namespace NeuralNet
{
	/// <summary>
	/// Gets the "cost", or error, in a neural net output
	///     when compared to the expected output.
	/// </summary>
	public interface ICostFunc
	{
		/// <summary>
		/// Gets the "error" in the network, given the expected output and the actual output.
		/// Also gets the rate of change of this "error" with respect to the actual outputs.
		/// </summary>
		/// <param name="out_Derivatives">
		/// Must be the same size as "expectedOutputs" and "actualOutputs".
		/// </param>
		void GetCost(Vector expectedOutputs, Vector actualOutputs,
					 out float out_Cost, Vector out_Derivatives);
	}

	/// <summary>
	/// Computes the square of the distance between the expected and actual values.
	/// </summary>
	public class CostFunc_Quadratic : ICostFunc
	{
		public void GetCost(Vector expected, Vector actual,
							out float out_Cost, Vector out_Derivatives)
		{
			Assert.AreEqual(expected.Count, actual.Count, "Expected and actual must be same size!");
			Assert.AreEqual(expected.Count, out_Derivatives.Count,
							"Expected and derivatives must be same size!");

			out_Cost = 0.0f;
			for (int i = 0; i < expected.Count; ++i)
			{
				out_Derivatives[i] = expected[i] - actual[i];
				out_Cost += out_Derivatives[i];
			}
			//Halve the cost, so we don't have to double each derivative.
			out_Cost *= 0.5f;
		}
	}
}
