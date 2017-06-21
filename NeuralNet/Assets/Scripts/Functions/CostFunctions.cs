using System;
using System.Collections.Generic;
using System.Linq;

using Mathf = UnityEngine.Mathf;


namespace NeuralNet
{
	//TODO: Need derivatives?

	/// <summary>
	/// Gets the "cost", or error, in a neural net output
	///     when compared to the expected output.
	/// </summary>
	public interface ICostFunc
	{
		float Cost(Vector expectedOutputs, Vector actualOutputs);
		float TotalCost(List<float> costs);
	}

	/// <summary>
	/// Computes the square of the distance between the expected and actual values.
	/// </summary>
	public class CostFunc_Quadratic : ICostFunc
	{
		public float Cost(Vector expected, Vector actual)
		{
			float err = 0.0f;
			for (int i = 0; i < expected.Count; ++i)
			{
				float f = expected[i] = actual[i];
				err += f * f;
			}
			return err;
		}
		public float TotalCost(List<float> costs)
		{
			return costs.Sum() / (float)(2 * costs.Count);
		}
	}
}
