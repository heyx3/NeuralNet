using System;
using Mathf = UnityEngine.Mathf;

public static class MyMath
{
	//Source for System.Random extensions: https://bitbucket.org/Superbest/superbest-random

	/// <summary>
	/// Generates normally distributed numbers.
	/// Each operation makes two Gaussians for the price of one,
	///     and apparently they can be cached or something for better performance, but who cares.
	/// </summary>
	public static float NextGaussian(this Random r,
									 float mean = 0.0f, float standardDeviation = 1.0f)
	{
		var u1 = (float)r.NextDouble();
		var u2 = (float)r.NextDouble();

		var rand_std_normal = Mathf.Sqrt(-2.0f * Mathf.Log(u1)) *
							  Mathf.Sin(2.0f * Mathf.PI * u2);

		var rand_normal = mean + (standardDeviation * rand_std_normal);

		return rand_normal;
	}

	/// <summary>
	/// Generates values from a triangular distribution.
	/// </summary>
	/// <remarks>
	/// See http://en.wikipedia.org/wiki/Triangular_distribution
	///     for a description of the triangular probability distribution
	///     and the algorithm for generating one.
	/// </remarks>
	/// <param name = "c">Mode (most frequent value)</param>
	public static float NextTriangular(this Random r, float a, float b, float c)
	{
		var u = (float)r.NextDouble();

		return (u < (c - a) / (b - a)) ?
			       a + Mathf.Sqrt(u * (b - a) * (c - a)) :
				   b - Mathf.Sqrt((1 - u) * (b - a) * (b - c));
	}

	/// <summary>
	/// Equally likely to return true or false.
	/// </summary>
	public static bool NextBoolean(this Random r)
	{
		return r.Next(2) > 0;
	}
}