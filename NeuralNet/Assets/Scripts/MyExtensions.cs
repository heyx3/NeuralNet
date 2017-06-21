using System;
using System.IO;

using RNG = System.Random;
using Mathf = UnityEngine.Mathf;

public static class MyExtensions
{
	//Extend BinaryReader to do big-endian reads.

	/// <summary>
	/// Reads an integer in big-endian.
	/// </summary>
	public static Int32 ReadInt32_BE(this BinaryReader br)
	{
		return (Int32)System.Net.IPAddress.NetworkToHostOrder(br.ReadInt32());
	}


	//Source for System.Random extensions: https://bitbucket.org/Superbest/superbest-random

	/// <summary>
	/// Generates normally distributed numbers.
	/// Each operation makes two Gaussians for the price of one,
	///     and apparently they can be cached or something for better performance, but who cares.
	/// </summary>
	public static float NextGaussian(this RNG rng,
									 float mean = 0.0f, float standardDeviation = 1.0f)
	{
		var u1 = (float)rng.NextDouble();
		var u2 = (float)rng.NextDouble();

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
	public static float NextTriangular(this RNG rng, float a, float b, float c)
	{
		var u = (float)rng.NextDouble();

		return (u < (c - a) / (b - a)) ?
			       a + Mathf.Sqrt(u * (b - a) * (c - a)) :
				   b - Mathf.Sqrt((1 - u) * (b - a) * (b - c));
	}

	/// <summary>
	/// Equally likely to return true or false.
	/// </summary>
	public static bool NextBoolean(this RNG rng)
	{
		return rng.Next(2) > 0;
	}
}