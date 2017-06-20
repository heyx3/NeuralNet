using System;
using System.Collections.Generic;
using Assert = UnityEngine.Assertions.Assert;
using Mathf = UnityEngine.Mathf;


namespace NeuralNet
{
	/// <summary>
	/// Trains a neural network to give the expected results for the given inputs.
	/// </summary>
	public class NetworkTrainer
	{
		public NeuronNetwork Network;
		public ICostFunc CostFunc;

		/// <summary>
		/// Pairs of expected input-output values.
		/// </summary>
		public Dictionary<Vector, Vector> TrainingSamples;


		public NetworkTrainer(NeuronNetwork network, ICostFunc costFunc,
							  Dictionary<Vector, Vector> trainingSamples)
		{
			Network = network;
			CostFunc = costFunc;
			TrainingSamples = trainingSamples;
		}
	}
}
