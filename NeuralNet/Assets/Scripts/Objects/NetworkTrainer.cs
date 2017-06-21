using System;
using System.Collections.Generic;
using System.Linq;

using RNG = System.Random;
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
		/// Inputs paired with the output they're supposed to have.
		/// Training samples are used for training the network.
		/// Validation samples are used to verify the network isn't overtrained
		///     (sort of like "teaching to the test").
		/// </summary>
		public Dictionary<Vector, Vector> TrainingSamples, ValidationSamples;


		public NetworkTrainer(NeuronNetwork network, ICostFunc costFunc,
							  IEnumerable<KeyValuePair<Vector, Vector>> trainingSamples,
							  IEnumerable<KeyValuePair<Vector, Vector>> validationSamples)
		{
			Network = network;
			CostFunc = costFunc;

			TrainingSamples = new Dictionary<Vector, Vector>();
			foreach (var pair in trainingSamples)
				TrainingSamples.Add(pair.Key, pair.Value);

			ValidationSamples = new Dictionary<Vector, Vector>();
			foreach (var pair in validationSamples)
				ValidationSamples.Add(pair.Key, pair.Value);
		}


		/// <summary>
		/// Runs several iterations of the training algorithm,
		///     where each iteration uses a different subset of the training samples.
		/// The epoch finishes when all training samples have been used.
		/// </summary>
		/// <param name="miniBatchSize">
		/// The number of training samples to use in each iteration.
		/// </param>
		public void RunEpoch(int miniBatchSize, RNG rng)
		{
			var unusedSamples = new List<KeyValuePair<Vector, Vector>>(TrainingSamples.Count);
			var currentSamples = new List<KeyValuePair<Vector, Vector>>(miniBatchSize);

			while (unusedSamples.Count > 0)
			{
				//Get the samples to use for this iteration.
				//If not enough samples are left over for the next batch, use them all now.
				for (int i = 0; i < miniBatchSize && unusedSamples.Count > 0; ++i)
				{
					int sampleI = rng.Next(unusedSamples.Count);
					currentSamples.Add(unusedSamples[sampleI]);
					unusedSamples.RemoveAt(sampleI);
				}
				if (unusedSamples.Count < miniBatchSize)
				{
					currentSamples.AddRange(unusedSamples);
					unusedSamples.Clear();
				}

				RunIteration(currentSamples);
			}
		}
		/// <summary>
		/// Runs a single iteration of the training algorithm.
		/// Returns the total "cost" of all training samples.
		/// </summary>
		/// <param name="sampleBatch">
		/// The training samples to use, or "null" to use all of this instance's samples.
		/// </param>
		public float RunIteration(List<KeyValuePair<Vector, Vector>> sampleBatch = null)
		{
			if (sampleBatch == null)
			{
				sampleBatch = new List<KeyValuePair<Vector, Vector>>(TrainingSamples.Count);
				sampleBatch.AddRange(TrainingSamples);
			}

			//Set up the list of values and derivatives for each layer.
			List<Vector> values = new List<Vector>(),
						 derivatives = new List<Vector>();
			for (int i = 0; i < Network.Layers.Count; ++i)
			{
				values.Add(new Vector(Network.Layers[i].NNodes));
				derivatives.Add(new Vector(Network.Layers[i].NNodes));
			}

			//Get the cost of every sample.
			List<float> costs = new List<float>(sampleBatch.Count);
			for (int i = 0; i < sampleBatch.Count; ++i)
			{
				var sample = sampleBatch[i];

				//Evaluate the network, then get the difference
				//    between what was expected and what was calculated.
				Network.Evaluate(sample.Key, values, derivatives);
				costs.Add(CostFunc.Cost(sample.Value, values[values.Count - 1]));
			}
			float totalCost = CostFunc.TotalCost(costs);


			//TODO: Do backpropagation.

			//TODO: Do gradient descent.

			return totalCost;
		}
	}
}
