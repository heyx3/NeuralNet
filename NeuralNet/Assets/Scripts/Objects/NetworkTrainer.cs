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
		public IGradientDescent GradientDescent;

		/// <summary>
		/// Inputs paired with the output they're supposed to have.
		/// Training samples are used for training the network.
		/// Validation samples are used to verify the network isn't overtrained
		///     (sort of like "teaching to the test").
		/// </summary>
		public Dictionary<Vector, Vector> TrainingSamples, ValidationSamples;

		/// <summary>
		/// The number of epochs this trainer has run so far.
		/// </summary>
		public uint NEpochs { get; private set; }
		/// <summary>
		/// The number of iterations in the current epoch this trainer has run so far.
		/// </summary>
		public uint NIterations { get; private set; }


		public NetworkTrainer(NeuronNetwork network,
							  ICostFunc costFunc, IGradientDescent gradientDescent,
							  IEnumerable<KeyValuePair<Vector, Vector>> trainingSamples,
							  IEnumerable<KeyValuePair<Vector, Vector>> validationSamples)
		{
			Network = network;
			CostFunc = costFunc;
			GradientDescent = gradientDescent;
			NEpochs = 0;
			NIterations = 0;

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

			NIterations = 0;
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

				NIterations += 1;
			}

			NEpochs += 1;
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
			List<Vector> weightedInputs = new List<Vector>(),
						 outputs = new List<Vector>(),
						 derivatives = new List<Vector>();
			for (int layerI = 0; layerI < Network.Layers.Count; ++layerI)
			{
				weightedInputs.Add(new Vector(Network.Layers[layerI].NNodes));
				outputs.Add(new Vector(Network.Layers[layerI].NNodes));
				derivatives.Add(new Vector(Network.Layers[layerI].NNodes));
			}

			//For every sample, run the network, get its cost, then use "backpropagation"
			//    to get the derivatives of the cost with respect to all weights/biases.
			//Average these derivatives across all samples.
			List<float> costs = new List<float>(sampleBatch.Count);
			var biasDerivatives = new List<Vector>(Network.Layers.Count);
			var weightDerivatives = new List<Matrix>(Network.Layers.Count);
			//Initialize the derivatives to 0.
			for (int layerI = 0; layerI < Network.Layers.Count; ++layerI)
			{
				var layer = Network.Layers[layerI];
				biasDerivatives.Add(new Vector(layer.Biases.Count));
				weightDerivatives.Add(new Matrix(layer.Weights.NRows, layer.Weights.NColumns));
				for (int nodeI = 0; nodeI < layer.NNodes; ++nodeI)
				{
					biasDerivatives[layerI][nodeI] = 0.0f;

					int nPreviousNodes = Network.NNodesInPreviousLayer(layerI);
					for (int previousNodeI = 0; previousNodeI < nPreviousNodes; ++previousNodeI)
						weightDerivatives[layerI][nodeI, previousNodeI] = 0.0f;
				}
			}
			//Test all the samples.
			for (int sampleI = 0; sampleI < sampleBatch.Count; ++sampleI)
			{
				var sample = sampleBatch[sampleI];

				//Get the output of the network.
				Network.Evaluate(sample.Key, weightedInputs, outputs, derivatives);

				//Evaluate the cost of the output.
				float cost;
				Vector costDerivative = new Vector(outputs[outputs.Count - 1].Count);
				CostFunc.GetCost(sample.Value, outputs[outputs.Count - 1],
								 out cost, costDerivative);
				costs.Add(cost);

				//Do backpropagation to get the derivatives of the biases and weights.
				DoBackpropagation(sample.Key, weightedInputs, outputs, derivatives, costDerivative,
								  biasDerivatives, weightDerivatives);
			}

			//Get the average derivatives by dividing the sums by N.
			float invSize = 1.0f / sampleBatch.Count;
			for (int layerI = 0; layerI < Network.Layers.Count; ++layerI)
			{
				var layer = Network.Layers[layerI];
				for (int nodeI = 0; nodeI < layer.NNodes; ++nodeI)
				{
					biasDerivatives[layerI][nodeI] *= invSize;

					int nPreviousNodes = Network.NNodesInPreviousLayer(layerI);
					for (int previousNodeI = 0; previousNodeI < nPreviousNodes; ++previousNodeI)
						weightDerivatives[layerI][nodeI, previousNodeI] *= invSize;
				}
			}

			//Run gradient descent.
			GradientDescent.ModifyNetwork(Network, NIterations, NEpochs,
										  biasDerivatives, weightDerivatives);

			return costs.Sum() / (float)costs.Count;
		}
		/// <summary>
		/// Efficiently finds the derivative of the cost function with respect to
		///     every bias and weight in the network.
		/// </summary>
		/// <param name="networkInputs">
		/// The inputs into the network (a.k.a. the first layer of neuron outputs).
		/// </param>
		/// <param name="layerWeightedInputs">
		/// The weighted input into every node.
		/// </param>
		/// <param name="layerOutputs">
		/// The output of every node. Equal to the weighted input, fed into the activation function.
		/// </param>
		/// <param name="layerDerivatives">
		/// The derivative of every node's output, with respect to its weighted input.
		/// </param>
		/// <param name="costDerivatives">
		/// The derivatives of the cost function with respect to each output of the final layer.
		/// </param>
		/// <param name="out_BiasDerivatives">
		/// The calculated derivatives of the cost function with respect to each bias in the network.
		/// It is assumed that the list and its elements are already initialized to the proper size.
		/// The derivatives are added to the values already in there.
		/// </param>
		/// <param name="out_WeightDerivatives">
		/// The calculated derivatives of the cost function
		///     with respect to every weight between two nodes.
		/// It is assumed that the list and its elements are already initialized to the proper size.
		/// The derivatives are added to the values already in there.
		/// </param>
		private void DoBackpropagation(Vector networkInputs,
									   List<Vector> layerWeightedInputs,
									   List<Vector> layerOutputs,
									   List<Vector> layerDerivatives,
									   Vector costDerivatives,
									   List<Vector> out_BiasDerivatives,
									   List<Matrix> out_WeightDerivatives)
		{
			//First calculate "error", which is the derivative of the cost
			//    with respect to each node's weighted input.

			var errors = new List<Vector>(Network.Layers.Count);
			for (int layerI = 0; layerI < Network.Layers.Count; ++layerI)
				errors.Add(new Vector(Network.Layers[layerI].NNodes));

			//The simplest layer to calculate error for is the final "output" layer --
			//    the layer directly responsible for cost.
			var lastLayerError = errors[Network.Layers.Count - 1];
			var lastLayerDerivatives = layerDerivatives[Network.Layers.Count - 1];
			for (int nodeI = 0; nodeI < lastLayerError.Count; ++nodeI)
				lastLayerError[nodeI] = costDerivatives[nodeI] * lastLayerDerivatives[nodeI];

			//From there, we can work backward to find the errors of previous node layers
			//    using the weights from that layer into this one.
			//This is why it's called "backpropagation".
			for (int layerI = Network.Layers.Count - 2; layerI >= 0; --layerI)
			{
				var thisLayerErrors = errors[layerI];
				var thisLayerDerivatives = layerDerivatives[layerI];

				var nextLayerErrors = errors[layerI + 1];
				var nextLayerDerivatives = layerDerivatives[layerI + 1];

				//This isn't super intuitive, but we're basically
				//    applying the weights coming in from layerI to the errors from layerI + 1
				//    then multiplying with the derivatives of the activation function on layerI.
				var transposeNextWeights = Network.Layers[layerI + 1].Weights.MakeTranspose();
				var weightedNextErrors = new Vector(transposeNextWeights, nextLayerErrors);

				for (int nodeI = 0; nodeI < Network.Layers[layerI].NNodes; ++nodeI)
					thisLayerErrors[nodeI] = weightedNextErrors[nodeI] * thisLayerDerivatives[nodeI];
			}

			//The derivative of cost with respect to a node's bias is equal to the error.
			for (int layerI = 0; layerI < Network.Layers.Count; ++layerI)
				for (int componentI = 0; componentI < Network.Layers[layerI].NNodes; ++componentI)
					out_BiasDerivatives[layerI][componentI] += errors[layerI][componentI];

			//The derivative of cost with respect to a weight from one node into another
			//    is equal to the error times the input node's weighted input.
			for (int layerI = 0; layerI < Network.Layers.Count; ++layerI)
			{
				var layer = Network.Layers[layerI];
				Vector prevLayerOutput = (layerI == 0 ?
										      networkInputs :
										      layerOutputs[layerI - 1]);

				for (int nodeI = 0; nodeI < layer.NNodes; ++nodeI)
				{
					for (int previousNodeI = 0; previousNodeI < prevLayerOutput.Count; ++previousNodeI)
					{
						out_WeightDerivatives[layerI][nodeI, previousNodeI] +=
							errors[layerI][nodeI] *
							prevLayerOutput[nodeI];
					}
				}
			}
		}
	}
}
