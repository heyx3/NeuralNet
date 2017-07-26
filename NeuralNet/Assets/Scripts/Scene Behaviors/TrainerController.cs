using System;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using NeuralNet;

namespace Tests
{
	/// <summary>
	/// Provides a GUI for creating/training a neural net to recognize handwritten images.
	/// </summary>
	public class TrainerController : MonoBehaviour
	{
		#region Func types <=> string

		private static readonly string[] costFuncs = new string[]
			{ "Quadratic" };
		private static int getIndex(ICostFunc func)
		{
			if (func is CostFunc_Quadratic)
				return 0;
			throw new NotImplementedException(func.GetType().FullName);
		}
		private static ICostFunc makeCostFunc(int index)
		{
			switch (index)
			{
				case 0: return new CostFunc_Quadratic();
				default: throw new NotImplementedException(index.ToString());
			}
		}

		private static readonly string[] activationFuncs = new string[]
			{ "Logistic" };
		private static int getIndex(IActivationFunc func)
		{
			if (func is ActivationFunc_Logistic)
				return 0;
			throw new NotImplementedException(func.GetType().FullName);
		}
		private static IActivationFunc makeActivationFunc(int index)
		{
			switch (index)
			{
				case 0: return new ActivationFunc_Logistic();
				default: throw new NotImplementedException(index.ToString());
			}
		}

		private static readonly string[] gradientDescentModes = new string[]
			{ "Constant" };
		private static int getIndex(IGradientDescent gradientDescent)
		{
			if (gradientDescent is GradientDescent_Constant)
				return 0;
			throw new NotImplementedException(gradientDescent.GetType().FullName);
		}
		private static IGradientDescent makeGradientDescent(int index)
		{
			switch (index)
			{
				case 0: return new GradientDescent_Constant(0.01f);
				default: throw new NotImplementedException(index.ToString());
			}
		}

		#endregion


		/// <summary>
		/// The sizes of the hidden layers of the neural network
		///     (i.e. not including the input/output layers).
		/// </summary>
		public List<int> HiddenLayerSizes = new List<int>() { 20, 20 };
		/// <summary>
		/// The activation function for each layer of the neural network
		///     (including the output, but not the input).
		/// </summary>
		public List<IActivationFunc> LayerActivationFuncs = new List<IActivationFunc>()
			{ new ActivationFunc_Logistic(), new ActivationFunc_Logistic(), new ActivationFunc_Logistic() };
		/// <summary>
		/// Affects the RNG used during training.
		/// </summary>
		public int Seed = 12345;

		/// <summary>
		/// The material/shader used for graphing a set of points stored in a Nx1 texture.
		/// </summary>
		public Material GraphRenderMat;
		private void Graph(Rect graphScreenPos, Texture2D graphData, int nGraphPoints,
						   float min, float max)
		{
			GraphRenderMat.mainTexture = graphData;
			GraphRenderMat.SetFloat("_MinY", min);
			GraphRenderMat.SetFloat("_MaxY", max);
			GraphRenderMat.SetFloat("_MaxX", nGraphPoints / (float)graphData.width);

			Graphics.DrawTexture(graphScreenPos, graphData, GraphRenderMat);
		}


		/// <summary>
		/// The trainer/network.
		/// </summary>
		private NetworkTrainer trainer;

		/// <summary>
		/// A Nx1 texture containing the cost of each run of the neural network in chronological order.
		/// The texture will generally be longer than the actual number of samples.
		/// </summary>
		private Texture2D runCosts = null;
		/// <summary>
		/// The number of actual values in the "runCosts" texture.
		/// </summary>
		private int nRunCosts = 0;
		/// <summary>
		/// The min/max cost found in the current history of network runs.
		/// </summary>
		private float minCost = float.PositiveInfinity,
					  maxCost = float.NegativeInfinity;
		/// <summary>
		/// The mean/standard deviation of the Gaussian ValueInitializer.
		/// </summary>
		private float valueInit_Gaussian_Mean = 0.0f,
					  valueInit_Gaussian_StdDev = 1.0f;


		private void ResetSamples()
		{
			nRunCosts = 0;
			minCost = float.PositiveInfinity;
			maxCost = float.NegativeInfinity;
		}
		private void AddSample(float cost)
		{
			//If the texture doesn't exist yet, create it.
			if (runCosts == null)
			{
				runCosts = new Texture2D(128, 1, TextureFormat.RFloat, false, true);
				runCosts.filterMode = FilterMode.Bilinear;
				runCosts.wrapMode = TextureWrapMode.Clamp;
			}

			UnityEngine.Assertions.Assert.IsTrue(nRunCosts <= runCosts.width);

			//If the texture has no more room, expand it.
			if (nRunCosts == runCosts.width)
			{
				Color[] oldPixels = runCosts.GetPixels();

				//Double the size of the texture.
				int newWidth = runCosts.width * 2;
				Color[] newPixels = new Color[newWidth * runCosts.height];
				for (int x = 0; x < newWidth; ++x)
				{
					for (int y = 0; y < runCosts.height; ++y)
					{
						int newI = x + (y * newWidth);
						if (x < runCosts.width)
							newPixels[newI] = oldPixels[x + (y * runCosts.width)];
						else
							newPixels[newI] = new Color(0, 0, 0, 0);
					}
				}

				//Apply the resizing to the actual texture.
				runCosts.Resize(newWidth, runCosts.height, runCosts.format, false);
				runCosts.SetPixels(newPixels);
				runCosts.Apply(false, false);
			}

			//Add the value to the end of the texture.
			runCosts.SetPixel(nRunCosts, 0, new Color(cost, 0.0f, 0.0f, 0.0f));
			runCosts.Apply(false, false);
			nRunCosts += 1;
			minCost = Math.Min(minCost, cost);
			maxCost = Math.Max(maxCost, cost);
		}

		private void Awake()
		{
			string errMsg;
			var trainingData = new HandwritingData(System.IO.Path.Combine(Application.streamingAssetsPath,
																		  "NeuralNet Data"),
												   out errMsg);
			if (errMsg.Length > 0)
			{
				Debug.LogError(errMsg);
				return;
			}

			//Convert the data to input/output vectors for the network.
			Func<HandwritingData.Image, KeyValuePair<Vector, Vector>> converter =
				(img) =>
				{
					Vector input = new Vector(img.Pixels.GetLength(0) * img.Pixels.GetLength(1));
					for (int y = 0; y < img.Pixels.GetLength(1); ++y)
						for (int x = 0; x < img.Pixels.GetLength(0); ++x)
							input[x + (y * img.Pixels.GetLength(0))] = img.Pixels[x, y];

					Vector output = new Vector(10);
					output[(int)img.Digit] = 1.0f;

					return new KeyValuePair<Vector, Vector>(input, output);
				};
			//Create the array of layer sizes, including input/output layers.
			int[] layerSizes = new int[HiddenLayerSizes.Count + 1];
			layerSizes[0] = trainingData.PixelWidth * trainingData.PixelHeight;
			layerSizes[layerSizes.Length - 1] = 10;
			for (int i = 1; i < layerSizes.Length - 1; ++i)
				layerSizes[i] = HiddenLayerSizes[i - 1];
			//Create the network/trainer.
			trainer = new NetworkTrainer(new NeuronNetwork(new System.Random(Seed),
														   new ActivationFunc_Logistic(),
														   new ValueInitializer_Gaussian(),
														   HiddenLayerSizes.ToArray()),
										 new CostFunc_Quadratic(),
										 new GradientDescent_Constant(1.0f),
										 trainingData.TrainingImages.Select(converter),
										 trainingData.ValidationImages.Select(converter));
		}
		private void OnGUI()
		{
			GUILayout.BeginArea(new Rect(0.0f, 0.0f, Screen.width, Screen.height));

			//Seed.
			GUILayout.BeginHorizontal();
			GUILayout.Label("Seed:");
			GUILayout.Label(Seed.ToString());
			if (GUILayout.Button("Generate new seed"))
				Seed = UnityEngine.Random.Range(0, 99999);
			GUILayout.EndHorizontal();

			GUILayout.Space(35.0f);

			GUILayout.Label("Hidden Layers:");
			MyGUI.BeginTab(15.0f);
			{
				bool layersChanged = false;
				for (int i = 0; i < trainer.Network.Layers.Count - 1; ++i)
				{
					GUILayout.BeginHorizontal();
					{
						uint u;
						if (uint.TryParse(GUILayout.TextField(trainer.Network.Layers[i].NNodes.ToString(),
															  GUILayout.MinWidth(50.0f)),
										  out u) &&
							u != trainer.Network.Layers[i].NNodes)
						{
							trainer.Network.Layers[i].Resize((int)u,
															 (i > 0) ?
																 trainer.Network.Layers[i - 1].NNodes :
															     trainer.Network.NInputNodes);
							layersChanged = true;
						}

						int currentFuncI = getIndex(trainer.Network.Layers[i].ActivationFunc);
						int nextFuncI = GUILayout.SelectionGrid(currentFuncI, activationFuncs,
																activationFuncs.Length);
						if (currentFuncI != nextFuncI)
						{
							trainer.Network.Layers[i].ActivationFunc = makeActivationFunc(nextFuncI);
							layersChanged = true;
						}

						GUILayout.FlexibleSpace();
						if (GUILayout.Button("x"))
						{
							trainer.Network.Layers.RemoveAt(i);
							i -= 1;
							layersChanged = true;

							if (i >= 0)
							{
								trainer.Network.Layers[i].Resize(trainer.Network.Layers[i].NNodes,
																 (i > 0) ?
																	 trainer.Network.Layers[i - 1].NNodes :
																	 trainer.Network.NInputNodes);
							}
						}
						GUILayout.FlexibleSpace();
					}
					GUILayout.EndHorizontal();
				}
				GUILayout.BeginHorizontal();
				if (GUILayout.Button("+"))
				{
					layersChanged = true;

					var prevLayer = (trainer.Network.Layers.Count == 1) ?
										null :
										trainer.Network.Layers[trainer.Network.Layers.Count - 2];
					Matrix weights = new Matrix((prevLayer == null ? 20 : prevLayer.NNodes),
												(prevLayer == null ?
												     trainer.Network.NInputNodes :
													 prevLayer.NNodes));
					Vector biases = new Vector(weights.NRows);
					for (int i = 0; i < biases.Count; ++i)
					{
						biases[i] = 0.0f;
						for (int j = 0; j < weights.NColumns; ++j)
							weights[i, j] = 0.0f;
					}

					trainer.Network.Layers.Insert(
						trainer.Network.Layers.Count - 1,
						new NeuronLayer(weights, biases,
										(prevLayer == null ?
											 trainer.Network.Layers[trainer.Network.Layers.Count - 1].ActivationFunc :
											 prevLayer.ActivationFunc)));
				}
				GUILayout.FlexibleSpace();
				GUILayout.EndHorizontal();
				if (layersChanged)
					ResetSamples();
			}
			MyGUI.EndTab();

			GUILayout.Space(35.0f);

			//Edit network parameters.
			bool changed = false;
			int currentI, nextI;
			//Cost function.
			GUILayout.BeginHorizontal();
			GUILayout.Label("Cost Func:");
			currentI = getIndex(trainer.CostFunc);
			nextI = GUILayout.SelectionGrid(currentI, costFuncs, costFuncs.Length);
			if (currentI != nextI)
			{
				trainer.CostFunc = makeCostFunc(nextI);
				changed = true;
			}
			GUILayout.EndHorizontal();
			//Gradient descent.
			GUILayout.BeginHorizontal();
			GUILayout.Label("Gradient Descent:");
			currentI = getIndex(trainer.GradientDescent);
			nextI = GUILayout.SelectionGrid(currentI, gradientDescentModes, gradientDescentModes.Length);
			if (currentI != nextI)
			{
				trainer.GradientDescent = makeGradientDescent(nextI);
				changed = true;
			}
			GUILayout.EndHorizontal();
			if (changed)
				ResetSamples();

			GUILayout.Space(15.0f);

			//Value initializer.
			GUILayout.BeginHorizontal();
			GUILayout.Label("Reset network:");
			bool reset = false;
			GUILayout.BeginVertical();
				if (GUILayout.Button("Gaussian"))
				{
					reset = true;
					trainer.Network.Reset(new System.Random(Seed),
										  new ValueInitializer_Gaussian(valueInit_Gaussian_Mean,
																		valueInit_Gaussian_StdDev));
				}
				float f;
				GUILayout.BeginHorizontal();
					GUILayout.Label("Mean:");
					if (float.TryParse(GUILayout.TextField(valueInit_Gaussian_Mean.ToString()), out f))
						valueInit_Gaussian_Mean = f;
				GUILayout.EndHorizontal();
				GUILayout.BeginHorizontal();
					GUILayout.Label("Std Dev:");
					if (float.TryParse(GUILayout.TextField(valueInit_Gaussian_StdDev.ToString()), out f))
						valueInit_Gaussian_StdDev = f;
				GUILayout.EndHorizontal();
			GUILayout.EndVertical();
			GUILayout.EndHorizontal();
			if (reset)
				ResetSamples();

			GUILayout.Space(35.0f);

			//Graph the network's cost over time.
			if (nRunCosts > 0)
			{
				GUILayout.Label("Network quality over time:");

				if (Event.current.type == EventType.Repaint)
				{
					Graph(GUILayoutUtility.GetRect(350.0f, 100.0f),
						  runCosts, nRunCosts, minCost, maxCost);
				}

				GUILayout.Space(35.0f);
			}

			//TODO: Button to run epochs.

			GUILayout.EndArea();
		}
	}
}
