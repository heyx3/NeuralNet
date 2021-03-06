﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.IO;

using Logging = UnityEngine.Debug;


namespace NeuralNet
{
	public class HandwritingData
	{
		//Source for data and file format: http://yann.lecun.com/exdb/mnist/

		public static readonly string File_TrainingInputs = "train-images.idx3-ubyte",
									  File_TrainingOutputs = "train-labels.idx1-ubyte",
									  File_ValidationInputs = "t10k-images.idx3-ubyte",
									  File_ValidationOutputs = "t10k-labels.idx1-ubyte";


		public enum Digits : byte
		{
			Zero = 0,
			One, Two, Three,
			Four, Five, Six,
			Seven, Eight, Nine
		}

		public struct Image
		{
			public float[,] Pixels;
			public Digits Digit;
			public Image(int width, int height, Digits digit)
			{
				Pixels = new float[width, height];
				Digit = digit;
			}
		}

		public Image[] TrainingImages, ValidationImages;

		public int PixelWidth
		{
			get
			{
				if (TrainingImages.Length > 0)
					return TrainingImages[0].Pixels.GetLength(0);
				else if (ValidationImages.Length > 0)
					return ValidationImages[0].Pixels.GetLength(0);
				else
					throw new Exception("No images exist");
			}
		}
		public int PixelHeight
		{
			get
			{
				if (TrainingImages.Length > 0)
					return TrainingImages[0].Pixels.GetLength(1);
				else if (ValidationImages.Length > 0)
					return ValidationImages[0].Pixels.GetLength(1);
				else
					throw new Exception("No images exist");
			}
		}


		/// <summary>
		/// Tries to load the handwriting data files from the given directory.
		/// </summary>
		/// <param name="outErrMsg">
		/// Outputs an error message, or the empty string if there was no error.
		/// </param>
		public HandwritingData(string directory, out string outErrMsg)
		{
			outErrMsg = ReadSamples(Path.Combine(directory, File_TrainingInputs),
									Path.Combine(directory, File_TrainingOutputs),
									out TrainingImages);
			if (outErrMsg.Length > 0)
			{
				outErrMsg = "Error with training files: " + outErrMsg;
				return;
			}

			outErrMsg = ReadSamples(Path.Combine(directory, File_ValidationInputs),
									Path.Combine(directory, File_ValidationOutputs),
									out ValidationImages);
			if (outErrMsg.Length > 0)
				outErrMsg = "Error with validation files: " + outErrMsg;
		}
		private string ReadSamples(string inputsPath, string outputsPath, out Image[] outArray)
		{
			outArray = null;

			try
			{
				using (var stream_inputs = new MemoryStream(File.ReadAllBytes(inputsPath)))
				using (var stream_outputs = new MemoryStream(File.ReadAllBytes(outputsPath)))
				using (var inputs = new BinaryReader(stream_inputs))
				using (var outputs = new BinaryReader(stream_outputs))
				{
					int inputMagicNumber = inputs.ReadInt32_BE(),
						outputMagicNumber = outputs.ReadInt32_BE();
					if (inputMagicNumber != 2051)
						return "Input magic number isn't 2051";
					if (outputMagicNumber != 2049)
						return "Output magic number isn't 2049";

					int nInputs = inputs.ReadInt32_BE(),
						nOutputs = outputs.ReadInt32_BE();
					if (nInputs != nOutputs)
						return nInputs.ToString() + " Inputs != " + nOutputs + " Outputs";

					outArray = new Image[nInputs];
					int sizeY = inputs.ReadInt32_BE(),
						sizeX = inputs.ReadInt32_BE();
					for (int i = 0; i < outArray.Length; ++i)
					{
						outArray[i] = new Image(sizeX, sizeY, (Digits)outputs.ReadByte());
						for (int y = 0; y < sizeY; ++y)
							for (int x = 0; x < sizeX; ++x)
								outArray[i].Pixels[x, y] = inputs.ReadByte() / 255.0f;
					}

					return "";
				}
			}
			catch (Exception e)
			{
				return "Exception: " + e.Message + "\n" + e.StackTrace;
			}
		}
	}
}
