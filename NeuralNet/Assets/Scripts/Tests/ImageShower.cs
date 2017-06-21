using System;
using System.Collections.Generic;
using System.Linq;
using System.IO;
using UnityEngine;


namespace Tests
{
	public class ImageShower : MonoBehaviour
	{
		private NeuralNet.HandwritingData data;

		private bool showValidationImages = false;
		private Texture2D currentImage;
		private int imageIndex = 0;


		private void Start()
		{
			data = new NeuralNet.HandwritingData(Path.Combine(Application.streamingAssetsPath,
															  "NeuralNet Data"));

			currentImage = new Texture2D(data.PixelWidth, data.PixelHeight, TextureFormat.ARGB32,
										 false, true);
			currentImage.filterMode = FilterMode.Point;
			currentImage.wrapMode = TextureWrapMode.Clamp;

			if (data.TrainingImages.Length > 0)
				UpdateImage();
		}
		private void OnGUI()
		{
			var images = (showValidationImages ?
							 data.ValidationImages :
							 data.TrainingImages);

			GUI.DrawTexture(new Rect(50.0f, 0.0f, 200.0f, 200.0f), currentImage);
			if (GUI.Button(new Rect(0.0f, 250.0f, 25.0f, 15.0f), "<"))
			{
				imageIndex = (imageIndex == 0 ?
								 images.Length - 1 :
								 imageIndex - 1);
				UpdateImage();
			}
			GUI.Label(new Rect(25.0f, 250.0f, 50.0f, 30.0f), imageIndex.ToString() + " - " + images[imageIndex].Digit.ToString());
			if (GUI.Button(new Rect(80.0f, 250.0f, 25.0f, 15.0f), ">"))
			{
				imageIndex = (imageIndex + 1) % images.Length;
				UpdateImage();
			}
			if (GUI.Button(new Rect(0.0f, 280.0f, 50.0f, 30.0f),
						   showValidationImages ? "Validation Images" : "Training Images"))
			{
				showValidationImages = !showValidationImages;
				images = (showValidationImages ?
					         data.ValidationImages :
							 data.TrainingImages);

				if (imageIndex >= images.Length)
				{
					imageIndex %= images.Length;
				}
				UpdateImage();
			}
		}

		private Color32[] colors = null;
		private void UpdateImage()
		{
			if (colors == null)
				colors = new Color32[currentImage.width * currentImage.height];

			var image = (showValidationImages ?
							 data.ValidationImages[imageIndex] :
							 data.TrainingImages[imageIndex]);
			for (int y = 0; y < currentImage.height; ++y)
				for (int x = 0; x < currentImage.width; ++x)
				{
					byte p = (byte)(255.0f * Mathf.Clamp01(image.Pixels[x, currentImage.height - y - 1]));
					colors[x + (y * currentImage.width)] = new Color32(p, p, p, 255);
				}

			currentImage.SetPixels32(colors);
			currentImage.Apply(true, false);
		}
	}
}