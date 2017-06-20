using System;
using UnityEngine.Assertions;

namespace NeuralNet
{
	/// <summary>
	/// An n-dimensional vector.
	/// </summary>
	public class Vector
	{
		public int Count { get { return values.Length; } }
		public float this[int i]
		{
			get { return values[i]; }
			set { values[i] = value; }
		}

		private float[] values;


		public Vector(int nValues, float componentValue = 0.0f)
		{
			values = new float[nValues];
			for (int i = 0; i < values.Length; ++i)
				values[i] = componentValue;
		}
		public Vector(Vector a, Func<float, float> converter)
		{
			values = new float[a.Count];
			for (int i = 0; i < values.Length; ++i)
				values[i] = converter(a[i]);
		}
		public Vector(Vector a, Vector b, Func<float, float, float> operation)
		{
			Assert.AreEqual(a.Count, b.Count);
			values = new float[a.Count];
			for (int i = 0; i < values.Length; ++i)
				values[i] = operation(a[i], b[i]);
		}
		public Vector(Matrix lhs, Vector rhs)
		{
			Assert.AreEqual(lhs.NColumns, rhs.Count);

			values = new float[rhs.Count];
			for (int i = 0; i < Count; ++i)
			{
				float f = 0.0f;
				for (int row = 0; row < lhs.NRows; ++row)
					f += lhs[row, i] * rhs[i];
				values[i] = f;
			}
		}
		public Vector(Vector lhs, Matrix rhs)
		{
			Assert.AreEqual(lhs.Count, rhs.NRows);

			values = new float[lhs.Count];
			for (int i = 0; i < Count; ++i)
			{
				float f = 0.0f;
				for (int col = 0; col < rhs.NColumns; ++col)
					f += lhs[i] * rhs[i, col];
				values[i] = f;
			}
		}


		public static Vector operator +(Vector a, Vector b) { return new Vector(a, b, (_a, _b) => _a + _b); }
		public static Vector operator -(Vector a, Vector b) { return new Vector(a, b, (_a, _b) => _a - _b); }
		public static Vector operator *(Vector a, Vector b) { return new Vector(a, b, (_a, _b) => _a * _b); }
		public static Vector operator /(Vector a, Vector b) { return new Vector(a, b, (_a, _b) => _a / _b); }

		public static Vector operator -(Vector v) { return new Vector(v, _v => -_v); }


		public float Dot(Vector b)
		{
			Assert.AreEqual(Count, b.Count);
			float f = 0.0f;
			for (int i = 0; i < Count; ++i)
				f += values[i] * b[i];
			return f;
		}

		public override string ToString()
		{
			switch (Count)
			{
				case 1: return values[0].ToString();
				case 2: return "{" + values[0] + ", " + values[1] + "}";
				case 3: return "{" + values[0] + ", " + values[1] + ", " + values[2] + "}";

				default:
					return values.Length.ToString() + "D vector";
			}
		}
	}
}