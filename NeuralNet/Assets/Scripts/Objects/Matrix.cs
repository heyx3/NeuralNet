using System;
using UnityEngine.Assertions;


namespace NeuralNet
{
	/// <summary>
	/// An NxM matrix.
	/// </summary>
	public class Matrix
	{
		public int NRows { get { return values.GetLength(1); } }
		public int NColumns { get { return values.GetLength(0); } }
		public float this[int row, int column]
		{
			get { return values[column, row]; }
			set { values[column, row] = value; }
		}

		private float[,] values;


		public Matrix(int rows, int columns, float diagonal = 1.0f)
		{
			values = new float[columns, rows];
			for (int row = 0; row < rows; ++row)
				for (int col = 0; col < columns; ++col)
					this[row, col] = (row == col ? diagonal : 0.0f);
		}
		public Matrix(Matrix lhs, Matrix rhs)
		{
			Assert.AreEqual(lhs.NColumns, rhs.NRows);
			int nComponents = lhs.NColumns;

			values = new float[lhs.NRows, rhs.NColumns];
			for (int row = 0; row < NRows; ++row)
			{
				for (int col = 0; col < NColumns; ++col)
				{
					float f = 0.0f;
					for (int component = 0; component < nComponents; ++component)
						f += lhs[row, component] * rhs[component, col];
					this[row, col] = f;
				}
			}
		}


		public Matrix MakeTranspose()
		{
			Matrix m = new Matrix(NColumns, NRows);

			for (int row = 0; row < NRows; ++row)
				for (int col = 0; col < NColumns; ++col)
					m[col, row] = this[row, col];

			return m;
		}
	}
}
