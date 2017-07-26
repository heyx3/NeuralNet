using System;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;

public static class MyGUI
{
	/// <summary>
	///Begins a tabbed area using GUILayout.
	/// </summary>
	/// <param name="space">
	/// The size of the tab.
	/// A negative value indicates a flexible space.
	/// A NaN value indicates no space.
	/// </param>
	public static void BeginTab(float space = -1.0f)
	{
		GUILayout.BeginHorizontal();

		if (!float.IsNaN(space))
			if (space < 0.0f)
				GUILayout.FlexibleSpace();
			else
				GUILayout.Space(space);

		GUILayout.BeginVertical();
	}
	/// <summary>
	/// Ends a tabbed area using GUILayout.
	/// </summary>
	/// <param name="space">
	/// The size of the tab at the end of the area.
	/// A negative value indicates a flexible space.
	/// A NaN value indicates no space.
	/// </param>
	public static void EndTab(float space = float.NaN)
	{
		GUILayout.EndVertical();

		if (!float.IsNaN(space))
			if (space < 0.0f)
				GUILayout.FlexibleSpace();
			else
				GUILayout.Space(space);

		GUILayout.EndHorizontal();
	}
}