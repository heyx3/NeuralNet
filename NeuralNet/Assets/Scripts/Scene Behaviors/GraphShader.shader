Shader "Graphing/Bar Graph"
{
	Properties
	{
		_MainTex ("Samples", 2D) = "white" {}

        _MinY ("Min Y Value", Float) = 0.0
        _MaxY ("Max Y Value", Float) = 1.0
        _MaxX ("Max UV x", Float) = 1.0

        _LineThickness ("Line Thickness", Float) = 0.05

        _LineColor ("Line Color", Color) = (1,1,1,1)
        _AboveColor ("Above line Color", Color) = (1,0,0,1)
        _BelowColor ("Below line Color", Color) = (0,1,0,1)
	}
	SubShader
	{
		Pass
		{
			CGPROGRAM
			#pragma vertex vert
			#pragma fragment frag

			#include "UnityCG.cginc"

			struct appdata
			{
				float4 vertex : POSITION;
				float2 uv : TEXCOORD0;
			};

			struct v2f
			{
				float2 uv : TEXCOORD0;
				float4 vertex : SV_POSITION;
			};

			v2f vert (appdata v)
			{
				v2f o;
				o.vertex = UnityObjectToClipPos(v.vertex);
				o.uv = v.uv;
				return o;
			}

			sampler2D _MainTex;
            float4 _MainTex_TexelSize;

            float _MinY, _MaxY, _MaxX;

            float _LineThickness;
            float4 _LineColor, _AboveColor, _BelowColor;

			fixed4 frag (v2f i) : SV_Target
			{
                i.uv.x = lerp(0.0f, _MaxX, i.uv.x);
                i.uv.y = lerp(_MinY, _MaxY, i.uv.y);

                float sample = tex2D(_MainTex, i.uv).x;

                float distAboveLine = i.uv.y - sample;
                float isOnLine = step(_LineThickness, abs(distAboveLine));

                float4 graphColor = lerp(_BelowColor, _AboveColor, step(0.0, distAboveLine));

                return lerp(_LineColor, graphColor, isOnLine);
			}
			ENDCG
		}
	}
}
