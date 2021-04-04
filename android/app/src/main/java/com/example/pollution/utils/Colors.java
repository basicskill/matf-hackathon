package com.example.pollution.utils;

import android.graphics.Color;

import androidx.core.graphics.ColorUtils;

public class Colors {
    private static float clip(double x) {
        return (float) Math.pow(Math.min(Math.max(x, 0), 1), 0.6);
    }

    public static int fromAqi(double aqi) {
        int start = 0xa000a4d1;
        int end = 0xd018333b;
        return ColorUtils.blendARGB(start, end, clip(aqi / 500.0));
    }

    public static float pm10Alpha(double x) {
        return clip(x/430);
    }

    public static float pm25Alpha(double x) {
        return clip(x/250);
    }

    public static float no2Alpha(double x) {
        return clip(x/400);
    }

    public static float o3Alpha(double x) {
        return clip(x/750);
    }

    public static float coAlpha(double x) {
        return clip(x/35);
    }

    public static float so2Alpha(double x) {
        return clip(x/1600);
    }
}
