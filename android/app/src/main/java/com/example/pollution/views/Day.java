package com.example.pollution.views;

import android.content.Context;
import android.os.Bundle;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.constraintlayout.widget.ConstraintLayout;
import androidx.fragment.app.Fragment;

import android.util.AttributeSet;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.LinearLayout;
import android.widget.TextView;

import com.example.pollution.R;
import com.example.pollution.api.Prediction;
import com.example.pollution.api.Predictions;
import com.example.pollution.utils.Colors;

import java.util.Arrays;

public class Day extends ConstraintLayout {
    private Predictions predictions;

    public Day(@NonNull Context context) {
        super(context);
    }

    public Day(@NonNull Context context, @Nullable AttributeSet attrs) {
        super(context, attrs);
    }

    public Day(@NonNull Context context, @Nullable AttributeSet attrs, int defStyleAttr) {
        super(context, attrs, defStyleAttr);
    }

    public Day(@NonNull Context context, @Nullable AttributeSet attrs, int defStyleAttr, int defStyleRes) {
        super(context, attrs, defStyleAttr, defStyleRes);
    }

    public Day(Context context, Prediction[] predictions) {
        super(context);

        inflate(context, R.layout.view_day, this);

        double sumAqi = 0;
        for (Prediction prediction : predictions) {
            sumAqi += prediction.getAqi();
        }
        double avgAqi = sumAqi / predictions.length;

        //int color = Colors.fromAqi(avgAqi);
        int color = Colors.fromAqi(predictions[0].getAqi());
        findViewById(R.id.day_root).setBackgroundColor(color);
        findViewById(R.id.scroll_hours).setBackgroundColor(color | 0xff<<24);

        ((TextView) findViewById(R.id.text_day)).setText(predictions[0].getDate());
        String emoji = "";
        if (avgAqi < 100) {
            emoji = "\uD83D\uDE42";
        }
        else if (avgAqi < 300) {
            emoji = "\uD83D\uDE10";
        }
        else {
            emoji = "\uD83D\uDE41";
        }
        ((TextView) findViewById(R.id.text_aqi)).setText(String.valueOf((int) avgAqi) + " " + emoji);

        LinearLayout hours = findViewById(R.id.linear_hours);
        for (Prediction prediction : predictions) {
            hours.addView(new Hour(context, prediction));
        }
    }
}