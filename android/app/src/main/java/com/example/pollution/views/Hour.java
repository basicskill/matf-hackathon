package com.example.pollution.views;

import android.content.Context;
import android.util.AttributeSet;
import android.util.Log;
import android.util.TypedValue;
import android.view.View;
import android.widget.TextView;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.constraintlayout.widget.ConstraintLayout;
import androidx.constraintlayout.widget.Constraints;

import com.example.pollution.R;
import com.example.pollution.api.Prediction;
import com.example.pollution.utils.Colors;

public class Hour extends ConstraintLayout {
    public Hour(@NonNull Context context) {
        super(context);
    }

    public Hour(@NonNull Context context, @Nullable AttributeSet attrs) {
        super(context, attrs);
    }

    public Hour(@NonNull Context context, @Nullable AttributeSet attrs, int defStyleAttr) {
        super(context, attrs, defStyleAttr);
    }

    public Hour(@NonNull Context context, @Nullable AttributeSet attrs, int defStyleAttr, int defStyleRes) {
        super(context, attrs, defStyleAttr, defStyleRes);
    }

    public Hour(Context context, Prediction prediction) {
        super(context);

        inflate(context, R.layout.view_hour, this);

        //findViewById(R.id.view_bar).setBackgroundColor(Colors.fromAqi(prediction.getAqi()));

        ((TextView) findViewById(R.id.text_hour)).setText(prediction.getTime());
        ((TextView) findViewById(R.id.text_aqi_hour)).setText(String.valueOf(prediction.getAqi()));

        findViewById(R.id.text_pm10).setAlpha(Colors.pm10Alpha(prediction.getPm10()));
        findViewById(R.id.text_pm25).setAlpha(Colors.pm25Alpha(prediction.getPm25()));
        findViewById(R.id.text_no2).setAlpha(Colors.no2Alpha(prediction.getNo2()));
        findViewById(R.id.text_o3).setAlpha(Colors.o3Alpha(prediction.getO3()));
        findViewById(R.id.text_so2).setAlpha(Colors.so2Alpha(prediction.getSo2()));
        findViewById(R.id.text_co).setAlpha(Colors.coAlpha(prediction.getCo()));

        float pix = (float) (findViewById(R.id.constr_bar).getLayoutParams().height * (prediction.getAqi() / 500.0));

        ConstraintLayout.LayoutParams params = (ConstraintLayout.LayoutParams) findViewById(R.id.view_bar).getLayoutParams();
        params.height = (int) pix + 1;
        findViewById(R.id.view_bar).setLayoutParams(params);
    }
}
