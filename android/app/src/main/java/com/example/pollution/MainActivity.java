package com.example.pollution;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.recyclerview.widget.RecyclerView;
import androidx.viewpager2.widget.ViewPager2;

import android.annotation.SuppressLint;
import android.os.Bundle;
import android.view.View;
import android.view.ViewGroup;
import android.webkit.WebView;
import android.widget.LinearLayout;
import android.widget.TextView;

import com.example.pollution.api.Api;
import com.example.pollution.api.Prediction;
import com.example.pollution.api.Predictions;
import com.example.pollution.utils.Colors;
import com.example.pollution.views.Day;
import com.example.pollution.views.Hour;

public class MainActivity extends AppCompatActivity {

    @SuppressLint("SetJavaScriptEnabled")
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        setTitle("Beograd");

        Api.getPredictions((Predictions response) -> {
            int height = findViewById(R.id.scroll_days).getMeasuredHeight();
            int width = findViewById(R.id.scroll_days).getMeasuredWidth();
            //ViewPager2 pager = findViewById(R.id.pager);
            //pager.setAdapter(new PagerAdapter(response));

            LinearLayout days = findViewById(R.id.linear_days);
            for (Prediction[] predictions : response.getPredictions()) {
                Day day = new Day(this, predictions);
                days.addView(day);
                ViewGroup.LayoutParams layoutParams = day.getLayoutParams();
                layoutParams.height = height;
                layoutParams.width = width;
                day.setLayoutParams(layoutParams);
            }
        }, null);

        /*WebView wv = findViewById(R.id.wv);
        wv.getSettings().setJavaScriptEnabled(true);
        wv.loadUrl("https://pollution-prediction.herokuapp.com/");*/
    }

    /*static class PagerAdapter extends RecyclerView.Adapter {
        private Predictions predictions;

        static class ViewHolder extends RecyclerView.ViewHolder {

            public ViewHolder(@NonNull View itemView) {
                super(itemView);
            }
        }

        public PagerAdapter(Predictions predictions) {
            this.predictions = predictions;
        }

        @NonNull
        @Override
        public RecyclerView.ViewHolder onCreateViewHolder(@NonNull ViewGroup parent, int viewType) {

            return new ViewHolder(new Hour(parent.getContext()));
        }

        @Override
        public void onBindViewHolder(@NonNull RecyclerView.ViewHolder holder, int position) {
            holder.itemView = new Day(holder.itemView.getContext(), predictions.getPredictions()[position]);
        }

        @Override
        public int getItemCount() {
            return predictions.getPredictions().length;
        }
    }*/

}