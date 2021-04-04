package com.example.pollution.api;

import androidx.annotation.NonNull;

import java.io.Serializable;
import java.util.Arrays;

public class Predictions implements Serializable {
    private Prediction[][] predictions;

    public Prediction[][] getPredictions() {
        return predictions;
    }

    public void setPredictions(Prediction[][] predictions) {
        this.predictions = predictions;
    }

    @NonNull
    @Override
    public String toString() {
        return Arrays.toString(predictions);
    }
}
