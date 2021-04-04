package com.example.pollution.api;

import android.content.Context;

import com.android.volley.Request;
import com.android.volley.RequestQueue;
import com.android.volley.Response;
import com.android.volley.toolbox.StringRequest;
import com.android.volley.toolbox.Volley;
import com.google.gson.Gson;

public class Api {
    private static final String HOST = "https://pollution-prediction.herokuapp.com/";

    private static RequestQueue queue;

    public static void init(Context context, Runnable okCallback, Response.ErrorListener errorListener) {
        queue = Volley.newRequestQueue(context);

        StringRequest request = new StringRequest(Request.Method.GET, HOST, response -> {okCallback.run();}, errorListener);
        queue.add(request);
    }

    public static void getPredictions(Response.Listener<Predictions> responseListener, Response.ErrorListener errorListener) {
        String endpoint = "api/predictions";

        String url = HOST + endpoint;

        StringRequest request = new StringRequest(Request.Method.GET, url, (String response) -> {
                    Predictions r = new Gson().fromJson(response, Predictions.class);
                    responseListener.onResponse(r);
                }, errorListener);

        queue.add(request);
    }
}
