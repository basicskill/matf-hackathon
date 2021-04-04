import numpy as np

def calculate_aqi(co, no2, o3, pm10, p25, so2):
    
    boundaries = {
    "pm10" : [0, 50, 100, 250, 350, 430],
    "p25" : [0, 30, 60, 90, 120, 250],
    "no2" : [0, 40, 80, 180, 280, 400],
    "o3" : [0, 50, 100, 168, 208, 748],
    "co" : [0, 1, 2, 10, 17, 34],
    "so2" : [0, 40, 80, 380, 800, 1600],
    "aqi" : [0, 50, 100, 200, 300, 400, 500]
    }
    data = {
        "co" : co,
        "no2" : no2,
        "o3" : o3,
        "pm10" : pm10,
        "p25" : p25,
        "so2" : so2
    }
    pollusions = ["co", "no2", "o3", "pm10", "p25", "so2"]

    aqi_max = 0
    for pol in pollusions:
        j = len(boundaries[pol])-1
        if data[pol] == -1:
            continue
        while data[pol] < boundaries[pol][j]:
            j-=1
        low_index = j
        high_index = j+1
        if high_index<=boundaries[pol]:
            delta_boundaries = boundaries[pol][high_index] - boundaries[pol][low_index]
        else:
            delta_boundaries = boundaries[pol][low_index] - boundaries[pol][low_index-1]
        aqi = ((boundaries["aqi"][high_index] - boundaries["aqi"][low_index]) / \
                delta_boundaries) * \
                (data[pol] - boundaries[pol][low_index]) + boundaries["aqi"][low_index]
        if aqi > aqi_max:
            aqi_max = aqi
    return aqi_max

def main():
    print(calculate_aqi(0.590997442, -1, -1, -1, -1, -1))

if __name__ == "__main__":
    main()