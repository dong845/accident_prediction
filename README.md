# accident_prediction

## accident_prediciton.ipynb (explore the relationship between static geographical features and accident)
- node features: [lat, lng, street_count, crossing, junction, railway, station] (total length is 7), mainly some geographical features around node
- edge attribute: [length, bridge, tunnel, lanes, oneway, maxspeed] (total length is 6), mainly some information about edge (factual road)
- label: 0 or 1, 0 is generated from some negative samples


## accident_prediction_time.ipynb (explore the relationship between time and accident)
- node features: [hour, month, day] (total length is 3), mainly some related time information about node
- edge attribute: [length, bridge, tunnel, lanes, oneway, maxspeed] (total length is 6), mainly some information about edge (factual road)
- label: 0 or 1, 0 is generated from some negative samples

- dataset: https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents
