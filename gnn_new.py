import pandas as pd
import numpy as np
import osmnx as ox
import torch.nn as nn
from torch.nn import Linear
import torch.nn.functional as F
import math
import torch
from torch_geometric.data import Data
from torch.optim import Adam, AdamW
from torch_geometric.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
from torch_geometric.nn import GATConv, GATv2Conv, TransformerConv
from collections import Counter
import random

random.seed(0)

class GNNModel(nn.Module):
    def __init__(self, classes):
        super(GNNModel, self).__init__()
        self.conv1 = GATv2Conv(6, 64, edge_dim=6)
        self.conv2 = GATv2Conv(64, 64, edge_dim=6)
        self.conv3 = GATv2Conv(64, 64, edge_dim=6)
        self.linear = Linear(64, 32)
        self.linear1 = Linear(32, classes)
    
    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = self.conv1(x, edge_index, edge_attr)
        x = F.relu(x)
        x = F.dropout(x, 0.4)
        x = self.conv2(x, edge_index, edge_attr)
        x = F.relu(x)
        x = F.dropout(x, 0.4)
        x = self.conv3(x, edge_index, edge_attr)
        x = F.relu(x)
        x = F.dropout(x, 0.4)
        x = F.relu(self.linear(x))
        x = F.dropout(x, 0.4)
        return self.linear1(x)

def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians 
    lon1 = math.radians(lon1)
    lat1 = math.radians(lat1)
    lon2 = np.radians(lon2)
    lat2 = np.radians(lat2)
    # lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a)) 
    r = 6378 # Radius of earth in kilometers
    return c * r

G = ox.graph_from_place('Los Angeles, California', network_type='drive')

data_path = "US_Accidents_Dec21_updated.csv"
data = pd.read_csv(data_path)
kaggle_LA = data[data["City"]=="Los Angeles"]
print(kaggle_LA.shape)
kaggle_LA['Start_Time'] =  pd.to_datetime(kaggle_LA['Start_Time'], format='%Y-%m-%d %H:%M:%S')
kaggle_LA['Year'] = pd.DatetimeIndex(kaggle_LA['Start_Time']).year
kaggle_LA_2020 = kaggle_LA[kaggle_LA["Year"]==2020]
print(kaggle_LA_2020.shape)

def addNegativeSamples(df):
    global balanced_kaggle_california_df
    #Specifying raw dataframe
    raw_df = df.copy()
    raw_df["New_Distance"] = df["Distance(mi)"]
    raw_df["Accident"] = 1

    #Specifying upper coordinate dataframe
    negative_upper_coordinate_df = df.copy()
    negative_upper_coordinate_df["New_Distance"] = negative_upper_coordinate_df["Distance(mi)"].apply(lambda x: x*random.uniform(0.0007,0.0009))
    negative_upper_coordinate_df["Accident"] = 0

    #Specifying lower coordinate dataframe
    negative_lower_coordinate_df = df.copy()
    negative_lower_coordinate_df["New_Distance"] = negative_lower_coordinate_df["Distance(mi)"].apply(lambda x: x*random.uniform(0.0007,0.0009))
    negative_lower_coordinate_df["Accident"] = 0
    

    #Offset "LONGITUD" and "LATITUDE" with +/- ~100 meters
    # negative_upper_coordinate_df["Start_Lng"] = negative_upper_coordinate_df["Start_Lng"].apply(lambda x: x-random.uniform((x/750), (x/300)))
    # negative_upper_coordinate_df["Start_Lat"] = negative_upper_coordinate_df["Start_Lat"].apply(lambda x: x-random.uniform((x/750), (x/300)))
    negative_upper_coordinate_df["Start_Lng"] = negative_upper_coordinate_df["Start_Lng"].apply(lambda x: x-(x/750))
    negative_upper_coordinate_df["Start_Lat"] = negative_upper_coordinate_df["Start_Lat"].apply(lambda x: x-(x/750))


    # negative_lower_coordinate_df["Start_Lng"] = negative_lower_coordinate_df["Start_Lng"].apply(lambda x: x+random.uniform((x/750), (x/300)))
    # negative_lower_coordinate_df["Start_Lat"] = negative_lower_coordinate_df["Start_Lat"].apply(lambda x: x+random.uniform((x/750), (x/300)))   
    negative_lower_coordinate_df["Start_Lng"] = negative_lower_coordinate_df["Start_Lng"].apply(lambda x: x+(x/750))
    negative_lower_coordinate_df["Start_Lat"] = negative_lower_coordinate_df["Start_Lat"].apply(lambda x: x+(x/750))   

    balanced_kaggle_california_df = pd.concat([raw_df, negative_lower_coordinate_df, negative_upper_coordinate_df],ignore_index=True)
    return balanced_kaggle_california_df

kaggle_LA_2020 = addNegativeSamples(kaggle_LA_2020)

kaggle_LA_2020["Lat"] = (kaggle_LA_2020["Start_Lat"]+kaggle_LA_2020["End_Lat"])/2
kaggle_LA_2020["Lng"] = (kaggle_LA_2020["Start_Lng"]+kaggle_LA_2020["End_Lng"])/2
# extract needed features and label
kaggle_LA_2020_new = kaggle_LA_2020[["Lat", "Lng", "New_Distance", "Crossing", "Junction", "Railway", "Station", "Accident"]]

nodes_list = sorted(list(G.nodes))
nodes_num = len(nodes_list)
print("nodes_num", nodes_num)
feature_matrix = np.zeros((nodes_num,8))
labels = []
times = 0
num_1s = 0
num_0s = 0
for node in nodes_list:
    lats = G.nodes[node]["y"]
    lngs = G.nodes[node]["x"]
    if "street_count" in G.nodes[node]:
        street_count = G.nodes[node]["street_count"]
    else:
        street_count = 0
    dist_min = float("inf")
    lat_accident = np.array(kaggle_LA_2020_new["Lat"])
    lng_accident = np.array(kaggle_LA_2020_new["Lng"])
    dist = haversine(lngs, lats, lng_accident, lat_accident)
    dist_index = np.argmin(dist)
    dist_tmp = np.min(dist)
    if dist_tmp<dist_min:
        dist_min = dist_tmp
        distance = float(kaggle_LA_2020_new["New_Distance"].iloc[dist_index])
        crossing = float(int(kaggle_LA_2020_new["Crossing"].iloc[dist_index]))
        junction = float(int(kaggle_LA_2020_new["Junction"].iloc[dist_index]))
        railway = float(int(kaggle_LA_2020_new["Railway"].iloc[dist_index]))
        station = float(int(kaggle_LA_2020_new["Station"].iloc[dist_index]))
    feature_matrix[times, 0] = lats
    feature_matrix[times, 1] = lngs
    feature_matrix[times, 2] = distance
    feature_matrix[times, 3] = street_count
    feature_matrix[times, 4] = crossing
    feature_matrix[times, 5] = junction
    feature_matrix[times, 6] = railway
    feature_matrix[times, 7] = station
    labels.append(kaggle_LA_2020_new["Accident"].iloc[dist_index])
    times+=1
    if kaggle_LA_2020["Accident"].iloc[dist_index]==1:
        num_1s+=1
    else:
        num_0s+=1
print(len(labels))
print("number of 1s:", num_1s)
print("number of 0s:", num_0s)

lat_min = np.min(feature_matrix[:, 0])
lat_max = np.max(feature_matrix[:, 0])
lng_min = np.min(feature_matrix[:, 1])
lng_max = np.max(feature_matrix[:, 1])
# 20 x 20 = 400 blocks
lat_interval = (lat_max-lat_min)/20
lng_interval = (lng_max-lng_min)/20
save_dict = {}
for i in range(feature_matrix.shape[0]):
    xx = min(int((feature_matrix[i, 0]-lat_min)/lat_interval), 19)
    yy = min(int((feature_matrix[i, 1]-lng_min)/lng_interval), 19)
    if tuple([xx, yy]) not in save_dict:
        save_dict[tuple([xx, yy])] = [i]
    else:
        save_dict[tuple([xx, yy])].append(i)

graphs = []
for key in save_dict:
    indexes = save_dict[key]
    if len(indexes)<=5:
        continue
    node_features = []
    nodes = []
    node_labels = []
    for index in indexes:
        node_features.append(list(feature_matrix[index, 2:]))
        node_labels.append(labels[index])
        nodes.append(nodes_list[index])
    node_features = np.array(node_features)    
    x = torch.tensor(node_features, dtype=torch.float)
    node_labels = np.array(node_labels)
    y = torch.tensor(node_labels)
    nodes_dict1 =  dict()
    val = 0
    for item in nodes:
        nodes_dict1[item] = val
        val+=1
    
    sub_G = G.subgraph(nodes)
    sources = []
    targets = []
    edge_attributes = []  # length, bridge, lanes, oneway, maxspeed, tunnel
    for edge in sub_G.edges:
        sources.append(nodes_dict1[edge[0]])
        targets.append(nodes_dict1[edge[1]])
        edge_attr = sub_G[edge[0]][edge[1]][0]
        if "length" in edge_attr:
            length = edge_attr["length"]
        else:
            length = 0
        
        if "bridge" in edge_attr:
            if edge_attr["bridge"]=="yes":
                bridge=1
            else:
                bridge=0
        else:
            bridge=0
            
        if "tunnel" in edge_attr:
            if edge_attr["tunnel"]=="yes":
                tunnel=1
            else:
                tunnel=0
        else:
            tunnel=0
        
        if "lanes" in edge_attr:
            if type(edge_attr["lanes"]) == list:
                temp = list(map(int, edge_attr["lanes"]))
                lanes = max(temp)
            else:
                lanes = int(edge_attr["lanes"])
        else:
            lanes = 0
        
        if "oneway" in edge_attr:
            oneway = int(edge_attr["oneway"])
        else:
            oneway = 0
            
        if "maxspeed" in edge_attr:
            if type(edge_attr["maxspeed"]) == list:
                temp = list(map(lambda x: int(x.split()[0]), edge_attr["maxspeed"]))
                maxspeed = max(temp)
            else:
                maxspeed = int(edge_attr["maxspeed"].split()[0])
        else:
            maxspeed = 0
        edge_attributes.append([length, bridge, tunnel, lanes, oneway, maxspeed])
    edge_attributes = np.array(edge_attributes)
    edge_attr = torch.tensor(edge_attributes, dtype=torch.float)
    edge_index = torch.tensor([sources, targets], dtype=torch.long)
    data = Data(edge_index=edge_index, x=x, y=y, edge_attr=edge_attr)
    graphs.append(data)
print("total number:", len(graphs))

batch_size = 16
random.shuffle(graphs)
train_sets = graphs[:-40]
test_sets = graphs[-40:]
train_loader = DataLoader(train_sets, batch_size=batch_size)
model = GNNModel(2)
optimizer = AdamW(model.parameters(), lr=3e-3, weight_decay=5e-4)
class_weights = torch.FloatTensor([0.3, 0.75])
criterion = nn.CrossEntropyLoss(weight=class_weights)
best_loss = float("inf")
epochs = 201
interval = 25

for epoch in range(epochs):
    model.train()
    losses = 0
    nums = len(train_sets)
    for _, data in enumerate(train_loader):
        data = data
        labels = data.y
        predicts = model(data)
        loss = criterion(predicts, labels)
        losses+=loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"epoch:{epoch}, loss:{losses/nums}")
    avg_loss = losses/nums
    if avg_loss<best_loss:
        best_loss = avg_loss
    if epoch%interval == 0:
        model.eval()
        acc_avg = 0
        times = 0
        predicted_y = []
        expected_y = []
        for _, data in enumerate(test_sets):
            _, pred = model(data).max(dim=1)
            predicted_y.extend(list(pred.numpy()))
            expected_y.extend(list(data.y.numpy()))
            correct = int(pred.eq(data.y).sum().item())
            acc = correct / int(data.x.shape[0])
            acc_avg+=acc
            times+=1
        print(f"acc: {acc_avg/times}")
        
predicted_y_final = np.array(predicted_y)
expected_y_final = np.array(expected_y)
print(Counter(list(expected_y_final)))
print(classification_report(expected_y_final, predicted_y_final))
print(confusion_matrix(expected_y_final, predicted_y_final))
test_auc = roc_auc_score(expected_y_final, predicted_y_final)
print(test_auc)

