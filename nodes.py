import numpy as np
from itertools import combinations, permutations
import random
import matplotlib.pyplot as plt
import networkx as nx
import tensorflow as tf
import pandas as pd
import csv
import os

import dash
from dash import dcc, html, callback_context
import dash_daq as daq
import networkx as nx 
import plotly.graph_objs as go
from colour import Color
from datetime import datetime 
from textwrap import dedent as d 
import json


r = lambda: random.randint(0,255)

class Network():
    def __init__(self,*,dropout=0.75,size=10,threshold=0.65,decay=0.05,init_activation=0.1):
        self.nodes=[]
        self.size=size
        self.age=0
        self.dropout=dropout
        self.threshold=threshold
        self.decay=decay
        self.initial_activation=init_activation

        self_loops = random.sample([(x,x) for x in range(size)], k=round(size*(dropout)))

        links = list(permutations(range(size), 2))
        links = random.sample(links, k=round(len(links) * (1-dropout)))
        links.extend(self_loops)

        for node in range(size):
            rel_links = list(filter(lambda y: y[0] == node, links))

            weights = list(np.round(np.random.uniform(0,1,len(rel_links)), 2))
            weights = np.round(weights / np.sum(weights), 3)

            self.nodes.append(Node(node, rel_links, weights))

    def __save__(self):
        edge_data = pd.DataFrame(self._get_ebunch_as_tuples(), columns=['start','finish','weight'])
        edge_data.to_csv(f'./edge-data-{self.age}.csv')
        node_data = pd.DataFrame(self._get_node_data(), columns=['node', 'activation'])
        node_data.to_csv(f'./node-data-{self.age}.csv')

    @classmethod
    def __load__(cls, age):
        assert os.path.exists(f'./edge-data-{age}.csv')
        assert os.path.exists(f'./node-data-{age}.csv')
        edge_data = pd.read_csv(f'./edge-data-{age}.csv')
        node_data = {}
        with open(f'./node-data-{age}.csv') as f:
            next(f)  # Skip the header
            reader = csv.reader(f, skipinitialspace=True)
            for row in reader:
                node_data[row['node']] = row['activation']

        edge_data = list(edge_data.itertuples(index=False, name=None))
        node_list = []
        for i in range(len(edge_data)):
            filtered_links = list(filter(lambda y: y[0] == i, edge_data))
            rel_links = [(y[0], y[1]) for y in filtered_links]
            weights = [(y[2] for y in filtered_links)]
            node_list.append(Node(i, rel_links, weights, activation=node_data[i]))
        G = cls(dropout=0.75, size = 0)
        G.nodes = node_list
        G.age = age
        return G

    def _get_node_data(self):
        nodeData= []
        for node in self.nodes:
            nodeData.append((node.ident, node.activation))
        return nodeData

    def _get_ebunch(self):
        ebunch = []
        for node in self.nodes:
            ebunch.extend(node.linkObjects)
        return ebunch
    
    def _get_ebunch_as_tuples(self, activation:bool=False):
        ebunch = []
        for node in self.nodes:
            for link in node.linkObjects:
                ebunch.append(link.export())
        return ebunch
    
    def activate(self):
        number_activated = round(len(self.nodes) * self.initial_activation)
        if number_activated == 0:
            number_activated = 1
        samples = random.sample(self.nodes, number_activated)
        for sample in samples:
            sample.set_activation(float(1))
    
    def _get_positions(self):
        G = nx.DiGraph()
        G.add_weighted_edges_from(self._get_ebunch_as_tuples())
        pos = nx.drawing.circular_layout(G)

        for node in range(len(G.nodes)):
            self.nodes[node].pos = list(pos[node])

        return pos

    def _get_active(self):
        return [i.activation for i in self.nodes]
    
    def _set_active(self, active):
        for node in range(len(self.nodes)):
            self.nodes[node].activation = active[node]

    def _get_active_matrix(self):
        markov = np.zeros([self.size, self.size])
        for link in self._get_ebunch_as_tuples():
            start, finish, weight = link
            markov[finish][start] = weight
        return markov

    def forward(self, epochs):
        markov = self._get_active_matrix()
        self._get_active_matrix
        current = np.asarray(self._get_active())
        for epoch in range(epochs):
            print(f"current: {current.shape}\nmarkov: {markov.shape}\n")
            current = np.matmul(current, markov)
            self.age += 1
        self._set_active(current.tolist())
        return current

    def web_app(self):
        pos = self._get_positions()
        self.activate()

        traceRecode = []

        for edge in self._get_ebunch():
            x0, y0 = self.nodes[edge.start].pos
            x1, y1 = self.nodes[edge.finish].pos
            weight = edge.weight
            trace = go.Scatter(x=tuple([x0, x1, None]), y=tuple([y0, y1, None]),
                            mode='lines',
                            line={'width': weight},
                            line_shape='spline',
                            opacity=1)
            traceRecode.append(trace)
    #################################################################
        node_trace = go.Scatter(x=[], y=[], hovertext=[], text=[], mode='markers+text', textposition="bottom center",
                            hoverinfo="text", marker={'size': 50, 'color': [node.activation for node in self.nodes]})

        for node in range(len(self.nodes)):
            x, y = self.nodes[node].pos
            hovertext = "Activation: " + str(self.nodes[node].activation) + " Outbound Connections: " + str(len(self.nodes[node].edges))
            text = f"{str(node)}: {str(round(self.nodes[node].activation,3))}"
            node_trace['x'] += tuple([x])
            node_trace['y'] += tuple([y])
            node_trace['hovertext'] += tuple([hovertext])
            node_trace['text'] += tuple([text])

        traceRecode.append(node_trace)
    ##################################################################
        middle_hover_trace = go.Scatter(x=[], y=[], hovertext=[], mode='markers', hoverinfo="text",
                                    marker={'size': 20, 'color': 'LightSkyBlue'},
                                    opacity=0)
    
        for edge in self._get_ebunch():
            x0, y0 = self.nodes[edge.start].pos
            x1, y1 = self.nodes[edge.finish].pos
            hovertext = "From: " + str(edge.start) + " To: " + str(
                edge.finish) + " Weighted At: " + str(edge.weight)
            middle_hover_trace['x'] += tuple([(x0 + x1) / 2])
            middle_hover_trace['y'] += tuple([(y0 + y1) / 2])
            middle_hover_trace['hovertext'] += tuple([hovertext])

        traceRecode.append(middle_hover_trace)

        figure = {
            "data": traceRecode,
            "layout": go.Layout(title='Interactive Graph Network Visualization', showlegend=False, hovermode='closest',
                                margin={'b': 40, 'l': 40, 'r': 40, 't': 40},
                                xaxis={'showgrid': False, 'zeroline': False, 'showticklabels': False},
                                yaxis={'showgrid': False, 'zeroline': False, 'showticklabels': False},
                                height=600,
                                clickmode='event+select',
                                annotations=[
                                    dict(
                                        ax=(self.nodes[edge[0]].pos[0] + self.nodes[edge[1]].pos[0]) / 2,
                                        ay=(self.nodes[edge[0]].pos[1] + self.nodes[edge[1]].pos[1]) / 2, axref='x', ayref='y',
                                        x=(self.nodes[edge[1]].pos[0] * 3 + self.nodes[edge[0]].pos[0]) / 4,
                                        y=(self.nodes[edge[1]].pos[1] * 3 + self.nodes[edge[0]].pos[1]) / 4, xref='x', yref='y',
                                        showarrow=True,
                                        arrowhead=3,
                                        arrowsize=4,
                                        arrowwidth=1,
                                        opacity=1
                                    ) for edge in self._get_ebunch_as_tuples()]
                                )}
        return figure

        

class Node():
    def __init__(self, number, links, weights=None, activation=0):
        self.pos = None
        self.activation = activation
        self.ident = number
        self.edges = links
        self.weights = weights
        edges = []
        for item in range(len(links)):
            a, b = links[item]
            c = weights[item]
            edges.append(Link((a,b),c))
            
        self.linkObjects = edges

        self.info = zip(links, weights)

    def add_edge(self, edge):
        assert type(edge) == tuple
        self.edges.append(edge)

    def set_activation(self, new):
        assert type(new) == float
        self.activation = new

    def forward(self):
        choice = random.random()
        options = abs(self.weights - choice)

        dispersion = []
        for weight in self.weights:
            dispersion.append(np.random.normal(weight))
        dispersion = list(np.array(dispersion) / sum(dispersion))

        result = {}
        for numb in range(len(self.weights)):
            result[self.links[numb]] = dispersion[numb]

        if random.randint(0,100) >= 5:
            return result
        result = {list(self.links).index(min(options)) : 1}
        return result

class Link():
    def __init__(self,connection:tuple, weight:float):
        self.start = connection[0]
        self.finish = connection[1]
        self.connection = connection
        self.weight = weight
    
    def export(self):
        return (self.start, self.finish, self.weight)
   
    def strengthen(self):
        if self.weight != 1:
            self.weight = self.weight * 1.05
            if self.weight > 1:
                self.weight = 1
        print(f"Connection for nodes {self.connection} is now at strength {self.weight}")
        
    def weaken(self):
        if self.weight != 0:
            self.weight = self.weight * 0.95
            if self.weight < 0:
                self.weight = 0
        print(f"Connection for nodes {self.connection} is now at strength {self.weight}")