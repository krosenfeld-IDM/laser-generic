#!/usr/bin/env python
# coding: utf-8

# # `grid()` function examples
# 
# The `grid()` function has an option to pass in a custom population function which receives the x and y indices of the current cell. This can be used to create specific scenarios.

# ## Central city
# 
# Let's construct a scenario with a large central population and smaller outlying populations.

# In[1]:


from laser.core.utils import grid

# In[2]:


M = 7
CY = M // 2
N = 7
CX = N // 2

def central_pop(x, y):
    d = abs(x - CX) + abs(y - CY) + 1
    p = 1_000_000 / (10 * d)
    return int(p)

scenario = grid(M, N, population_fn=central_pop)
prj = scenario.to_crs(3857)
ax = prj.plot(column="population", cmap="viridis", legend=True, figsize=(12, 9))
prj.centroid.plot(ax=ax, color="red", marker="x", markersize=100)
_ = ax.set_xlabel("meters")
_ = ax.set_ylabel("meters")

# ## Ring
# 
# Let's construct a scenario with an outer ring of highly populated nodes.

# In[3]:


LEFT = 0
RIGHT = N - 1
TOP = M - 1
BOTTOM = 0

def ring_pop(x, y):

    dx = min(x - LEFT, RIGHT - x)
    dy = min(y - BOTTOM, TOP - y)
    d = min(dx, dy) + 1

    pop = int(round(1_000_000 / (5 * d)))

    return pop

scenario = grid(M, N, population_fn=ring_pop)
prj = scenario.to_crs(3857)
ax = prj.plot(column="population", cmap="viridis", legend=True, figsize=(12, 9))
prj.centroid.plot(ax=ax, color="red", marker="x", markersize=100)
_ = ax.set_xlabel("meters")
_ = ax.set_ylabel("meters")

# ## Linear
# 
# Let's do a 1-D scenario.

# In[4]:


ROWS = 1
CY = ROWS // 2
COLUMNS = 9
CX = COLUMNS // 2

def linear_pop(row, col):
    d = abs(col - CX) + 1
    pop = int(round(1_000_000 / d))
    return pop

scenario = grid(ROWS, COLUMNS, population_fn=linear_pop)
prj = scenario.to_crs(3857)
ax = prj.plot(column="population", cmap="viridis", legend=True, figsize=(12, 9))
prj.centroid.plot(ax=ax, color="red", marker="x", markersize=100)
_ = ax.set_xlabel("meters")
_ = ax.set_ylabel("meters")

# ## Non-`grid()` custom scenario
# 
# Let's look at the GeoDataFrame returned by `grid()` and create our own, custom hub-and-spoke scenario.

# In[5]:


scenario.head()


# In[6]:


import geopandas as gpd
from shapely import Point
import math

# We will create a central city of 1,000,000 population with five surrounding towns of 100,000 population each 150 kilometers away.
# Each surrounding town will have two intermediate population centers of 25,000 between it and the central city.
nodes = []
nodeid = 0
# Add central city node at 0,0 with a circle geometry of radius 10 km
nodes.append({"name": "Central City", "population": 1_000_000, "geometry": Point(0, 0).buffer(10_000), "nodeid": nodeid})
nodeid += 1
# Surrounding towns
NTOWNS  = 5
for i in range(NTOWNS):
    angle = (i * 2 * math.pi / NTOWNS)
    tx = 150_000 * (cos := math.cos(angle))
    ty = 150_000 * (sin := math.sin(angle))
    nodes.append({"name": f"Town {i+1}", "population": 100_000, "geometry": Point(tx, ty).buffer(10_000), "nodeid": nodeid})
    nodeid += 1
    # Intermediate population centers
    ix1 = 50_000 * cos
    iy1 = 50_000 * sin
    nodes.append({"name": f"Interim {i+1}a", "population": 25_000, "geometry": Point(ix1, iy1).buffer(5_000), "nodeid": nodeid})
    nodeid += 1
    ix2 = 100_000 * cos
    iy2 = 100_000 * sin
    nodes.append({"name": f"Interim {i+1}b", "population": 25_000, "geometry": Point(ix2, iy2).buffer(5_000), "nodeid": nodeid})
    nodeid += 1

scenario = gpd.GeoDataFrame(nodes, crs="EPSG:3857") # Mark as 3857 since we're working in meters/metres.
scenario.head()

# In[7]:


scenario.plot(column="population", cmap="viridis", legend=True, figsize=(12, 9))

# ## SEIR model
# 
# Let's run an SEIR model on that scenario. We will seed infections in one of the radial cities.

# In[8]:


from laser.core import PropertySet
from laser.generic import Model
import laser.generic.SEIR as SEIR
from laser.core.distributions import normal

exp_mean = 7.0
exp_stddev = 1.0
inf_mean = 14.0
inf_stddev = 2.0
R0 = 2.0
beta = R0 / inf_mean

parameters = PropertySet({"nticks": 365, "beta": beta, "gravity_c": 2.5})
scenario["S"] = scenario.population
scenario["E"] = 0
scenario["I"] = 0
scenario["R"] = 0
scenario.loc[scenario.name == "Town 1", "S"] -= 100
scenario.loc[scenario.name == "Town 1", "I"] += 100
prj = scenario.to_crs(4326)
model = Model(prj, parameters)
incubation_distribution = normal(loc=exp_mean, scale=exp_stddev)
infectious_distribution = normal(loc=inf_mean, scale=inf_stddev)
model.components = [
    SEIR.Susceptible(model),
    SEIR.Recovered(model),
    SEIR.Infectious(model, infectious_distribution, 1),
    SEIR.Exposed(model, incubation_distribution, infectious_distribution, 1, 1),
    SEIR.Transmission(model, incubation_distribution, 1)
]
model.run()

import matplotlib.pyplot as plt

fig, axes = plt.subplots(4, 1, figsize=(14, 18), sharex=True)
compartments = ['S', 'E', 'I', 'R']

for idx, comp in enumerate(compartments):
    for node in scenario.index:
        axes[idx].plot(getattr(model.nodes, comp)[:, node], label=scenario.loc[node, "name"])
    axes[idx].set_ylabel(comp)
    axes[idx].legend(loc='upper right', fontsize='small')
axes[-1].set_xlabel("Time (days)")
plt.tight_layout()
plt.show()

# In[9]:


# model.network


# In[10]:


import matplotlib.pyplot as plt

fig, axes = plt.subplots(3, 4, figsize=(20, 15))
ticks = list(range(0, 361, 30))

for ax, tick in zip(axes.flat, ticks):
    prj["I_tick"] = model.nodes.I[tick, :] # / prj.population
    prj.plot(column="I_tick", cmap="Reds", legend=True, ax=ax)
    ax.set_title(f"Infectious at tick {tick}")
    ax.axis('off')

plt.tight_layout()
plt.show()
