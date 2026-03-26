#!/usr/bin/env python
# coding: utf-8

# # Modeling the spread of rabies in one dimension
# 
# In this notebook, we will move from the 2 connected patches described in [Notebook 8](https://github.com/laser-base/laser-generic/blob/50796ab5a253df168255e8a3ac54b766f6aa55d5/notebooks/08_2patch_SIR_wbirths_correlation.ipynb) to a 1-D grid of connected population patches.  This is the scenario explored to describe the spatial spread of rabies in foxes in the work of [Kallen, Arcuri, and Murray, Journal of Theoretical Biology (1985) 116, 377-393](https://pubmed.ncbi.nlm.nih.gov/4058027/). The relevant model equations (note that these are now PDEs rather than the ODEs of previous notebooks) are:
# 
# $$
# \frac{\partial S}{\partial t} = -KIS
# $$
# 
# $$
# \frac{\partial I}{\partial t} = D \frac{\partial^2 I}{\partial x^2} + KIS - \mu I
# $$ 
# 
# and come to the conclusion that this system supports traveling waves with velocity bounded below by 
# 
# $$
# c = 2 \sqrt{D \mu (1/r -1)}
# $$
# 
# where $r = \frac{\mu}{K S_0}$ and $S_0$ is the initial density of susceptibles.
# 
# The analogy to the SIR model equations we have been using is clear, though this model conceptually different in specific ways from that model as we now have spatially dimensional quantities.  For example, I & S are infective/susceptible population densities rather than population counts, which affects translating between the $K$ of this model and the $\beta$ of the SIR; the derived quantity $r$ is clearly analogous to $\frac{1}{R_0}$ of the SIR model, but requires the presence of the initial susceptible density to be properly dimensionless, and rabies being invariably fatal, $\mu$ represents the mortality of rabies rather than the $\gamma$ we would use for recovery.  
# 
# With all that said, if we are careful about translation between parameters, and appropriately construct the transmission network to appropriately reflect 1-D diffusion, we ought to be able to reproduce this model in our SIR implementation, and recapitulate the traveling wave behavior of the outbreak.  As we do this, we will also run into two well-understood issues in simulating traveling waves of the expected wavespeed on a finite lattice, both biasing us to produce slower waves on average.  
# 
# The first comes from discretization - we are transitioning from a PDE continuous in space, time, and the relevant variable $I$, to a simulation that is discrete in all 3.  This discretization will make it impossible for infection to travel more than 1 lattice spacing per simulation timestep, so in parameter regimes where the expected wavespeed approaches a lattice spacing per simulation timestep, our model will be incapable of demonstrating that.  
# 
# The second slowdown ill come from discretization of the population itself, and from stochasticity.  This phenomenon is sometimes referred to as the "attofox" problem, and is discussed, e.g., in the chapter on Spatial Modeling in Keeling and Rohani, Modeling Infectious Diseases in Humans and Animals.  In a model with deterministic dynamics and continuous populations, any fraction of infectivity deposited in a population can generate fractional infectives, and deterministically kick off the start of a local outbreak.  In a discrete population, stochastic model, we cannot create new infections in smaller units than a single infected individual. This causes stochastic delays relative to the continuous model - these delays can arise both in exportation - it can take longer to infect an individual in an uninfected location, and it can take longer for the outbreak to then progress from 1 individual to more, and the wavespeed will on average be slower.  
# 
# In fact, as long as $D>0$ and $1/r >1$, the continuous equation is guaranteed to support traveling waves, but for small D and small 1/r the stochastic version may simply die out in the initial node before successfully propagating to adjacent nodes, or it may die out at some subsequent node along the way.
# 
# So as we fit this result, we should expect to see the observed wavespeed consistently be slower than expected at very high propagation speeds.  At lower propagation speeds, the match between model and theory will vary with the values of D, $\beta$\, $\gamma$ - we should expect better match between theory and observation but with a general bias for observation to be slower.

# In[1]:


import numpy as np
import pandas as pd
from laser.core.propertyset import PropertySet
import laser.core.distributions as dists
from laser.generic import SIR
from laser.generic import Model
from laser.core.utils import grid
import laser.core
import laser.generic
import matplotlib.pyplot as plt
import os
from scipy.optimize import fsolve

print(f"{np.__version__=}")
print(f"{laser.core.__version__=}")
print(f"{laser.generic.__version__=}")


# In[2]:


#Set up the network and initial conditions
npatches = 201
pop = 1000
scenario = grid(M=1, N=npatches, node_size_degs=0.05, population_fn=lambda x,y: pop, origin_x=0, origin_y=0)
initial_infected = 3
scenario["I"] = 0
scenario["R"] = 0
scenario["S"] = scenario.population
scenario.loc[scenario.nodeid==101,"I"] = initial_infected
scenario.loc[scenario.nodeid==101,"S"] = scenario.loc[scenario.nodeid==101,"population"]-initial_infected

# ## The transmission network
# Spatial transmission in this rabies model is contained in the diffusive term $D \frac{\partial^2 I}{\partial x^2}$.  We are operating on a discretized 1-D grid, and will take advantage of the usual way to discretize a second derivative
# 
# $$
# D \frac{\partial^2 I}{\partial x^2} \rightarrow D \frac {I_{x-h} - 2 I_{x} + I_{x+h}} {h^2}
# $$
# 
# To translate this into a network structure for the model, we can arbitrarily set the  length scale $h$ to be 1 in units of the grid spacing.  Looking at the above, from the perspective of an arbitrary node $i$, the transmission matrix must "send" $D I_i$ units of infectivity to the adjacent nodes on either side, and remove that infectivity from the local pool.   Looking at the spatial transmission code (line 1006-8 in src/laser_generic/models/components.py), we see that spatial transmission is handled by multiplying the transmission matrix and the vector of infectivities by node, and that it internally handles subtracting the "outbound infectivity" from the local source.  Thus, the above discretization can be accomplished by filling the value $D$ on the first superdiagonal and subdiagonal of the transmission matrix, and 0 everywhere else.

# In[3]:


infmean = 10.0
R0 = 10.0
nticks = 2 * 365
D = .01
parameters = PropertySet(
    {
        "seed": np.random.randint(0, 1000000),
        "nticks": nticks,
        "verbose": True,
        "beta": R0 / infmean,
        "inf_mean": infmean,
    }
)

infdurdist = dists.exponential(scale=parameters.inf_mean)
model = Model(scenario, parameters)
model.network = 0*model.network
np.fill_diagonal(model.network[1:], D)
np.fill_diagonal(model.network[:,1:], D)

model.components = [
    SIR.Susceptible(model),
    SIR.Recovered(model),
    SIR.Infectious(model, infdurdist),
    SIR.Transmission(model, infdurdist),
]
model.run()



# In[4]:


plt.figure(figsize=(9, 4))
plt.imshow(model.nodes.I.T, aspect='auto', origin='lower')
plt.xlabel('Time (tick)')
plt.ylabel('Patch')
plt.title('Infectious (I) over time and space')
plt.colorbar(label='Infectious')
plt.show()


# In[5]:


from scipy.stats import linregress

# Find the time (tick) of peak I for each node
peak_times = model.nodes.I.argmax(axis=0)  # shape: (npatches,)

# Get the node indices for right-going (101 to 200) and left-going (100 to 0) waves
right_nodes = np.arange(101, npatches)
left_nodes = np.arange(100, -1, -1)

# Fit a line to peak_times vs node index for each direction to estimate velocity (slope = 1/velocity)

# Right-going wave
right_peak_times = peak_times[right_nodes]
slope_r, intercept_r, r_value_r, p_value_r, std_err_r = linregress(right_nodes[right_peak_times>0], right_peak_times[right_peak_times > 0])
velocity_right = 1 / slope_r if slope_r != 0 else np.nan

# Left-going wave
left_peak_times = peak_times[left_nodes]
slope_l, intercept_l, r_value_l, p_value_l, std_err_l = linregress(left_nodes[left_peak_times>0], left_peak_times[left_peak_times > 0])
velocity_left = -1 / slope_l if slope_l != 0 else np.nan  # negative because left

print(f"Estimated right-going wavespeed: {velocity_right:.3f} patches/tick")
print(f"Estimated left-going wavespeed: {velocity_left:.3f} patches/tick")
print(f"Expected wavespeed: {2*np.sqrt(D/infmean*(R0-1)):.3f} patches/tick")

# In[6]:


# Set up parameter ranges
n_samples = 100
D_samples = np.logspace(np.log10(3e-4), np.log10(3e-1), n_samples)
R0_samples = np.random.uniform(1.5, 10, n_samples)
infmean_samples = np.random.uniform(7, 50, n_samples)

results = []

for i, (D, R0, infmean) in enumerate(zip(D_samples, R0_samples, infmean_samples)):
    # Set up parameters for this run
    parameters = PropertySet(
        {
            "seed": np.random.randint(0, 1000000),
            "nticks": nticks,
            "verbose": False,
            "beta": R0 / infmean,
            "inf_mean": infmean,
        }
    )
    infdurdist = dists.exponential(scale=parameters.inf_mean)
    model = Model(scenario.copy(), parameters)
    model.network = 0 * model.network
    np.fill_diagonal(model.network[1:], D)
    np.fill_diagonal(model.network[:, 1:], D)
    model.components = [
        SIR.Susceptible(model),
        SIR.Recovered(model),
        SIR.Infectious(model, infdurdist),
        SIR.Transmission(model, infdurdist),
    ]
    model.run(f"Sim {i+1}/{n_samples}")
    # Estimate wavespeed as before
    peak_times = model.nodes.I.argmax(axis=0)
    right_nodes = np.arange(101, npatches)
    left_nodes = np.arange(100, -1, -1)
    right_peak_times = peak_times[right_nodes]
    left_peak_times = peak_times[left_nodes]
    # Right
    mask_r = right_peak_times > 0
    if np.any(mask_r):
        slope_r, *_ = linregress(right_nodes[mask_r], right_peak_times[mask_r])
        velocity_right = 1 / slope_r if slope_r != 0 else np.nan
    else:
        velocity_right = np.nan
    # Left
    mask_l = left_peak_times > 0
    if np.any(mask_l):
        slope_l, *_ = linregress(left_nodes[mask_l], left_peak_times[mask_l])
        velocity_left = -1 / slope_l if slope_l != 0 else np.nan
    else:
        velocity_left = np.nan
    # Expected wavespeed (from theory)
    expected_speed = 2 * np.sqrt(D / infmean * (R0 - 1))
    results.append({
        "D": D,
        "R0": R0,
        "infmean": infmean,
        "velocity_right": velocity_right,
        "velocity_left": velocity_left,
        "expected_speed": expected_speed,
    })

results_df = pd.DataFrame(results)


# In[7]:


plt.figure(figsize=(8, 6))
plt.plot(results_df["expected_speed"], results_df["velocity_left"], 'r.', label="Velocity left", markersize=12)
plt.plot(results_df["expected_speed"], results_df["velocity_right"], 'b.', label="Velocity right", markersize=12)
plt.plot([1e-3, 1], [1e-3, 1], 'k--', label="y=x")
plt.xscale('log')
plt.yscale('log')
plt.xlabel("Expected wavespeed (patches/tick)")
plt.ylabel("Measured wavespeed (patches/tick)")
plt.title("Comparison of expected and observed wavespeeds")
min_val = np.nanmin(np.abs([results_df["expected_speed"].min(), results_df["velocity_right"].min(), results_df["velocity_left"].min()]))
max_val = np.nanmax([results_df["expected_speed"].max(), results_df["velocity_right"].max(), results_df["velocity_left"].max()])
plt.xlim(min_val, max_val)
plt.ylim(min_val, max_val)
plt.legend()
plt.tight_layout()
plt.show()


# ## Results
# 
# As expected, at high speeds the observed wavespeed is consistently slower than theoretical expetactation.  At slower wavespeeds, the model is more able to reproduce waves traveling around the correct speed, but there is still a bias towards the model being slower than expected.  Below, we will take a look at how the errors behave with model parameters.

# In[8]:


# Compute relative errors
rel_err_left = (results_df["velocity_left"] - results_df["expected_speed"]) / results_df["expected_speed"]
rel_err_right = (results_df["velocity_right"] - results_df["expected_speed"]) / results_df["expected_speed"]

# Plot: colored by R0
plt.figure(figsize=(8, 6))
sc = plt.scatter(results_df["expected_speed"], rel_err_right, c=results_df["R0"]/results_df['infmean'], cmap='viridis', label="Right", marker='o')
plt.colorbar(sc, label="beta")
plt.scatter(results_df["expected_speed"], rel_err_left, c=results_df["R0"]/results_df['infmean'], cmap='viridis', marker='x', label="Left")
plt.xscale('log')
plt.xlabel("Expected wavespeed (patches/tick)")
plt.ylabel("Relative error (measured - expected) / expected")
plt.title("Relative error vs. expected speed (colored by beta)")
plt.legend()
plt.tight_layout()
plt.show()

# Plot: colored by D
plt.figure(figsize=(8, 6))
sc = plt.scatter(results_df["expected_speed"], rel_err_right, c=np.log10(results_df["D"]), cmap='plasma', label="Right", marker='o')
plt.colorbar(sc, label="log10(D)")
plt.scatter(results_df["expected_speed"], rel_err_left, c=np.log10(results_df["D"]), cmap='plasma', marker='x', label="Left")
plt.xscale('log')
plt.xlabel("Expected wavespeed (patches/tick)")
plt.ylabel("Relative error (measured - expected) / expected")
plt.title("Relative error vs. expected speed (colored by log10(D))")
plt.legend()
plt.tight_layout()
plt.show()

# Plot: colored by infmean
plt.figure(figsize=(8, 6))
sc = plt.scatter(results_df["expected_speed"], rel_err_right, c=results_df["infmean"], cmap='cividis', label="Right", marker='o')
plt.colorbar(sc, label="inf_mean")
plt.scatter(results_df["expected_speed"], rel_err_left, c=results_df["infmean"], cmap='cividis', marker='x', label="Left")
plt.xscale('log')
plt.xlabel("Expected wavespeed (patches/tick)")
plt.ylabel("Relative error (measured - expected) / expected")
plt.title("Relative error vs. expected speed (colored by inf_mean)")
plt.legend()
plt.tight_layout()
plt.show()


# ## Discussion
# 
# We can see clear vertical gradients in the error plots colored by $\beta$ and infectious duration - lower values of $beta$, and longer infectious durations are both clearly correlated with producing observed waves that are slower than theoretical expectations.  This makes some sense - lower values of both parameters translate into lower per-infective, per-timestep force of infection, which can cause delays in both the important processes of spatial spread of infections and the growth of local outbreaks in each node.  The dependence on D is present but less obvious, as this parameter can only affect the delays related to spatial spread.
