#!/usr/bin/env python
# coding: utf-8

# # SI model with constant-population demographics
# 
# Building up from the SI model without demography, we next explore the addition of basic demographics - adding a birth rate & an equivalent, age-independent mortality rate $\mu$ to keep constant total population.  The disease model remains the SI model.
# 
# $$
# \dot{S} = -\frac{\beta*S*I}{N} + \mu N - \mu S
# $$
# 
# $$
# \dot{I} = \frac{\beta*S*I}{N} - \mu I
# $$
# 
# With a bit of manipulation as in the previous example, this can be worked back into the form of a generalized logistic growth differential equation
# $$ \dot{I} = \beta I (1-\frac{\mu}{\beta}-\frac{I}{N})$$
# 
# With solution 
# $$
# \frac{Nx}{1+(\frac{Nx}{I_0}-1)e^{-\beta x t}}
# $$
# 
# $$
# x = (1-\frac{\mu}{\beta})
# $$
# 
# 
# This notebook tests the implementation and behavior of the model as follows:
# ## Construct the model
# 
# In the first few cells, we do all the necessary imports.  Then we construct a single-patch LASER model with four components: the `Susceptible` and `Infected` states, and the `Transmission`, and `ConstantPopVitalDynamics` \"flows\" between states.  Finally, we initialize with a single infection and run.  The `Susceptible`, `Infected` and `Transmission` components are described in the the model with no births.  `ConstantPopVitalDynamics` enables births and deaths while strictly enforcing constant population rather than having separate stochastic birth and death processes that induce population variance over time.  To do this, births are implemented by randomly choosing an existing agent and re-initializing them as a newborn; in this way, each birth of a new agent is exactly offset by the death of an existing one. 
# 
# ## Sanity check
# 
# The first test ensures certain basic constraints are being obeyed by the model.  We confirm that at each timestep, $S_t=N_t-I_t$.
# 
# ## Scientific test
# 
# Finally, we come to the scientific test.  First, we run a basic test on a single model, and show the result matches expectation with known inputs.  Then, as before, we select a few values of β and μ, run the model, fit the outputs to the logistic equation, and compare the fitted values of the parameters to the known value.  The equation above makes it clear that attempting to fit β, μ and N all together will be degenerate - the logistic equation has two parameters, not three; in this case those two are the products Nx and βx.  Thus, we fix the total population size N in the fit to focus on the more interesting β and μ parameters.  And as before, because we are approximating continuously compounding growth, in the logistic equation, with a discrete time-step approximation, we expect the fitted values of β to be biased slightly downward - that is, the modeled trajectory is slightly slower than the continuous-time counterpart.  This error grows as β gets larger; the test fails if any of the fitted β values are more than 5% away from the known values.  Furthermore, if the outbreak is seeded by only one infection, it is possible that that one infection dies prior to infecting anyone else; we therefore seed with a few infections rather than one.
# 
# 

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from laser.core.propertyset import PropertySet
from scipy.optimize import curve_fit
from scipy.optimize import minimize

import laser.core
import laser.generic

import laser.generic.SI as SI
from laser.generic import Model
from laser.generic.utils import ValuesMap
from laser.core.utils import grid
from laser.generic.vitaldynamics import ConstantPopVitalDynamics

print(f"{np.__version__=}")
print(f"{laser.core.__version__=}")
print(f"{laser.generic.__version__=}")


# In[ ]:


pop = 1e6
init_inf = 1
# Just for fun, use coordinates of Seattle, WA
latitude = 47+(36+(35/60))/60
longitude = -(122+(19+(59/60))/60)
scenario = grid(M=1, N=1, node_size_degs=0.08983, population_fn=lambda x, y:  pop,  origin_x=longitude, origin_y=latitude)
scenario["S"] = scenario.population - init_inf
scenario["I"] = init_inf
parameters = PropertySet({"prng_seed": 314159265, "nticks": 730, "verbose": True, "beta": 0.04, "cbr": 40.0})
# Question - if I provide a scalar birth rate, should the demographics steps know to convert that to a ratemap behind the scenes?
birthrates = ValuesMap.from_scalar(parameters.cbr, nticks=parameters.nticks, nnodes=1)

# Run until we see an outbreak
outbreak = False

while not outbreak:
    parameters.prng_seed += 1
    model = Model(scenario, parameters, birthrates=ValuesMap.from_scalar(0.0, nticks=parameters.nticks, nnodes=1))

    model.components = [
        SI.Susceptible(model),
        SI.Infectious(model),
        ConstantPopVitalDynamics(model, birthrates),
        SI.Transmission(model),
        ]

    model.run()

    outbreak = np.any(model.nodes.I[200] > 0)

# ## Sanity checks
# Check that the relationships between susceptible, infected, and total population hold.  In the version with no births, we could also test that the infected = the sum of past incidence, but with births now playing a role that no longer holds.

# In[3]:


plt.plot(model.nodes.I.astype("int"), lw=4)
N = model.nodes.S + model.nodes.I
plt.plot(N - model.nodes.S, "--", lw=3)
plt.yscale("log")
plt.legend(["Population minus currently infected", "Susceptible", "Population minus cumulative infections (incidence)"])

print("S = N-I:  " + str(np.isclose(model.nodes.S, N - model.nodes.I).all()))

# The below plot shows the model output, the expected logistic growth curve, and the expected logistic growth curve fit to the model with a free offset $t_0$ to account for stochasticity in when the outbreak takes off.  The resulting plot should show good concordance between the model output and the expected logistic equation with the known model inputs $\beta$ and population.
# The goodness of this fit could be turned into a strict pass/fail test down the line.

# In[4]:


def SI_logistic(t, beta, popsize, cbr, t0):
    mu = (1 + cbr / 1000) ** (1 / 365) - 1
    x = 1 - mu / beta
    return popsize * x / (1 + (popsize * x - 1) * np.exp(-beta * x * (t - t0)))


t = np.arange(model.params.nticks)


def objective(t0):
    return np.sum((1 - SI_logistic(t, model.params.beta, pop, model.params.cbr, t0) / np.squeeze(model.nodes.I[1:,:])) ** 2)


result = minimize(objective, x0=10)
t0_opt = result.x[0]

plt.plot(model.nodes.I, lw=4)
plt.plot(SI_logistic(t, model.params.beta, pop, model.params.cbr, 0), lw=3)
plt.plot(SI_logistic(t, model.params.beta, pop, model.params.cbr, t0_opt), "r:", lw=3)
plt.yscale("log")
plt.legend(["Model output", "Logistic growth with known inputs, t0=0", f"Logistic growth with known inputs, best-fit t0 = {t0_opt:.1f}"])
plt.show()

# ## Scientific testing
# Finally, we run the model for a range of β & birth rate  parameters, we freely fit the model output to the logistic equation, and we compare the known input parameters against the parameters fitted from output.  
# 
# Because we are approximating continuously compounding growth by discrete-time compounding growth, we should expect the fitted β  to consistently be slightly underestimated relative to the true β , with the relative difference growing as β  gets larger.   
# 
# In the future, we could probably compute the expected error from this approximation.  But for now, to make this a pass-fail test, we will raise a flag if the fitted parameters are more than 5% different than the known ones.

# In[ ]:


NTESTS = 10
seeds = [42 + i for i in range(NTESTS)]
pop = 1e5
nticks = 5 * 365
betas = [0.005 + 0.005 * i for i in range(1, NTESTS+1)]
cbrs = np.random.randint(15, 50, NTESTS)

scenario = grid(M=1, N=1, node_size_degs=0.08983, population_fn=lambda x, y:  pop,  origin_x=longitude, origin_y=latitude)
init_inf = 5
scenario["S"] = scenario.population - init_inf
scenario["I"] = init_inf

output = []
for i, (seed, beta, cbr) in enumerate(zip(seeds, betas, cbrs)):

    parameters = PropertySet({"prng_seed": seed, "nticks": nticks, "verbose": True, "beta": beta, "cbr": cbr})
    recyclerates = ValuesMap.from_scalar(parameters.cbr, nticks=parameters.nticks, nnodes=1)
    model = Model(scenario, parameters, birthrates=ValuesMap.from_scalar(0.0, nticks=parameters.nticks, nnodes=1))

    model.components = [
        SI.Susceptible(model),
        SI.Infectious(model),
        ConstantPopVitalDynamics(model, recyclerates),
        SI.Transmission(model),
        ]
    model.run(label=f"SI Model {i+1:2} of {NTESTS} (beta={beta:0.05f}, cbr={cbr:3})")
    cases = model.nodes.I[:-1, 0]
    popt, pcov = curve_fit(
        SI_logistic,
        np.arange(nticks),
        cases,
        p0=[beta * (1 + 0.1 * np.random.normal()), pop, cbr * (1 + 0.1 * np.random.normal()), 1],
        bounds=([0, pop - 1, 0, -100], [1, pop + 1, 600, 100]),
    )
    # Pandas complains about concatenating rows to an empty DataFrame, so build a list of dicts and convert at the end
    output.append({
        "seed": seed,
        "beta": beta,
        "cbr": cbr,
        "cases": np.array(cases),
        "fitted_beta": popt[0],
        "fitted_size": popt[1],
        "fitted_cbr": popt[2],
        "fitted_t0": popt[3],
        "births": np.array(model.nodes.births[:-1, 0])
    })

output = pd.DataFrame(output)


# In[6]:


plt.figure()
plt.plot(output["beta"], output["fitted_beta"], "o")
plt.xlim(0.00, 0.06)
plt.ylim(0.00, 0.06)
plt.figure()
plt.plot(output["beta"], (output["beta"] - output["fitted_beta"]) / output["fitted_beta"], "o")
plt.xlim(0.00, 0.06)
plt.ylim(-0.1, 0.10)
plt.figure()
plt.plot(output["cbr"], output["fitted_cbr"], "o")
plt.xlim(15, 50)
plt.ylim(15, 50)
plt.figure()
plt.plot(output["cbr"], (output["cbr"] - output["fitted_cbr"]) / output["fitted_cbr"], "o")
plt.xlim(15, 50)
plt.ylim(-0.2, 0.2)

# In[7]:


print(
    r"All fitted beta are within 10% of known beta: " + str(np.all(np.abs((output["beta"] - output["fitted_beta"]) / output["beta"]) < 0.10))
)
print(r"All fitted CBR are within 10% of known CBR: " + str(np.all(np.abs((output["cbr"] - output["fitted_cbr"]) / output["cbr"]) < 0.10)))

# In[8]:


output

# In[9]:


row = output.iloc[9]
cases_row = row["cases"]
fitted_beta = row["fitted_beta"]
fitted_size = row["fitted_size"]
fitted_cbr = row["fitted_cbr"]
fitted_t0 = row["fitted_t0"]

plt.figure()
plt.plot(cases_row, label="Case Trace")
plt.plot(SI_logistic(np.arange(nticks), fitted_beta, fitted_size, fitted_cbr, fitted_t0), label="Logistic Fit", ls="--")
plt.legend()
plt.xlabel("Time (days)")
plt.ylabel("Cases")
plt.title("Case Trace and Logistic Fit for Row 0")
# plt.ylim(9.8e5, 1e6)
plt.show()

# Just an interesting note - because I kicked the simulations off with 3 infections rather than 1 (to ensure that the initial infection doesn't die before infecting someone), we consistently get a negative fit value for t0, and one that is a larger negative value for the lowest β values.
