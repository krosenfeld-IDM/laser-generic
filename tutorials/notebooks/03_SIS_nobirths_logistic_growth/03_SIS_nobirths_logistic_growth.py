#!/usr/bin/env python
# coding: utf-8

# # SIS model with no demographics
# 
# Moving on from the SI model to something slighlty more complex, we will add a state transition from infected back to susceptible, with no period of immunity - this is the SIS model.  It turns out that the equations governing this will be analogous to the SI model with demography - rather than births & mortality providing routes to add susceptibles and subtract infectives, this transition from infective -> susceptible will provide essentially the same mechanism. 
# 
# $$
# \dot{S} = -\frac{\beta*S*I}{N} + \gamma I
# $$
# 
# $$
# \dot{I} = \frac{\beta*S*I}{N} - \gamma I
# $$
# 
# As before, subbing $S = N-I$ into the second equation gives us
# 
# $$ \dot{I} = \beta I ( 1-\frac{\gamma}{\beta}-\frac{I}{N})$$
# 
# And comparing against the SI model with births, it is clear this equation is of the same form, with solution
# $$
# \frac{Nx}{1+(\frac{Nx}{I_0}-1)e^{-\beta x t}}
# $$
# 
# $$
# x = (1-\frac{\gamma}{\beta})
# $$
# 
# 
# This notebook tests the implementation and behavior of the model as follows:
# ## Construct the model
# In the first few cells, we do all the necessary imports.  Then we construct a single-patch LASER model with three components: `Susceptibility`, `Transmission`, and `Infection_SIS` - this component will require a new agent property `itimer`, and upon expiration of `itimer` agents will return to the susceptible state.  Finally, we initialize with a single infection and run.  The `Susceptibility` and `Transmission` components are previously described. 
# 
# 
# ## Sanity check
# The first test ensures certain basic constraints are being obeyed by the model.  We confirm that at each timestep, $S_t=N_t-I_t$. 
# 
# ## Scientific test
# Finally, we come to the scientific test.  As before, we first test on a single instance of the model and show that the expected output is recovered.  Then, we select a few values of $\beta$ and $\gamma$, run the model, fit the outputs to the logistic equation, and compare the fitted value of $\beta$ and $\gamma$ to the known values; all of the considerations noted in the SI with births model, in terms of how to approach this fit, are echoed again here.  Of particular concern is the approximation of an exponential transition from infected back to susceptible - as we are doing a first-order finite timestep integration here, that approximation will probably produce an error linear in $\gamma \Delta t$ between the analytic result and the modeled result.  In fact, in a lot of real disease models, we have compartment dwell times in the exposed and infective states that are only a handful of $\Delta t$ long, but when doing real epi modeling and calibrating model parameters to uncertain data, this is generally not likely to be a dominant source of bias, uncertainty, etc.  But when comparing specifically against an analytic result, it can become significant.  
# 
# 

# In[1]:


import matplotlib.pyplot as plt
import numba as nb
import numpy as np
import pandas as pd
from laser.core.propertyset import PropertySet
from scipy.optimize import curve_fit
from scipy.optimize import minimize

import laser.core
import laser.generic

print(f"{np.__version__=}")
print(f"{laser.core.__version__=}")
print(f"{laser.generic.__version__=}")


# In[2]:


import laser.core.distributions as dists
import laser.generic.SIS as SIS
from laser.generic import Model
from laser.core.utils import grid

# To make sure we don't accumulate lots of finite time-step error, make inf mean quite long in units of timestep

# In[3]:


pop = 3e5

scenario = grid(M=1, N=1, population_fn=lambda x,y: pop, origin_x=0, origin_y=0)
initial_infected = 1
scenario["S"] = scenario.population - initial_infected
scenario["I"] = initial_infected
parameters = PropertySet({"prng_seed": 42, "nticks": 3000, "beta": 0.1, "inf_mean": 32})

infdurdist = dists.exponential(scale=parameters.inf_mean)


# In[4]:


# Run until we get an outbreak
outbreak = False
while not outbreak:
    parameters.prng_seed += 1
    model = Model(scenario, parameters)

    model.components = [SIS.Susceptible(model), SIS.Infectious(model, infdurdist), SIS.Transmission(model, infdurdist)]

    model.run()
    outbreak = np.any(model.nodes.I[200] > 0)


# ## Sanity checks
# Check that the relationships between susceptible, infected, and total population hold.
# 
# 

# In[5]:


fig, axs = plt.subplots(1, 2, figsize=(18, 5))

# Panel 2: Susceptible over time
axs[0].plot(model.nodes.S[1:], lw=4)
axs[0].plot(model.nodes.S[:-1] + model.nodes.newly_recovered[:-1] - model.nodes.newly_infected[:-1])
axs[0].set_yscale("log")
axs[0].set_title("Susceptible over time")
axs[0].set_xlabel("Time")
axs[0].set_ylabel("Susceptible")

# Panel 3: Population minus cumulative infections (incidence)
axs[1].plot(model.nodes.I[1:], lw=4)
axs[1].plot(model.nodes.I[:-1] - model.nodes.newly_recovered[:-1] + model.nodes.newly_infected[:-1])
axs[1].set_yscale("log")
axs[1].set_title("Population minus cumulative infections")
axs[1].set_xlabel("Time")
axs[1].set_ylabel("Incidence")

plt.tight_layout()

print("S[t] = S[t-1] + recovered[t-1] - incidence[t-1]: ", np.allclose(model.nodes.S[1:], model.nodes.S[:-1] + model.nodes.newly_recovered[:-1] - model.nodes.newly_infected[:-1]))
print("I[t] = I[t-1] - recovered[t-1] + incidence[t-1]: ", np.allclose(model.nodes.I[1:], model.nodes.I[:-1] - model.nodes.newly_recovered[:-1] + model.nodes.newly_infected[:-1]))


# ### Single-simulation check
# As before, starting with a single infection induces some stochasticity in terms of when the outbreak really starts to take off, and so we fit the expected behavior with a free offset parameter below.

# In[6]:


def SIS_logistic(t, beta, popsize, gamma, t0):
    x = 1 - gamma / beta
    return popsize * x / (1 + (popsize * x - 1) * np.exp(-beta * x * (t - t0)))


t = np.arange(model.params.nticks+1)


def objective(t0):
    return np.sum(
        (1 - SIS_logistic(t, model.params.beta, pop, 1 / model.params.inf_mean, t0) / np.squeeze(model.nodes.I)) ** 2
    )


result = minimize(objective, x0=10)
t0_opt = result.x[0]

plt.plot(model.nodes.I, lw=4)
plt.plot(SIS_logistic(t, model.params.beta, pop, 1 / model.params.inf_mean, 0), lw=3)
plt.plot(SIS_logistic(t, model.params.beta, pop, 1 / model.params.inf_mean, t0_opt), "r:", lw=3)
plt.yscale("log")
plt.legend(["Model output", "Logistic growth with known inputs, t0=0", f"Logistic growth with known inputs, best-fit t0 = {t0_opt:.1f}"])

# ## Scientific testing
# Finally, we run the model for a range of $\beta$ & $\gamma$  parameters, we freely fit the model output to the logistic equation, and we compare the known input parameters against the parameters fitted from output.  
# 
# We will use only relatively large values of $\gamma$ for this procedure.  The reason why will become clear in a second test, where we demonstrate that there is an error between the expected final size and the modeled final size, and that this error shrinks with $\gamma \Delta t$, as we would expect a first-order approximation error to.  
# 
# To make this a pass-fail test, we will raise a flag if the fitted parameters are more than 5% different than the known ones.

# In[7]:


NTESTS = 10
nticks = 730
t = np.arange(nticks)
betarange = [0.02, 0.1]
gammarange = [1 / 300, 1 / 100]
seeds = list(range(NTESTS))
pop = 1e5
betas = np.random.uniform(betarange[0], betarange[1], NTESTS)
gammas = np.random.uniform(gammarange[0], gammarange[1], NTESTS)
scenario = grid(M=1, N=1, population_fn=lambda x,y: pop, origin_x=0, origin_y=0)
initial_infected = 3
scenario["S"] = scenario.population - initial_infected
scenario["I"] = initial_infected

output = []
for i, (seed, beta, gamma) in enumerate(zip(seeds, betas, gammas)):
    parameters = PropertySet({"prng_seed": seed, "nticks": nticks, "verbose": True, "beta": beta, "inf_mean": 1 / gamma})

    model = Model(scenario, parameters)

    infdurdist = dists.exponential(scale=parameters.inf_mean)

    model.components = [SIS.Susceptible(model), SIS.Infectious(model, infdurdist), SIS.Transmission(model, infdurdist)]

    model.run(label=f"SIS {i+1:2} of {NTESTS}, seed={seed}, beta={beta:.3f}, gamma={gamma:.5f}")
    cases = model.nodes.I[1:,0]
    popt, pcov = curve_fit(
        SIS_logistic,
        t,
        cases,
        p0=[np.mean(betarange), pop, np.mean(gammarange), 1],
        bounds=([betarange[0] / 2, pop - 1, gammarange[0] / 2, -300], [betarange[1] * 2, pop + 1, gammarange[1] * 2, 300]),
    )

    output.append(
        {
            "seed": seed,
            "beta": beta,
            "gamma": gamma,
            "cases": [np.array(cases)],
            "fitted_beta": popt[0],
            "fitted_gamma": popt[2],
            "fitted_t0": popt[3],
            "recovered": [np.array(model.nodes.newly_recovered[1:,0])]
        }
    )

output = pd.DataFrame(output)


# In[8]:


plt.figure()
plt.plot(output["beta"], output["fitted_beta"], "o")
plt.xlim(betarange[0], betarange[1])
plt.ylim(betarange[0], betarange[1])
plt.figure()
plt.plot(output["beta"], 1 - output["beta"] / output["fitted_beta"], "o")
plt.xlim(betarange[0], betarange[1])
plt.ylim(-0.25, 0.25)
plt.figure()
plt.plot(output["gamma"], output["fitted_gamma"], "o")
plt.xlim(gammarange[0]*0.9, gammarange[1]*1.1)
plt.ylim(gammarange[0]*0.9, gammarange[1]*1.1)
plt.figure()
plt.plot(output["gamma"], 1 - output["gamma"] / output["fitted_gamma"], "o")
plt.xlim(gammarange[0]*0.9, gammarange[1]*1.1)
plt.ylim(-0.25, 0.25)

# In[9]:


print(
    "All fitted beta are within 10% of known beta: " + str(np.all(np.abs((output["beta"] - output["fitted_beta"]) / output["beta"]) < 0.10))
)
print(
    "All fitted gamma are within 20% of known gamma: "
    + str(np.all(np.abs((output["gamma"] - output["fitted_gamma"]) / output["gamma"]) < 0.20))
)

# In[10]:


output

# ### Quick demonstration of first-order error accumulation
# As noted before, for $\gamma \Delta t$ large, first-order integration like we are doing here can accumulate substantial error.  Calculating exactly how error will accumulate in an integrator for a dynamic process like this is beyond the scope here, and probably depends on a lot of details.  E.g., the ordering of steps - in a given step, does the `transmission` update from S->I state occur before or after the `infection` update that sends agents from I->S?  Do we use midpoint methods, timer countdowns, or take advantage of the unique memorylessness of the exponential distribution to simply remove a random fraction each time?  All that is beyond scope here, but just want to demonstrate that the error in the equilibrium value $I(t \rightarrow \infty)$ becomes large when the mean infectious period $\frac{1}{\gamma}$ is on the same order as $\Delta t$

# In[11]:


# %%capture

gammas = [1 / infmean for infmean in [1, 1.5, 2, 2.5, 3, 6, 12, 18, 30, 45, 60, 90, 120, 180, 240, 300]]
betas = [3 * gamma for gamma in gammas]

NTESTS = len(gammas)
nticks = 3000
seeds = list(range(NTESTS))
pop = 1e5
final_expected = np.array([])
final_observed = np.array([])
scenario = grid(M=1, N=1, population_fn=lambda x,y: pop, origin_x=0, origin_y=0)
initial_infected = 20
scenario["S"] = scenario.population - initial_infected
scenario["I"] = initial_infected

for i, (seed, beta, gamma) in enumerate(zip(seeds, betas, gammas)):
    parameters = PropertySet({"prng_seed": seed, "nticks": nticks, "verbose": True, "beta": beta, "inf_mean": 1 / gamma})

    model = Model(scenario, parameters)

    infdurdist = dists.exponential(scale=parameters.inf_mean)

    model.components = [SIS.Susceptible(model), SIS.Infectious(model, infdurdist), SIS.Transmission(model, infdurdist)]

    model.run(label=f"SIS {i+1:2} of {NTESTS}, seed={seed:2}, beta={beta:.3f}, gamma={gamma:.5f}")
    final_observed = np.append(final_observed, model.nodes.I[-1,0])
    final_expected = np.append(final_expected, pop * (1 - gamma / beta))

# In[ ]:


plt.plot(gammas, np.abs(1 - final_observed / final_expected), "o")
plt.xlabel(r"$\gamma$")
plt.ylabel("$| 1 - \\frac{I(\\infty)_{obs}}{I(\\infty)_{exp}} |$")
plt.title(r"Error in equilibrium infected fraction increases roughly linearly in $\gamma \Delta t$")
plt.show()

# In[13]:


# Validate for each row that mean(recovered / cases) is close to gamma
for idx, row in output.iterrows():
    cases_arr = np.array(row["cases"][0])
    recovered_arr = np.array(row["recovered"][0])
    # Avoid division by zero
    valid = cases_arr > 0
    avg_ratio = np.mean(recovered_arr[valid] / cases_arr[valid])
    print(f"Row {idx}: gamma={row['gamma']:.6f}, avg(recovered/cases)={avg_ratio:.6f}, diff={abs(avg_ratio - row['gamma']):.6f}")

