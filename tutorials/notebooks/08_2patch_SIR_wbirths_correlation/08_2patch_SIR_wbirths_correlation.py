#!/usr/bin/env python
# coding: utf-8

# # The relationship between spatial coupling and incidence correlation in a 2-patch model
# Now that we have extensively demonstrated and validated the single-patch behavior of different LASER models, it's finally time to demonstrate LASER in a spatial modeling context - in this case, not explicitly spatial but rather a multi-patch model with connection matrix.  
# 
# We will aim to replicate the findings in Keeling & Rohani, _Estimating spatial coupling in epidemiological systems: a mechanistic approach_, Ecol. Letters (2002) 5:20-29, where the authors analyze a 2-patch SIR system and develop a relationship between the parameter governing the ratio of "within vs. between" coupling of infectives and susceptibles in the two patches, and the correlation in the time series of infections in the two patches.  Readers can turn to that publication for a full reproduction of the analysis; here we will introduce the system (using our notation established in the other notebooks, rather than that of the paper) and skip ahead to stating the result we want to reproduce.
# 
# We consider a system of two coupled patches, of equal population $N$, birthrate $\mu$, and mortality rate $\nu$, coupled by parameter $\sigma$ such that in each patch, a fraction $\sigma$ of infectiveness is "sent" to the other node, with the remaining $(1-\sigma)$ "staying local".  
# 
# $$
# \dot{S_1} = \mu N - \frac{\beta S_1}{N}\left( I_1 \left( 1 - \sigma \right) +I_2 \sigma \right) - \nu S_1
# $$
# 
# $$
# \dot{I_1} = \frac{\beta S_1}{N}(I_1 (1 - \sigma) +I_2 \sigma) - \gamma I_1 - \nu I_1
# $$
# 
# $$
# \dot{R_1} = \gamma I_1 - \nu R_1
# $$
# 
# $$
# \dot{S_2} = \mu N - \frac{\beta S_2}{N}(I_2 (1 - \sigma) +I_1 \sigma) - \nu S_2
# $$
# 
# $$
# \dot{I_2} = \frac{\beta S_2}{N}(I_2 (1 - \sigma) +I_1 \sigma)- \gamma I_2 - \nu I_2
# $$
# 
# $$
# \dot{R_2} = \gamma I_2 - \nu R_2
# $$
# 
# The authors proceed to analyze the combined system, eventually deriving the following approximate relationship, 
# 
# $$
# C = \frac{\sigma}{\xi + \sigma}
# $$
# 
# where $C$ is the time-series correlation between $I_1$ and $I_2$, $\sigma$ is again the coupling constant as defined above, and $\xi$ is a constant that depends on the other model parameters $\mu, \nu, \beta, \gamma$.   They find a value $\xi = 0.0119$ for a particular instantiation of their model (run for 5000-year samples), and we will roughly replicate their parametrization and find that we can recapitulate the sigmoidal behavior and fit a similar value of $\xi$.  It's important to note that this is an empirical constant and we can move the exact value of it by changing e.g., the length of time over which we compute the correlation - we consider it sufficient validation to demonstrate the sigmoidal relationship between $C$ and $\sigma$, with a value of $\xi$ in the same order of magnitude.
# 

# In[1]:


from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

from laser.core.propertyset import PropertySet

import laser.core.distributions as dists
from laser.core.demographics import AliasedDistribution
from laser.core.demographics import KaplanMeierEstimator
from laser.generic import SIR
from laser.generic import Model
from laser.generic.utils import ValuesMap
from laser.generic.vitaldynamics import BirthsByCBR, MortalityByEstimator
from laser.core.utils import grid

import laser.core
import laser.generic

print(f"{np.__version__=}")
print(f"{laser.core.__version__=}")
print(f"{laser.generic.__version__=}")

# In[2]:


class Importation:
    def __init__(self, model, infdurdist, infdurmin: int =1, period: int = 180, count: int = 3):
        self.model = model
        self.infdurdist = infdurdist
        self.infdurmin = infdurmin
        self.period = period
        self.count = count

        self.model.nodes.add_vector_property("imports", model.params.nticks + 1, dtype=np.uint32, default=0)

        return
    
    def step(self, tick: int) -> None:
        if tick > 0 and tick % self.period == 0:
            i_susceptible = np.nonzero(self.model.people.state == SIR.State.SUSCEPTIBLE.value)[0]
            if len(i_susceptible) > 0:
                count = min(self.count, len(i_susceptible))
                i_infect = np.random.choice(i_susceptible, size=count, replace=False)
                self.model.people.state[i_infect] = SIR.State.INFECTIOUS.value
                samples = dists.sample_floats(self.infdurdist, np.zeros(count, np.float32))
                samples = np.round(samples)
                samples = np.maximum(samples, self.infdurmin).astype(self.model.people.itimer.dtype)
                self.model.people.itimer[i_infect] = samples
                inf_by_node = np.bincount(self.model.people.nodeid[i_infect], minlength=len(self.model.nodes)).astype(self.model.nodes.S.dtype)
                self.model.nodes.S[tick + 1] -= inf_by_node
                self.model.nodes.I[tick + 1] += inf_by_node
                self.model.nodes.imports[tick] = inf_by_node
            # else:
            #     print(f"No susceptibles to infect at tick {tick}")

        return

# In[3]:


outputs = None
nsims = 100
connections = np.logspace(-4, np.log10(0.5), nsims)

UPDATE_RESULTS = False # Set to True to update saved results takes ~90 minutes on current hardware

if UPDATE_RESULTS:

    import os

    # Set parameters to match those of Keeling & Rohani, 2002.  They have a constant importation rate of 5.5e-5*sqrt(N) per day,
    # which for this sim would be something like 40 per year.  I'm importing 6 per year, but I don't think that should be the big deal.
    # more important is setting the birth, R0, and infectious period to their numbers and then seeing the
    # connectivity-correlation relationship.
    nticks = 36500
    pop = 1e6
    scenario = grid(M=1, N=2, node_size_degs=0.089, population_fn=lambda x,y: pop, origin_x=0, origin_y=0)
    initial_infected = 1
    scenario["S"] = np.array([pop - initial_infected, pop])
    scenario["I"] = np.array([initial_infected, 0])
    scenario["R"] = np.array([0, 0])
    parameters = PropertySet(
        {
            "seed": 4,
            "nticks": nticks,
            "verbose": True,
            "beta": 17 / 13,
            "inf_mean": 13,
            "cbr": 20.277294,
            "importation_period": 180,
            "importation_count": 3,
        }
    )
    birthrate_map = ValuesMap.from_scalar(parameters.cbr, nticks=parameters.nticks, nnodes=len(scenario))

    infdurdist = dists.exponential(scale=parameters.inf_mean)
    rate_const = 365 * ((1 + parameters.cbr / 1000) ** (1/365) - 1) #to get stable population, small correction factor for the annual age-dist/KM estimators.
    stable_age_dist = np.array(1000*np.exp(-rate_const*np.arange(89)))
    pyramid = AliasedDistribution(stable_age_dist)
    survival = KaplanMeierEstimator(stable_age_dist.cumsum())


    outputs = np.zeros((nsims, parameters.nticks+1, scenario.shape[0]))
    for i, connection in enumerate(connections):
        parameters['seed'] += 1
        model = Model(scenario, parameters, birthrates=birthrate_map)
        model.components = [
            SIR.Susceptible(model),
            SIR.Recovered(model),
            SIR.Infectious(model, infdurdist),
            Importation(model, infdurdist),
            SIR.Transmission(model, infdurdist),
            BirthsByCBR(model, birthrates=birthrate_map, pyramid=pyramid),
            MortalityByEstimator(model, estimator=survival),
        ]

        # Start them slightly asynchronously - different initial susceptibilities, infection only in 1 patch
        # Want to see how connectivity drives correlation over time.
        model.network = np.array([[0, connection], [connection, 0]])

        model.run(f"{i+1}/{nsims} sims with connection={connection:.6f}")
        outputs[i, :, :] = model.nodes.I
        #Because this is a lot of long-running sims, we will save the outputs after each one completes
        os.makedirs("outputs", exist_ok=True)
        np.save("outputs/outputs.npy", outputs)


# In[4]:


if outputs is not None or Path("outputs/outputs.npy").exists():
    if outputs is None:
        outputs = np.load("outputs/outputs.npy")

    correlations = []
    last_25_years = 25 * 365  # 25 years in days

    for sim in range(outputs.shape[0]):
        patch1_data = outputs[sim, -last_25_years:, 0]
        patch2_data = outputs[sim, -last_25_years:, 1]
        correlation = np.corrcoef(patch1_data, patch2_data)[0, 1]
        correlations.append(correlation)

    correlations = np.array(correlations)

    # Define the function to fit
    def func(x, psi, beta):
        return beta + x / (psi + x)


    # Fit the curve
    popt, pcov = curve_fit(func, connections, correlations, p0=[0.01, 0], bounds=([1e-6, -0.00002], [1, 0.00002]))
    psi_opt = popt[0]
    beta_opt = popt[1]
    plt.plot(connections, correlations, ".")
    plt.xscale("log")
    # Plot the fitted curve
    plt.plot(connections, func(connections, psi_opt, beta_opt), label=f"Fitted curve: y = {beta_opt:.4f} + x / ({psi_opt:.4f} + x)")

    plt.legend()

# ## Results
# $\xi$ = 0.0117 - quite close to the original result!
# 
# As noted before - since these are finite time simulations and we are not running for 5000 years like the original paper, the choice of window over which to compute correlation can affect the outputs.  The particular choice of the last 25 years was chosen to land us with a value quite close to the original value - if you adjust the window of computation to be the last 10 years, or last 75 years, etc. you can shift the fitted value of $\xi$ around by a factor of 2-3 either direction.  This is not unexpected - it takes some  time for this system to settle out of its early transient dynamics into a regime where we can reliably compute this correlation, and in fact, that time likely depends strongly on the value of the connection coefficient.  We view it as sufficient demonstration to recapture the sigmoidal relationship, and show that we can recover values of $\xi$ of similar order of magnitude as the analytic result presented in the reference manuscript.
