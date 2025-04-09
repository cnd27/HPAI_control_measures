import model
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
np.set_printoptions(legacy='1.25')

seed = 0
np.random.seed(seed)
pre_load = True
data_start = pd.to_datetime('2022-10-01')
data_end = pd.to_datetime('2023-09-30')

# Load and format the data
data = model.Data('../data/premises_data_model',
                  '../data/case_data_',
                  '../data/matched_premises,
                  '../data/NUTS1_Jan_2018_SGCB_in_the_UK.shp',
                  '../data/CTYUA_MAY_2023_UK_BGC.shp',
                  date_start=data_start, date_end=data_end, grid_number=20, adapt=False, select_region=None)

# Choose model options
model0 = model.Model(data, transmission_type=1, kernel_type='cauchy', spatial=False, combine=False, cull_data=False)
max_iter = 211000
chains = [0, 1]

# Run the MCMC
if pre_load:
    model0.load_chains(chain_nums=chains, max_iter=max_iter, to_fit=None)
else:
    model.run_mcmc(chain_num=seed, max_iter=max_iter, first_iter=10000, burn_in=11000, ind_update=True,
                   prop_updates=0.05, save=True)
# Format the posterior samples
model0.format_posts(burn_in=11000, values_per_chain=1000)

# Run the projections
if pre_load:
    model0.load_projections(reps=10000, max_iter=max_iter, date_start=data_start, date_end=data_end)
else:
    model0.simulate_model(reps=10000)
    model0.save_projections(max_iter=max_iter)

base_all = np.sum(model0.notified_day >= 0, axis=1)
base_50 = np.median(np.sum(model0.notified_day >= 0, axis=1))
levels = np.tile(np.array([0.2, 0.4, 0.6, 0.8]), 20)
durations = np.tile(np.repeat(np.array([7, 14, 21, 28]),4), 5)
zones = [x for x in [5, 10, 15, 'county', 'region'] for _ in range(16)]

# Run enhanced control projections
if pre_load:
    control_all = np.load('../data/control_all.npy')
    control_50 = np.load('../data/control_50.npy')
    control_975 = np.load('../data/control_975.npy')
    control_025 = np.load('../data/control_025.npy')
else:
    final_size_50 = np.zeros(80)
    final_size_975 = np.zeros(80)
    final_size_025 = np.zeros(80)
    final_size_all = np.zeros((10000,80))

    for d in range(len(levels)):
        level = levels[d]
        duration = durations[d]
        zone = zones[d]
        data = model.Data('../data/premises_data_model',
                          '../data/case_data_',
                          '../data/matched_premises,
                          '../data/NUTS1_Jan_2018_SGCB_in_the_UK.shp',
                          '../data/CTYUA_MAY_2023_UK_BGC.shp',
                          date_start=data_start, date_end=data_end, grid_number=20, adapt=False, select_region=None)
        modelb = model.Model(data, transmission_type=1, kernel_type='cauchy', spatial=False, combine=False, cull_data=False)
        max_iter = 211000
        chains = [0, 1]
        modelb.load_chains(chain_nums=chains, max_iter=max_iter, to_fit=None)
        modelb.format_posts(burn_in=11000, values_per_chain=1000)
        modelb.simulate_model(reps=1000, include_total=False, biosecurity_level=level, biosecurity_zone=zone, biosecurity_duration=duration)

        final_size_50[d] = np.median(np.sum(modelb.notified_day >= 0, axis=1))
        final_size_975[d] = np.percentile(np.sum(modelb.notified_day >= 0, axis=1), 97.5)
        final_size_025[d] = np.percentile(np.sum(modelb.notified_day >= 0, axis=1), 2.5)
        final_size_all[:,d] = np.sum(modelb.notified_day >= 0, axis=1)

    control_all = final_size_all.reshape(10000, 5,4,4)
    control_50 = final_size_50.reshape(5,4,4)
    control_975 = final_size_975.reshape(5,4,4)
    control_025 = final_size_025.reshape(5,4,4)


# Plot the figures
model0.plot_figure_1()
model0.plot_figure_2()
model0.plot_figure_3(control_all=control_all, base_all=base_all)
model0.plot_figure_4(base_50=base_50, control_50=control_50, control_975=control_975, control_025=control_025)
plt.show()
