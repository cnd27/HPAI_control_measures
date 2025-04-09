"""Module for individual-based compartmental infection model for farms with HPAI."""
import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.ticker import MaxNLocator
from matplotlib.gridspec import GridSpec
from matplotlib import cm
from matplotlib.path import Path
from matplotlib.patches import Rectangle, PathPatch
import warnings
import copy
import calendar
import geopandas as gpd
from shapely.geometry import Point
import openpyxl


class Data:
    """Class for loading and storing data."""
    def __init__(self, file_path_farms, file_path_cases=None, file_path_match=None, file_path_regions=None, file_path_counties=None,
                 file_path_grid=None, date_start=None, date_end=None, past_date_start=None, grid_number=None, grid_size=None, adapt=True,
                 select_region=None):
        self.file_path_farms = file_path_farms
        self.file_path_cases = file_path_cases
        self.file_path_match = file_path_match
        self.file_path_regions = file_path_regions
        self.file_path_counties = file_path_counties
        self.file_path_grid = file_path_grid
        self.date_start = date_start
        self.date_end = date_end
        self.past_date_start = past_date_start
        if self.past_date_start is None:
            self.past_date_start = self.date_start
        self.grid_number = grid_number
        self.adapt = adapt
        self.select_region = select_region
        data_farms = np.loadtxt(self.file_path_farms)
        self.location_x = data_farms[:, 2]
        self.location_y = data_farms[:, 3]
        self.location_tree  = cKDTree(np.column_stack((self.location_x, self.location_y)))
        self.species_pop = data_farms[:, 4:].T
        self.n_species = self.species_pop.shape[0]
        self.mean_farm_size = np.mean(self.species_pop, axis=1)
        self.pop_over_mean = self.species_pop / self.mean_farm_size[:, np.newaxis]
        self.n_farms = len(self.location_x)
        self.grid_size = grid_size
        if not adapt:
            if ((grid_number is not None) and (grid_size is not None)) or ((grid_size is None) and (grid_number is None)):
                raise ValueError("Either grid_number or grid_size must be specified.")

        # Load the regions and counties
        if file_path_regions is not None:
            try:
                polygons = gpd.read_file(file_path_regions)
                polygons = polygons[:-1]
                points_list = [Point(1000*self.location_x[i], 1000*self.location_y[i]) for i in range(self.n_farms)]
                points_gdf = gpd.GeoDataFrame(geometry=points_list, crs=polygons.crs)
                points_with_polygons = gpd.sjoin(points_gdf, polygons, how='left', predicate='within')
                self.region = points_with_polygons.index_right.values
                nan_regions = np.where(np.isnan(self.region))[0]
                for i in range(np.sum(np.isnan(self.region))):
                    self.region[nan_regions[i]] = np.argmin(polygons.distance(points_list[nan_regions[i]]))
                self.region = self.region.astype(int)
                self.region_names = ['North East', 'North West', 'Yorkshire and the Humber', 'East Midlands',
                                     'West Midlands', 'East of England', 'London', 'South East', 'South West', 'Wales',
                                     'Scotland']
                self.region_names_ab = ['NE', 'NW', 'Y&H', 'EM', 'WM', 'EoE', 'LDN', 'SE', 'SW', 'WAL', 'SCT']
                self.polygons = polygons
            except FileNotFoundError:
                print("File '" + file_path_regions + "' not found.")
        if file_path_counties is not None:
            try:
                polygons = gpd.read_file(file_path_counties)
                polygons = polygons
                points_list = [Point(1000*self.location_x[i], 1000*self.location_y[i]) for i in range(self.n_farms)]
                points_gdf = gpd.GeoDataFrame(geometry=points_list, crs=polygons.crs)
                points_with_polygons = gpd.sjoin(points_gdf, polygons, how='left', predicate='within')
                self.county = points_with_polygons.index_right.values
                nan_counties = np.where(np.isnan(self.county))[0]
                for i in range(np.sum(np.isnan(self.county))):
                    self.county[nan_counties[i]] = np.argmin(polygons.distance(points_list[nan_counties[i]]))
                self.county = self.county.astype(int)
                self.county_names = polygons.CTYUA23NM.tolist()
                self.polygons_c = polygons
                self.county_names = np.delete(self.county_names, np.s_[153:164])
                self.polygons_c = self.polygons_c.drop(index=range(153, 164))
                self.polygons_c = self.polygons_c.reset_index(drop=True)
                self.county[self.county > 153] -= 11
            except FileNotFoundError:
                print("File '" + file_path_regions + "' not found.")

        if select_region is not None:
            self.location_x = self.location_x[self.region == select_region]
            self.location_y = self.location_y[self.region == select_region]
            self.species_pop = self.species_pop[:, self.region == select_region]
            self.pop_over_mean = self.pop_over_mean[:, self.region == select_region]
            self.n_farms = len(self.location_x)
            self.included_farms = np.arange(len(self.region))[self.region == select_region]
            self.region = self.region[self.region == select_region]

        # Get edges of UK coordinates
        min_x = np.min(self.location_x)
        max_x = np.max(self.location_x)
        min_y = np.min(self.location_y)
        max_y = np.max(self.location_y)
        uk_length = max(max_x - min_x, max_y - min_y)
        if self.file_path_grid is None:
            if self.adapt:
                lambda_hat = self.n_farms / (self.grid_number ** 2)
                self.grid_size = np.array([uk_length])
                if max_x - min_x > max_y - min_y:
                    self.grid_location_x = np.array([min_x])
                    self.grid_location_y = np.array([min_y - ((max_x - min_x) - (max_y - min_y)) / 2])
                else:
                    self.grid_location_x = np.array([min_x - ((max_y - min_y) - (max_x - min_x)) / 2])
                    self.grid_location_y = np.array([min_y])
                self.farms_in_grid = np.array([self.n_farms])
                continue_adapt = np.array([True])
                while np.sum(continue_adapt) > 0:
                    continue_adapt_idx = np.where(continue_adapt)[0]
                    for i in range(len(continue_adapt_idx)):
                        current_eq = (np.log(self.farms_in_grid[continue_adapt_idx[i]]) - np.log(lambda_hat)) ** 2
                        tmp_grid_size = np.tile(self.grid_size[continue_adapt_idx[i]] / 2, 4)
                        tmp_grid_location_x = np.tile(self.grid_location_x[continue_adapt_idx[i]], 4) + np.array(
                            [0, 1, 0, 1]) * self.grid_size[continue_adapt_idx[i]] / 2
                        tmp_grid_location_y = np.tile(self.grid_location_y[continue_adapt_idx[i]], 4) + np.array(
                            [0, 0, 1, 1]) * self.grid_size[continue_adapt_idx[i]] / 2
                        tmp_farms_in_grid = np.zeros(4)
                        for j in range(4):
                            tmp_farms_in_grid[j] = np.sum((tmp_grid_location_x[j] <= self.location_x) & (
                                        self.location_x < tmp_grid_location_x[j] + tmp_grid_size[j]) &
                                                          (tmp_grid_location_y[j] <= self.location_y) & (
                                                                      self.location_y < tmp_grid_location_y[j] +
                                                                      tmp_grid_size[j]))
                        new_eq = np.sum(
                            (np.log(tmp_farms_in_grid[tmp_farms_in_grid > 0]) - np.log(lambda_hat)) ** 2) / np.sum(
                            tmp_farms_in_grid > 0)
                        if current_eq <= new_eq:  # or self.grid_size[continue_adapt_idx[i]] / 2 < 10:
                            continue_adapt[continue_adapt_idx[i]] = False
                        else:
                            self.grid_size = np.delete(self.grid_size, continue_adapt_idx[i])
                            self.grid_size = np.insert(self.grid_size, continue_adapt_idx[i],
                                                       tmp_grid_size[tmp_farms_in_grid > 0])
                            self.grid_location_x = np.delete(self.grid_location_x, continue_adapt_idx[i])
                            self.grid_location_x = np.insert(self.grid_location_x, continue_adapt_idx[i],
                                                             tmp_grid_location_x[tmp_farms_in_grid > 0])
                            self.grid_location_y = np.delete(self.grid_location_y, continue_adapt_idx[i])
                            self.grid_location_y = np.insert(self.grid_location_y, continue_adapt_idx[i],
                                                             tmp_grid_location_y[tmp_farms_in_grid > 0])
                            self.farms_in_grid = np.delete(self.farms_in_grid, continue_adapt_idx[i])
                            self.farms_in_grid = np.insert(self.farms_in_grid, continue_adapt_idx[i],
                                                           tmp_farms_in_grid[tmp_farms_in_grid > 0])
                            continue_adapt = np.delete(continue_adapt, continue_adapt_idx[i])
                            continue_adapt = np.insert(continue_adapt, continue_adapt_idx[i],
                                                       np.ones(np.sum(tmp_farms_in_grid > 0), dtype=bool))
                self.n_grids = len(self.grid_size)
                farm_grid_tmp = (((self.grid_location_x <= self.location_x[:, np.newaxis]) &
                                  (self.grid_location_x + self.grid_size > self.location_x[:, np.newaxis])) &
                                 ((self.grid_location_y <= self.location_y[:, np.newaxis]) &
                                  (self.grid_location_y + self.grid_size > self.location_y[:, np.newaxis])))
                self.farm_grid = np.argmax(farm_grid_tmp, axis=1)
                self.dist2 = (np.maximum(0, np.abs(self.grid_location_x[:, np.newaxis] - self.grid_location_x) -
                                         (self.grid_size[:, np.newaxis] / 2 + self.grid_size / 2)) ** 2 +
                              np.maximum(0, np.abs(self.grid_location_y[:, np.newaxis] - self.grid_location_y) -
                                         (self.grid_size[:, np.newaxis] / 2 + self.grid_size / 2)) ** 2)
            else:
                if self.grid_number is not None:
                    edges_x = np.linspace(min_x, min_x + uk_length, grid_number + 1)
                    edges_y = np.linspace(min_y, min_y + uk_length, grid_number + 1)
                else:
                    edges_x = np.arange(min_x, min_x + uk_length + grid_size, grid_size)
                    edges_y = np.arange(min_y, min_y + uk_length + grid_size, grid_size)
                grid_x = np.digitize(self.location_x, edges_x) - 1
                grid_y = np.digitize(self.location_y, edges_y) - 1
                base_grid = grid_x + np.max(grid_x + 1) * grid_y
                unique_base_grid = np.unique(base_grid)
                grid_numbers = np.arange(0, len(unique_base_grid))
                if self.grid_number is not None:
                    self.grid_size = (edges_x[1] - edges_x[0]) * np.ones(len(unique_base_grid))
                else:
                    self.grid_size = grid_size * np.ones(len(unique_base_grid))
                # Assign farms to grid.
                farms_in_base_grid = np.bincount(base_grid)
                birds_in_base_grid = np.bincount(base_grid, weights=np.sum(self.species_pop, axis=0))
                birds_in_base_grid_1 = np.bincount(base_grid, weights=self.species_pop[0, :])
                birds_in_base_grid_2 = np.bincount(base_grid, weights=self.species_pop[1, :])
                birds_in_base_grid_3 = np.bincount(base_grid, weights=self.species_pop[2, :])
                self.farms_in_grid = farms_in_base_grid[farms_in_base_grid > 0]
                self.birds_in_grid = birds_in_base_grid[birds_in_base_grid > 0]
                self.birds_in_grid_1 = birds_in_base_grid_1[birds_in_base_grid > 0]
                self.birds_in_grid_2 = birds_in_base_grid_2[birds_in_base_grid > 0]
                self.birds_in_grid_3 = birds_in_base_grid_3[birds_in_base_grid > 0]
                self.grid_location_x = min_x + self.grid_size * (np.mod(unique_base_grid, np.max(grid_x + 1)))
                self.grid_location_y = min_y + self.grid_size * (np.floor(unique_base_grid / np.max(grid_x + 1)))
                self.farm_grid = np.array([grid_numbers[np.where(unique_base_grid == val)][0] for val in base_grid])
                self.n_grids = len(unique_base_grid)
                all_grid_x = self.grid_location_x[:, np.newaxis] - self.grid_location_x
                all_grid_y = self.grid_location_y[:, np.newaxis] - self.grid_location_y
                self.dist2 = np.maximum(0, np.abs(all_grid_x) - self.grid_size) ** 2 + np.maximum(0, np.abs(
                    all_grid_y) - self.grid_size) ** 2
        else:
            # Load the grid from a file
            data_grid = np.load(self.file_path_grid)
            self.grid_size = data_grid.grid_size
            self.farms_in_grid = data_grid.farms_in_grid
            self.birds_in_grid = data_grid.birds_in_grid
            self.birds_in_grid_1 = data_grid.birds_in_grid_1
            self.birds_in_grid_2 = data_grid.birds_in_grid_2
            self.birds_in_grid_3 = data_grid.birds_in_grid_3
            self.grid_location_x = data_grid.grid_location_x
            self.grid_location_y = data_grid.grid_location_y
            self.farm_grid = data_grid.farm_grid
            self.n_grids = data_grid.n_grids
            self.dist2 = data_grid.dist2
        self.dist2[self.dist2 < 1] = 0
        if self.file_path_cases is not None:
            data_cases = pd.read_excel(self.file_path_cases)
            self.matched_farm = np.loadtxt(file_path_match)
            # Convert dates to days since reference date.
            self.end_day = (self.date_end - self.date_start).days
            self.past_start_day = (self.past_date_start - self.date_start).days
            self.report_day = np.array((pd.to_datetime(data_cases['ReportDate']) - self.date_start).dt.days.values)
            self.culled_day = np.array((pd.to_datetime(data_cases['ConfirmationDate']) - self.date_start).dt.days.values)
            self.culled_day[self.report_day > self.culled_day] = self.report_day[self.report_day > self.culled_day]
            self.cull_times = self.culled_day - self.report_day
            # Remove farms that are reported after the end date
            data_cases = data_cases[(self.report_day <= self.end_day)]
            self.matched_farm = self.matched_farm[(self.report_day <= self.end_day)].astype(int)
            self.culled_day = self.culled_day[(self.report_day <= self.end_day)]
            self.cull_times = self.cull_times[(self.report_day <= self.end_day)]
            self.report_day = self.report_day[(self.report_day <= self.end_day)]
            # Remove secondary infections of farms in the time period
            unique_farms, counts = np.unique(self.matched_farm, return_counts=True)
            non_unique_farms = unique_farms[np.where(counts > 1)[0]]
            if len(non_unique_farms) > 0:
                remove_indices = np.concatenate(
                    [np.where(self.matched_farm == value)[0][:-1] for value in non_unique_farms if
                     value in self.matched_farm])
                keep_indices = np.setdiff1d(np.arange(len(self.matched_farm)), remove_indices)
                data_cases = data_cases.iloc[keep_indices]
                self.matched_farm = self.matched_farm[keep_indices]
                self.report_day = self.report_day[keep_indices]
                self.culled_day = self.culled_day[keep_indices]
                self.cull_times = self.cull_times[keep_indices]
            self.date_report = pd.to_datetime(data_cases['ReportDate'])
            self.date_culled= pd.to_datetime(data_cases['ConfirmationDate'])
            self.n_data = np.sum(self.report_day >= 0)
        if select_region is not None:
            include_inf_region = np.isin(self.matched_farm, self.included_farms)
            self.matched_farm = np.array([np.where(self.included_farms == value)[0][0] for value in self.matched_farm[include_inf_region]])
            self.report_day = self.report_day[include_inf_region]
            self.culled_day = self.culled_day[include_inf_region]
            self.cull_times = self.cull_times[include_inf_region]
            data_cases = data_cases[include_inf_region]
            self.date_report = pd.to_datetime(data_cases['ReportDate'])
            self.date_culled= pd.to_datetime(data_cases['ConfirmationDate'])
            self.n_data = np.sum(self.report_day >= 0)

    def plot_the_grid(self):
        """Plot the grid across the UK."""
        fig, ax = plt.subplots(figsize=(5, 8))
        ax.scatter(self.location_x, self.location_y, s=3, edgecolors="w", linewidth=0.1)
        for i in range(self.n_grids):
            ax.add_patch(plt.Rectangle((self.grid_location_x[i], self.grid_location_y[i]), self.grid_size[i],
                                       self.grid_size[i], fill=False, edgecolor='r', lw=0.5))
        ax.set_xticks([])
        ax.set_yticks([])

    def plot_counties_regions(self):
        """Plot the counties and regions across the UK."""
        fig, ax = plt.subplots(1, 2, figsize=(12, 8))

        colors = plt.cm.get_cmap("Set3", 12)
        self.polygons_c["color"] = [colors(i % 12) for i in range(len(self.polygons_c))]
        adjacency = []
        for i, poly1 in enumerate(self.polygons_c.geometry):
            for j, poly2 in enumerate(self.polygons_c.geometry):
                if i < j:
                    if poly1.intersects(poly2):
                        adjacency.append((i, j))
        for i in range(len(self.polygons_c)):
            pairs = [pair[1] if pair[0] == i else pair[0] for pair in adjacency if i in pair]
            for j in pairs:
                if self.polygons_c["color"][i] == self.polygons_c["color"][j]:
                    current_colors = np.where(np.sum(np.isin(colors.colors, self.polygons_c["color"][pairs].tolist()), axis=1) == 4)[0]
                    color_options = np.delete(np.arange(12), current_colors)
                    new_color = colors(np.random.choice(color_options))
                    self.polygons_c.at[i, "color"] = new_color
                    break
        self.polygons_c.plot(ax=ax[0], color=self.polygons_c["color"], edgecolor="black", linewidth=0.5)

        ax1_pos = ax[0].get_position()
        ax[0].set_ylim(0, 1.08e6)
        inset_x = ax1_pos.width * 8.6e4 / (ax[0].get_xlim()[1] - ax[0].get_xlim()[0])
        inset_y = ax1_pos.height * 1.3e5 / (ax[0].get_ylim()[1] - ax[0].get_ylim()[0])
        inset_ax = fig.add_axes(
            [ax1_pos.x0 + ax1_pos.width - inset_x + 0.014, ax1_pos.y0 + ax1_pos.height - inset_y + 0.069, inset_x, inset_y])
        self.polygons_c.plot(ax=inset_ax, color=self.polygons_c["color"], edgecolor="black", linewidth=0.5)
        inset_ax.set_xlim([3.9e5, 4.76e5])
        inset_ax.set_ylim([1.1e6, 1.23e6])
        inset_ax.set_xticks([])
        inset_ax.set_yticks([])

        colors = plt.cm.get_cmap("tab20", 11)
        self.polygons["color"] = [colors(i) for i in range(11)]
        self.polygons.plot(ax=ax[1], color=self.polygons["color"], edgecolor="black", linewidth=0.5)

        ax2_pos = ax[1].get_position()
        ax[1].set_ylim(0, 1.08e6)
        inset_x = ax2_pos.width * 8.6e4 / (ax[1].get_xlim()[1] - ax[1].get_xlim()[0])
        inset_y = ax2_pos.height * 1.3e5 / (ax[1].get_ylim()[1] - ax[1].get_ylim()[0])
        inset_ax = fig.add_axes(
            [ax2_pos.x0 + ax2_pos.width - inset_x + 0.078, ax2_pos.y0 + ax2_pos.height - inset_y + 0.069, inset_x, inset_y])
        self.polygons.plot(ax=inset_ax, color=self.polygons["color"], edgecolor="black", linewidth=0.5)
        inset_ax.set_xlim([3.9e5, 4.76e5])
        inset_ax.set_ylim([1.1e6, 1.23e6])
        inset_ax.set_xticks([])
        inset_ax.set_yticks([])

        ax[0].set_title("Counties of Great Britain")
        ax[1].set_title("Regions of Great Britain")
        ax[0].set_xticks([])
        ax[0].set_yticks([])
        ax[1].set_xticks([])
        ax[1].set_yticks([])

        plt.tight_layout()

    def bird_data_hist(self, c_i=95, birds=True, print_fig=None, best=10, rain_cloud=True):

        fig, ax = plt.subplots(3, 4, figsize=(12, 10), gridspec_kw={'height_ratios': [1.5, 1, 1.5], 'width_ratios': [1, 1, 1, 1.4]})
        gs = ax[0, 0].get_gridspec()
        for i in range(3):
            for j in range(4):
                ax[i, j].remove()
        ax1 = fig.add_subplot(gs[0, :2])
        ax2 = fig.add_subplot(gs[0, 2])
        ax3 = fig.add_subplot(gs[0, 3])
        ax4 = fig.add_subplot(gs[1, 0])
        ax5 = fig.add_subplot(gs[1, 1])
        ax6 = fig.add_subplot(gs[1, 2])
        ax7 = fig.add_subplot(gs[1, 3])
        ax8 = fig.add_subplot(gs[2, 0])
        ax9 = fig.add_subplot(gs[2, 1])
        ax10 = fig.add_subplot(gs[2, 2])
        ax11 = fig.add_subplot(gs[2, 3])

        ax1.hist(np.sum(self.species_pop, axis=0), bins=10 ** np.arange(0,7, 0.2), color='#984ea3', edgecolor='black')
        ax1.set_xscale('log')
        ax1.set_xlim([1, 5e6])
        ax1.set_xlabel("Birds per premises")
        ax1.set_ylabel("Counts")

        ax11.bar(["Galliformes\nonly ", "Waterfowl\nonly", "Other birds\nonly", "Mixed\nbirds"],
                [np.sum((self.species_pop[1, :] == 0) & (self.species_pop[2, :] == 0)),
                 np.sum((self.species_pop[0, :] == 0) & (self.species_pop[2, :] == 0)),
                 np.sum((self.species_pop[1, :] == 0) & (self.species_pop[0, :] == 0)),
                 self.n_farms - (np.sum((self.species_pop[1, :] == 0) & (self.species_pop[2, :] == 0)) +
                 np.sum((self.species_pop[0, :] == 0) & (self.species_pop[2, :] == 0)) +
                 np.sum((self.species_pop[1, :] == 0) & (self.species_pop[0, :] == 0))) ],
                color=['#e41a1c', '#377eb8', '#4daf4a', '#ff7f00'], edgecolor='black', linewidth=0.5)
        ax11.set_ylabel("Premises")
        ax11.tick_params(axis='x', labelsize=9)

        ax3.bar(["Galliformes", "Waterfowl", "Other birds"], np.sum(self.species_pop, axis=1),
                color=['#e41a1c', '#377eb8', '#4daf4a'], edgecolor='black', linewidth=0.5)
        ax3.set_ylabel("Birds in premises")

        ax4.hist(self.species_pop[0, (self.species_pop[1, :] == 0) & (self.species_pop[2, :] == 0)], bins=10 ** np.arange(0,7, 0.2), color='#e41a1c', edgecolor='black')
        ax4.set_xscale('log')
        ax4.set_xlim([1, 5e6])
        ax4.set_xlabel("Galliformes per premises")
        ax4.set_ylabel("Counts")
        ax5.hist(self.species_pop[1, (self.species_pop[0, :] == 0) & (self.species_pop[2, :] == 0)], bins=10 ** np.arange(0,5.4, 0.2), color='#377eb8', edgecolor='black')
        ax5.set_xscale('log')
        ax5.set_xlim([1, 5e6])
        ax5.set_xlabel("Waterfowl per premises")
        ax5.set_ylabel("Counts")
        ax6.hist(self.species_pop[2, (self.species_pop[0, :] == 0) & (self.species_pop[1, :] == 0)], bins=10 ** np.arange(0,6.4, 0.2), color='#4daf4a', edgecolor='black')
        ax6.set_xscale('log')
        ax6.set_xlim([1, 5e6])
        ax6.set_xlabel("Other birds per premises")
        ax6.set_ylabel("Counts")
        ax7.hist(self.species_pop[2, ~((self.species_pop[1, :] == 0) & (self.species_pop[0, :] == 0)) & ~((self.species_pop[2, :] == 0) & (self.species_pop[0, :] == 0)) & ~((self.species_pop[1, :] == 0) & (self.species_pop[2, :] == 0))], bins=10 ** np.arange(0,6.4, 0.2), color='#ff7f00', edgecolor='black')
        ax7.set_xscale('log')
        ax7.set_xlim([1, 5e6])
        ax7.set_xlabel("Mixed birds per premises")
        ax7.set_ylabel("Counts")
        plt.tight_layout()
        plt.subplots_adjust(wspace=0.33)

        cmap = colors.LinearSegmentedColormap.from_list("black_to_custom", ["#000004", "#984EA3", "#FFE7FF"])#cm.get_cmap("inferno")
        norm = colors.LogNorm(vmin=1e2, vmax=1e7)
        uk = self.polygons.dissolve()
        uk.plot(ax=ax2, edgecolor=None, facecolor='#000004')
        uk = uk.geometry.iloc[0]
        paths = [Path(np.array(polygon.exterior.coords)) for polygon in uk.geoms]
        uk_path = Path.make_compound_path(*paths)
        uk_patch = PathPatch(uk_path, transform=ax2.transData, facecolor='none', edgecolor='none')
        for i in range(self.n_grids):
            if 1000*self.grid_location_y[i] < 1.08e6:
                color = cmap(norm(self.birds_in_grid[i]))
                rect = plt.Rectangle((1000*self.grid_location_x[i], 1000*self.grid_location_y[i]), 1000*self.grid_size[i],
                                           1000*self.grid_size[i], facecolor=color, edgecolor=None)
                rect.set_clip_path(uk_patch)
                ax2.add_patch(rect)
        ax2.set_ylim([0, 1.08e6])
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array(np.mean(self.birds_in_grid, axis=0))
        s2 = plt.colorbar(sm, ax=ax2, orientation="vertical", label="Birds in premises per grid cell")
        ax2.set_xticks([])
        ax2.set_yticks([])
        ax2_pos = ax2.get_position()
        inset_x = ax2_pos.width * 8.6e4 / (ax2.get_xlim()[1] - ax2.get_xlim()[0])
        inset_y = ax2_pos.height * 1.3e5 / (ax2.get_ylim()[1] - ax2.get_ylim()[0])
        inset_ax = fig.add_axes(
            [ax2_pos.x0 + ax2_pos.width -0.063, ax2_pos.y0 + ax2_pos.height -0.036, inset_x, inset_y])
        uk = self.polygons.dissolve()
        uk.plot(ax=inset_ax, edgecolor=None, facecolor='#000004')
        uk = uk.geometry.iloc[0]
        paths = [Path(np.array(polygon.exterior.coords)) for polygon in uk.geoms]
        uk_path = Path.make_compound_path(*paths)
        uk_patch = PathPatch(uk_path, transform=ax2.transData, facecolor='none', edgecolor='none')
        for i in range(self.n_grids):
            if 1000*self.grid_location_y[i] > 1.08e6:
                color = cmap(norm(self.birds_in_grid[i]))
                rect = plt.Rectangle((1000*self.grid_location_x[i], 1000*self.grid_location_y[i]), 1000*self.grid_size[i],
                                           1000*self.grid_size[i], facecolor=color, edgecolor=None)
                rect.set_clip_path(uk_patch)
                inset_ax.add_patch(rect)
        inset_ax.set_xlim([3.9e5, 4.76e5])
        inset_ax.set_ylim([1.1e6, 1.23e6])
        inset_ax.set_xticks([])
        inset_ax.set_yticks([])

        cmap = colors.LinearSegmentedColormap.from_list("black_to_custom", ["#000004", "#E41A1C", "#FFB3B5"])
        norm = colors.LogNorm(vmin=1e2, vmax=1e7)
        uk = self.polygons.dissolve()
        uk.plot(ax=ax8, edgecolor=None, facecolor='#000004')
        uk = uk.geometry.iloc[0]
        paths = [Path(np.array(polygon.exterior.coords)) for polygon in uk.geoms]  # Handle multiple polygons
        uk_path = Path.make_compound_path(*paths)  # Combine into one Path
        uk_patch = PathPatch(uk_path, transform=ax8.transData, facecolor='none', edgecolor='none')
        for i in range(self.n_grids):
            if 1000*self.grid_location_y[i] < 1.08e6:
                color = cmap(norm(self.birds_in_grid_1[i]))
                rect = plt.Rectangle((1000*self.grid_location_x[i], 1000*self.grid_location_y[i]), 1000*self.grid_size[i],
                                           1000*self.grid_size[i], facecolor=color, edgecolor=None)
                rect.set_clip_path(uk_patch)
                ax8.add_patch(rect)
        ax8.set_ylim([0, 1.08e6])
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array(np.mean(self.birds_in_grid_1, axis=0))
        s8 = plt.colorbar(sm, ax=ax8, orientation="vertical", label="Galliformes in premises per grid cell")
        ax8.set_xticks([])
        ax8.set_yticks([])
        ax8_pos = ax8.get_position()
        inset_x = ax8_pos.width * 8.6e4 / (ax8.get_xlim()[1] - ax8.get_xlim()[0])
        inset_y = ax8_pos.height * 1.3e5 / (ax8.get_ylim()[1] - ax8.get_ylim()[0])
        inset_ax = fig.add_axes(
            [ax8_pos.x0 + ax8_pos.width -0.063, ax8_pos.y0 + ax8_pos.height -0.038, inset_x, inset_y])
        uk = self.polygons.dissolve()
        uk.plot(ax=inset_ax, edgecolor=None, facecolor='#000004')
        uk = uk.geometry.iloc[0]
        paths = [Path(np.array(polygon.exterior.coords)) for polygon in uk.geoms]
        uk_path = Path.make_compound_path(*paths)
        uk_patch = PathPatch(uk_path, transform=ax8.transData, facecolor='none', edgecolor='none')
        for i in range(self.n_grids):
            if 1000*self.grid_location_y[i] > 1.08e6:
                color = cmap(norm(self.birds_in_grid_1[i]))
                rect = plt.Rectangle((1000*self.grid_location_x[i], 1000*self.grid_location_y[i]), 1000*self.grid_size[i],
                                           1000*self.grid_size[i], facecolor=color, edgecolor=None)
                rect.set_clip_path(uk_patch)
                inset_ax.add_patch(rect)
        inset_ax.set_xlim([3.9e5, 4.76e5])
        inset_ax.set_ylim([1.1e6, 1.23e6])
        inset_ax.set_xticks([])
        inset_ax.set_yticks([])

        cmap = colors.LinearSegmentedColormap.from_list("black_to_custom", ["#000004", "#377EB8", "#D0FFFF"])
        norm = colors.LogNorm(vmin=1e2, vmax=1e7)
        uk = self.polygons.dissolve()
        uk.plot(ax=ax9, edgecolor=None, facecolor='#000004')
        uk = uk.geometry.iloc[0]
        paths = [Path(np.array(polygon.exterior.coords)) for polygon in uk.geoms]
        uk_path = Path.make_compound_path(*paths)
        uk_patch = PathPatch(uk_path, transform=ax9.transData, facecolor='none', edgecolor='none')
        for i in range(self.n_grids):
            if 1000*self.grid_location_y[i] < 1.08e6:
                color = cmap(norm(self.birds_in_grid_2[i]))
                rect = plt.Rectangle((1000*self.grid_location_x[i], 1000*self.grid_location_y[i]), 1000*self.grid_size[i],
                                           1000*self.grid_size[i], facecolor=color, edgecolor=None)
                rect.set_clip_path(uk_patch)
                ax9.add_patch(rect)
        ax9.set_ylim([0, 1.08e6])
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array(np.mean(self.birds_in_grid_2, axis=0))
        s9 = plt.colorbar(sm, ax=ax9, orientation="vertical", label="Waterfowl in premises per grid cell")
        ax9.set_xticks([])
        ax9.set_yticks([])
        ax9_pos = ax9.get_position()
        inset_x = ax9_pos.width * 8.6e4 / (ax9.get_xlim()[1] - ax9.get_xlim()[0])
        inset_y = ax9_pos.height * 1.3e5 / (ax9.get_ylim()[1] - ax9.get_ylim()[0])
        inset_ax = fig.add_axes(
            [ax9_pos.x0 + ax9_pos.width -0.063, ax9_pos.y0 + ax9_pos.height - 0.38, inset_x, inset_y])
        uk = self.polygons.dissolve()
        uk.plot(ax=inset_ax, edgecolor=None, facecolor='#000004')
        uk = uk.geometry.iloc[0]
        paths = [Path(np.array(polygon.exterior.coords)) for polygon in uk.geoms]
        uk_path = Path.make_compound_path(*paths)
        uk_patch = PathPatch(uk_path, transform=ax9.transData, facecolor='none', edgecolor='none')
        for i in range(self.n_grids):
            if 1000*self.grid_location_y[i] > 1.08e6:
                color = cmap(norm(self.birds_in_grid_2[i]))
                rect = plt.Rectangle((1000*self.grid_location_x[i], 1000*self.grid_location_y[i]), 1000*self.grid_size[i],
                                           1000*self.grid_size[i], facecolor=color, edgecolor=None)
                rect.set_clip_path(uk_patch)
                inset_ax.add_patch(rect)
        inset_ax.set_xlim([3.9e5, 4.76e5])
        inset_ax.set_ylim([1.1e6, 1.23e6])
        inset_ax.set_xticks([])
        inset_ax.set_yticks([])

        cmap = colors.LinearSegmentedColormap.from_list("black_to_custom", ["#000004", "#4DAF4A", "#E6FFE3"])
        norm = colors.LogNorm(vmin=1e2, vmax=1e7)
        uk = self.polygons.dissolve()
        uk.plot(ax=ax10, edgecolor=None, facecolor='#000004')
        uk = uk.geometry.iloc[0]
        paths = [Path(np.array(polygon.exterior.coords)) for polygon in uk.geoms]
        uk_path = Path.make_compound_path(*paths)
        uk_patch = PathPatch(uk_path, transform=ax10.transData, facecolor='none', edgecolor='none')
        for i in range(self.n_grids):
            if 1000*self.grid_location_y[i] < 1.08e6:
                color = cmap(norm(self.birds_in_grid_3[i]))
                rect = plt.Rectangle((1000*self.grid_location_x[i], 1000*self.grid_location_y[i]), 1000*self.grid_size[i],
                                           1000*self.grid_size[i], facecolor=color, edgecolor=None)
                rect.set_clip_path(uk_patch)
                ax10.add_patch(rect)
        ax10.set_ylim([0, 1.08e6])
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array(np.mean(self.birds_in_grid_3, axis=0))
        s10 = plt.colorbar(sm, ax=ax10, orientation="vertical", label="Other birds in premises per grid cell")
        ax10.set_xticks([])
        ax10.set_yticks([])
        ax10_pos = ax10.get_position()
        inset_x = ax10_pos.width * 8.6e4 / (ax10.get_xlim()[1] - ax10.get_xlim()[0])
        inset_y = ax10_pos.height * 1.3e5 / (ax10.get_ylim()[1] - ax10.get_ylim()[0])
        inset_ax = fig.add_axes(
            [ax10_pos.x0 + ax10_pos.width -0.063, ax10_pos.y0 + ax10_pos.height - 0.038, inset_x, inset_y])
        uk = self.polygons.dissolve()
        uk.plot(ax=inset_ax, edgecolor=None, facecolor='#000004')
        uk = uk.geometry.iloc[0]
        paths = [Path(np.array(polygon.exterior.coords)) for polygon in uk.geoms]
        uk_path = Path.make_compound_path(*paths)
        uk_patch = PathPatch(uk_path, transform=ax10.transData, facecolor='none', edgecolor='none')
        for i in range(self.n_grids):
            if 1000*self.grid_location_y[i] > 1.08e6:
                color = cmap(norm(self.birds_in_grid_3[i]))
                rect = plt.Rectangle((1000*self.grid_location_x[i], 1000*self.grid_location_y[i]), 1000*self.grid_size[i],
                                           1000*self.grid_size[i], facecolor=color, edgecolor=None)
                rect.set_clip_path(uk_patch)
                inset_ax.add_patch(rect)
        inset_ax.set_xlim([3.9e5, 4.76e5])
        inset_ax.set_ylim([1.1e6, 1.23e6])
        inset_ax.set_xticks([])
        inset_ax.set_yticks([])

        fig.text(0.0095, 0.981, 'A', ha='center', va='center', fontweight='bold', fontsize=20)
        fig.text(0.485, 0.981, 'B', ha='center', va='center', fontweight='bold', fontsize=20)
        fig.text(0.713, 0.981, 'C', ha='center', va='center', fontweight='bold', fontsize=20)
        fig.text(0.0095, 0.612, 'D', ha='center', va='center', fontweight='bold', fontsize=20)
        fig.text(0.25, 0.612, 'E', ha='center', va='center', fontweight='bold', fontsize=20)
        fig.text(0.48, 0.612, 'F', ha='center', va='center', fontweight='bold', fontsize=20)
        fig.text(0.707, 0.612, 'G', ha='center', va='center', fontweight='bold', fontsize=20)
        fig.text(0.0155, 0.343, 'I', ha='center', va='center', fontweight='bold', fontsize=20)
        fig.text(0.246, 0.343, 'J', ha='center', va='center', fontweight='bold', fontsize=20)
        fig.text(0.472, 0.343, 'K', ha='center', va='center', fontweight='bold', fontsize=20)
        fig.text(0.695, 0.343, 'L', ha='center', va='center', fontweight='bold', fontsize=20)

        pos = ax2.get_position()
        ax2.set_position([pos.x0 - 0.04, pos.y0, pos.width, pos.height])
        pos = ax8.get_position()
        ax8.set_position([pos.x0 - 0.04, pos.y0, pos.width, pos.height])
        pos = ax9.get_position()
        ax9.set_position([pos.x0 - 0.04, pos.y0, pos.width, pos.height])
        pos = ax10.get_position()
        ax10.set_position([pos.x0 - 0.04, pos.y0, pos.width, pos.height])
        pos = s2.ax.get_position()
        s2.ax.set_position([pos.x0 - 0.04, pos.y0, pos.width, pos.height])
        pos = s8.ax.get_position()
        s8.ax.set_position([pos.x0 - 0.04, pos.y0, pos.width, pos.height])
        pos = s9.ax.get_position()
        s9.ax.set_position([pos.x0 - 0.04, pos.y0, pos.width, pos.height])
        pos = s10.ax.get_position()
        s10.ax.set_position([pos.x0 - 0.04, pos.y0, pos.width, pos.height])


class Model:
    """Class for the compartmental infection model."""
    def __init__(self, data, n_comps=5, inf_comps=None, data_comps=None, mean_exits=None, rand_exit_comp=None,
                 cull_data=False, init_rand_exit=False, transmission_type=1, kernel_type='cauchy', spatial=False,
                 combine=True):
        self.data = data
        if not isinstance(n_comps, (int, np.integer)):
            raise TypeError("self.n_compartments must be a scalar integer")
        self.n_comps = n_comps
        if inf_comps is None:
            self.inf_comps = np.array([False, False, True, True, False], dtype=bool)
        else:
            if len(inf_comps) != self.n_comps:
                raise TypeError("Invalid infectious list: must be of length n_compartments.")
            self.inf_comps = inf_comps
            for element in self.inf_comps:
                if not isinstance(element, bool):
                    raise TypeError("Invalid infectious list: must an array of booleans.")
            if self.inf_comps[0]:
                raise ValueError("Invalid infectious list: First compartment must not be infectious.")
        self.inf_comps_id = np.where(self.inf_comps)[0]
        if data_comps is None:
            self.data_comps = np.array([False, False, False, True, False], dtype=bool)
        else:
            if len(data_comps) != self.n_comps:
                raise TypeError("Invalid infectious list: must be of length n_compartments.")
            self.data_comps = data_comps
            for element in self.data_comps:
                if not isinstance(element, bool):
                    raise TypeError("Invalid infectious list: must an array of booleans.")
        self.data_comp_id = np.where(self.data_comps)[0]
        if mean_exits is None:
            self.mean_exits = np.array([4, 8, 3])
        else:
            self.mean_exits = mean_exits
        if rand_exit_comp is None:
            self.rand_exit_comp = np.array([False, True, False], dtype=bool)
        else:
            self.rand_exit_comp = rand_exit_comp
            for element in self.rand_exit_comp:
                if not isinstance(element, (bool, np.bool_)):
                    raise TypeError("Invalid exit rate list: must an array of booleans.")
        if len(self.mean_exits) != len(self.rand_exit_comp):
            raise TypeError("Invalid exit rates: the length of mean_exits and rand_exit_time_comp must be equal.")
        if len(self.mean_exits) != self.n_comps - 2:
            raise TypeError("Invalid exit rates: must be two fewer than the number of compartments.")
        self.rand_exit_time_comp_id = np.where(self.rand_exit_comp)[0]
        self.cull_data = cull_data
        self.init_rand_exit = init_rand_exit
        self.transmission_type = transmission_type
        if not isinstance(self.transmission_type, int) or (self.transmission_type < 0) or (self.transmission_type > 5):
            raise TypeError("Invalid transmission type: must be an integer between 0 and 4.")
        self.kernel_type = kernel_type
        if self.kernel_type not in ['cauchy', 'exp']:
            raise ValueError("Invalid kernel type: must be 'cauchy' or 'exp'.")
        self.spatial = spatial
        if self.spatial and ((self.transmission_type == 2) or (self.transmission_type == 4)):
            raise ValueError("Spatial model must not be used with transmission type 2 or 4.")
        self.combine = combine
        self.n_species = self.data.n_species
        self.par_names = ['epsilon', 'gamma', 'delta', 'omega', 'psi', 'phi', 'xi', 'zeta', 'nu', 'rho', 'a', 'b']
        self.pars = dict()
        self.pars['value'] = {'epsilon': np.array([1e-5]), 'gamma': np.array([0.05] + [0.9 * 0.05] * (np.sum(self.inf_comps) - 1)),
                              'delta': np.array([2.0]), 'omega': np.array([1.33]), 'psi': np.array([0.5] * self.n_species),
                              'phi': np.array([0.5] * self.n_species), 'xi': np.array([1] + [0.5] * (self.n_species - 1)),
                              'zeta': np.array([1] + [0.5] * (self.n_species - 1)), 'nu': np.array([1.5, 0.5]),
                              'rho': np.array([0]), 'a': np.array([4.0]), 'b': np.array([2.0])}
        self.pars['length'] = {'epsilon': 1, 'gamma': np.sum(self.inf_comps), 'delta': 1, 'omega': 1, 'psi': self.n_species,
                                 'phi': self.n_species, 'xi': self.n_species, 'zeta': self.n_species, 'nu': 2, 'rho': 1,
                                 'a': 1, 'b': 1}
        self.pars['fitting'] = {'epsilon': np.array([False]), 'gamma': np.array([False] * np.sum(self.inf_comps)),
                                'delta': np.array([False]), 'omega': np.array([False]), 'psi': np.array([False] * self.n_species),
                                'phi': np.array([False] * self.n_species), 'xi': np.array([False] * self.n_species),
                                'zeta': np.array([False] * self.n_species), 'nu': np.array([False, False]),
                                'rho': np.array([False]), 'a': np.array([False]), 'b': np.array([False])}
        self.pars['initial'] = copy.deepcopy(self.pars['value'])
        self.pars['current'] = copy.deepcopy(self.pars['value']) #np.array([0.05] + [0.9] * (np.sum(self.inf_comps) - 1))
        self.pars['prior_type'] = {'epsilon': ['gamma'], 'gamma': ['gamma'] * np.sum(self.inf_comps),
                                    'delta': ['gamma'], 'omega': ['gamma'], 'psi': ['beta'] * self.n_species,
                                    'phi': ['beta'] * self.n_species, 'xi': ['gamma'] * self.n_species,
                                    'zeta': ['gamma'] * self.n_species, 'nu': ['gamma', 'beta'],
                                    'rho': ['gamma'], 'a': ['gamma'], 'b': ['gamma']}
        self.pars['prior_pars'] = {'epsilon': np.array([[1, 1e-5]]),
                                   'gamma': np.vstack(([1, 0.01], np.tile([1, 0.8], [(np.sum(self.inf_comps) - 1), 1]))),
                                    'delta': np.array([[2, 1]]), 'omega': np.array([[200, 1/150]]),
                                    'psi': np.tile([2, 2], [self.n_species, 1]), 'phi': np.tile([2, 2], [self.n_species, 1]),
                                    'xi': np.tile([1, 1], [self.n_species, 1]), 'zeta': np.tile([1, 1], [self.n_species, 1]),
                                    'nu': np.array([[2, 2], [2, 2]]), 'rho': np.array([[1, 1]]), 'a': np.array([[100, 1/25]]),
                                    'b': np.array([[100, 1/200]])}
        self.pars['description'] = {'epsilon': ['Baseline infectious\npressure'],
                                    'gamma': ['Infectious pressure\nfrom infected farms', 'Multiplicative factor\nfor notified farms'],
                                    'delta': ['Scale parameter in\ntransmission kernel'],
                                    'omega': ['Exponent in\ntransmission kernel'],
                                    'psi': ['Exponent for\ninfected Galliformes', 'Exponent for\ninfected waterfowl',
                                            'Exponent for\ninfected other birds'],
                                    'phi': ['Exponent for\nsusceptible Galliformes',
                                            'Exponent for\nsusceptible waterfowl',
                                            'Exponent for\nsusceptible other birds'],
                                    'xi': ['None', 'Relative transmissibility\nof waterfowl to Galliformes',
                                           'Relative transmissibility\nof other birds to Galliformes'],
                                    'zeta': ['None', 'Relative susceptibility\nof waterfowl to Galliformes',
                                             'Relative susceptibility\nof other birds to Galliformes'],
                                    'nu': ['Shape of seasonality', 'Timing of seasonality'],
                                    'rho': ['Transmission decay\nrate after culling'],
                                    'a': ['Shape parameter for the\ndistribution of the latent period'],
                                    'b': ['Scale parameter for the\ndistribution of the latent period']}

        for name in self.par_names:
            if len(self.pars['value'][name]) != self.pars['length'][name]:
                raise TypeError("Invalid parameter value for " + name + ": must be of correct length.")
            if len(self.pars['fitting'][name]) != self.pars['length'][name]:
                raise TypeError("Invalid parameter fitting for " + name + ": must be of correct length.")
            if len(self.pars['initial'][name]) != self.pars['length'][name]:
                raise TypeError("Invalid parameter initial for " + name + ": must be of correct length.")
            if len(self.pars['current'][name]) != self.pars['length'][name]:
                raise TypeError("Invalid parameter current for " + name + ": must be of correct length.")
            if len(self.pars['prior_type'][name]) != self.pars['length'][name]:
                raise TypeError("Invalid parameter prior_type for " + name + ": must be of correct length.")
            if (self.pars['prior_pars'][name].shape[0] != self.pars['length'][name]) or (self.pars['prior_pars'][name].shape[1] != 2):
                raise TypeError("Invalid parameter prior_pars for " + name + ": must be of correct length.")
            if len(self.pars['description'][name]) != self.pars['length'][name]:
                raise TypeError("Invalid parameter description for " + name + ": must be of correct length.")
        if self.pars['fitting']['xi'][0]:
            raise ValueError("Transmissibility of Galliformes must not be fitted.")
        if self.pars['fitting']['zeta'][0]:
            raise ValueError("Susceptibility of Galliformes must not be fitted.")
        if transmission_type == 0:
            self.pars['value']['gamma'] = np.array([0.0] * np.sum(self.inf_comps))
            self.pars['initial']['gamma'] = np.array([0.0] * np.sum(self.inf_comps))
            self.pars['current'] ['gamma']= np.array([0.0] * np.sum(self.inf_comps))
            self.pars['value']['delta'] = np.array([0.0])
            self.pars['initial']['delta'] = np.array([0.0])
            self.pars['current']['delta'] = np.array([0.0])
            self.pars['value']['omega'] = np.array([0.0])
            self.pars['initial']['omega'] = np.array([0.0])
            self.pars['current']['omega'] = np.array([0.0])
            self.pars['value']['psi'] = np.array([0.0] * self.n_species)
            self.pars['initial']['psi'] = np.array([0.0] * self.n_species)
            self.pars['current']['psi'] = np.array([0.0] * self.n_species)
            self.pars['value']['phi'] = np.array([0.0] * self.n_species)
            self.pars['initial']['phi'] = np.array([0.0] * self.n_species)
            self.pars['current']['phi'] = np.array([0.0] * self.n_species)
            self.pars['value']['xi'] = np.array([1.0] + [0.0] * (self.n_species - 1))
            self.pars['initial']['xi'] = np.array([1.0] + [0.0] * (self.n_species - 1))
            self.pars['current']['xi'] = np.array([1.0] + [0.0] * (self.n_species - 1))
            self.pars['value']['zeta'] = np.array([1.0] + [0.0] * (self.n_species - 1))
            self.pars['initial']['zeta'] = np.array([1.0] + [0.0] * (self.n_species - 1))
            self.pars['current']['zeta'] = np.array([1.0] + [0.0] * (self.n_species - 1))
        if (transmission_type == 3) or (transmission_type == 4):
            self.pars['value']['rho'] = np.array([1.0])
            self.pars['initial']['rho'] = np.array([1.0])
            self.pars['current']['rho'] = np.array([1.0])
        if transmission_type == 5:
            self.pars['value']['rho'] = np.array([4.0])
            self.pars['initial']['rho'] = np.array([4.0])
            self.pars['current']['rho'] = np.array([4.0])
        if self.kernel_type == 'exp':
            self.pars['value']['delta'] = np.array([0.5])
            self.pars['initial']['delta'] = np.array([0.5])
            self.pars['current']['delta'] = np.array([0.5])
            self.pars['prior_pars']['delta'] = np.array([[1, 0.5]])
            self.pars['value']['gamma'] = np.array([0.005] + [0.9 * 0.005] * (np.sum(self.inf_comps) - 1))
            self.pars['initial']['gamma'] = np.array([0.005] + [0.9 * 0.005] * (np.sum(self.inf_comps) - 1))
            self.pars['current']['gamma'] = np.array([0.005] + [0.9 * 0.005] * (np.sum(self.inf_comps) - 1))
            self.pars['prior_pars']['gamma'][0] = np.array([[1, 0.005]])
            self.pars['value']['omega'] = np.array([0.0])
            self.pars['initial']['omega'] = np.array([0.0])
            self.pars['current']['omega'] = np.array([0.0])
        if self.spatial:
            self.pars['value']['nu'] = np.array([0.0, 0.0])
            self.pars['initial']['nu'] = np.array([0.0, 0.0])
            self.pars['current']['nu'] = np.array([0.0, 0.0])

        if cull_data:
            self.report_day = self.data.report_day[self.data.report_day > -self.data.cull_times]
            self.infected_farms = self.data.matched_farm[self.data.report_day > -self.data.cull_times]
            self.past_infected_farms = self.data.matched_farm[(self.data.report_day <= -self.data.cull_times) & (self.data.report_day >= -self.data.past_start_day)]
        else:
            self.report_day = self.data.report_day[self.data.report_day > -self.mean_exits[-1]]
            self.infected_farms = self.data.matched_farm[self.data.report_day > -self.mean_exits[-1]]
            self.past_infected_farms = self.data.matched_farm[(self.data.report_day <= -self.mean_exits[-1]) & (self.data.report_day >= -self.data.past_start_day)]

        self.data_farms = copy.deepcopy(self.infected_farms)
        self.end_day = self.data.end_day
        self.rand_exits = np.zeros((len(self.infected_farms), len(self.rand_exit_time_comp_id)))
        if self.init_rand_exit:
            for i in range(self.rand_exits.shape[1]):
                self.rand_exits[:, i] = np.random.gamma(self.pars['value']['a'][i], self.pars['value']['b'][i], len(self.infected_farms))
        else:
            rnd = np.random.RandomState(1)
            for i in range(self.rand_exits.shape[1]):
                self.rand_exits[:, i] = rnd.gamma(self.pars['value']['a'][i], self.pars['value']['b'][i], len(self.infected_farms))
        self.exposure_rate = None
        self.sus = None
        self.transmission = None
        self.transmission_scale = None
        self.farms_kernel = None
        self.season_times = None
        self.max_iter = None
        self.first_iter = None
        self.burn_in = None
        self.ind_update = None
        self.log_update = None
        self.prop_updates = None
        self.to_fit = None
        self.first_exposed = None
        self.first_sigma = None
        self.xy_farms = np.array([self.data.location_x, self.data.location_y]).reshape((2, 1, self.data.n_farms))

        self.cov_rate = 10
        self.lambda_rate = 50
        self.lambda_iter = 1

        self.neg_log_like_chain = None
        self.neg_log_post_chain = None
        self.pars_chain = None
        self.infected_chain = None
        self.exit_chain = None
        # self.total_exposure_chain = None
        self.num_occult_chain = None
        self.mean_exit_chain = None

        self.neg_log_like_chains = None
        self.neg_log_post_chains = None
        self.pars_chains = None
        self.infected_chains = None
        self.exit_chains = None
        # self.total_exposure_chains = None
        self.num_occult_chains = None
        self.mean_exit_chains = None

        self.neg_log_like_post = None
        self.neg_log_post_post = None
        self.pars_post = None
        self.infected_post = None
        self.exit_post = None
        # self.total_exposure_post = None
        self.num_occult_post = None
        self.mean_exit_post = None

        self.acceptance = 0
        self.acceptance_exit = 0
        self.updates_exit = 0
        self.acceptance_add = 0
        self.updates_add = 0
        self.acceptance_rem = 0
        self.updates_rem = 0

    def kernel(self, distance2):
        """Calculate the kernel value for a given squared distance."""
        if self.kernel_type == 'cauchy':
            out = self.pars['value']['delta'] / ((self.pars['value']['delta'] ** 2 + distance2) ** self.pars['value']['omega'])
            return out
        elif self.kernel_type == 'exp':
            out = np.exp(-(distance2 ** 0.5) * self.pars['value']['delta'])
            return out
        else:
            raise ValueError("kernel_type must be 'cauchy' or 'exp'.")

    def run_mcmc(self, chain_num=0, max_iter=10000, first_iter=1000, burn_in=1000, ind_update=True, log_update=True,
                 prop_updates=0.05, to_fit=None, first_sigma=True, save=True, save_tmp=None):
        """Run the MCMC algorithm."""
        if burn_in > max_iter:
            raise ValueError("Burn-in must be less than max_iter.")
        if first_iter > max_iter:
            raise ValueError("first_iter must be less than max_iter.")
        start_iter = 0
        accept = 0
        accept_exit = 0
        accept_add = 0
        accept_rem = 0
        update_exit = 0
        update_add = 0
        update_rem = 0
        self.max_iter = max_iter
        self.first_iter = first_iter
        self.burn_in = burn_in
        self.ind_update = ind_update
        self.log_update = log_update
        self.prop_updates = prop_updates
        self.first_sigma = first_sigma
        self.save = save
        # Parameters to fit under model assumptions
        if to_fit is None:
            if self.spatial:
                if self.transmission_type == 0:
                    self.to_fit = ['epsilon']
                elif (self.transmission_type == 1) or (self.transmission_type == 2):
                    self.to_fit = ['epsilon', 'gamma', 'delta', 'psi', 'phi', 'xi', 'zeta']
                elif (self.transmission_type == 3) or (self.transmission_type == 4):
                    self.to_fit = ['epsilon', 'gamma', 'delta', 'psi', 'phi', 'xi', 'zeta', 'rho']
                elif self.transmission_type == 5:
                    self.to_fit = ['epsilon', 'gamma', 'delta', 'psi', 'phi', 'xi', 'zeta']
            else:
                if self.transmission_type == 0:
                    self.to_fit = ['epsilon', 'nu']
                elif (self.transmission_type == 1) or (self.transmission_type == 2):
                    self.to_fit = ['epsilon', 'gamma', 'delta', 'psi', 'phi', 'xi', 'zeta', 'nu']
                elif (self.transmission_type == 3) or (self.transmission_type == 4):
                    self.to_fit = ['epsilon', 'gamma', 'delta', 'psi', 'phi', 'xi', 'zeta', 'nu', 'rho']
                elif self.transmission_type == 5:
                    self.to_fit = ['epsilon', 'gamma', 'delta', 'psi', 'phi', 'xi', 'zeta', 'nu']
        else:
            self.to_fit = to_fit
        for name in self.to_fit:
            if name not in self.par_names:
                raise ValueError("Invalid parameter name: " + name + " not in par_names.")
            else:
                if (name == 'xi') or (name == 'zeta'):
                    self.pars['fitting'][name] = np.array([False] + [True] * (self.pars['length'][name] - 1))
                else:
                    self.pars['fitting'][name] = np.array([True] * self.pars['length'][name])
        self.n_to_fit = 0
        for name in self.par_names:
            self.n_to_fit += np.sum(self.pars['fitting'][name])
        # Sort special joint update for zeta and xi
        if self.combine and self.pars['fitting']['gamma'][0]:
            self.pars['current']['xi'][1:] *= self.pars['current']['gamma'][0]
            self.pars['current']['zeta'][1:] *= self.pars['current']['gamma'][0]
        if np.sum(self.pars['fitting']['gamma'][1:]) > 0:
            self.pars['current']['gamma'][1:] /= self.pars['current']['gamma'][0]
        current_tmp = copy.deepcopy(self.pars['current'])
        self.pars['current'] = dict()
        for name in self.to_fit:
            self.pars['current'][name] = current_tmp[name][self.pars['fitting'][name]]
            if self.log_update:
                for i in range(np.sum(self.pars['fitting'][name])):
                    if np.array(self.pars['prior_type'][name])[self.pars['fitting'][name]][i] == 'gamma':
                        self.pars['current'][name][i] = np.log(self.pars['current'][name][i])
                    elif np.array(self.pars['prior_type'][name])[self.pars['fitting'][name]][i] == 'beta':
                        self.pars['current'][name][i] = np.log(self.pars['current'][name][i]) - np.log(1 - self.pars['current'][name][i])
        self.pars['mu'] = copy.deepcopy(self.pars['current'])

        if log_update:
            self.pars['sigma'] = {'epsilon': np.array([0.2]),
                                    'gamma': np.array([0.5] + [0.1] * (np.sum(self.inf_comps) - 1)),
                                    'delta': np.array([0.2]),
                                    'omega': np.array([0.05]),
                                    'psi': np.array([3.0] * self.n_species),
                                    'phi': np.array([3.0] * self.n_species),
                                    'xi': np.array([1.0] * (self.n_species)),
                                    'zeta': np.array([1.0] * (self.n_species)),
                                    'nu': np.array([1.0, 1.0]),
                                    'rho': np.array([0.5]),
                                    'a': np.array([0.3]),
                                    'b': np.array([0.3])}
        else:
            self.pars['sigma'] = {'epsilon': np.array([2e-12]),
                                    'gamma': np.array([1e-5] + [0.1] * (np.sum(self.inf_comps) - 1)),
                                    'delta': np.array([0.02]),
                                    'omega': np.array([0.1]),
                                    'psi': np.array([0.05] * self.n_species),
                                    'phi': np.array([0.05] * self.n_species),
                                    'xi': np.array([1.0] * (self.n_species)),
                                    'zeta': np.array([1.0] * (self.n_species)),
                                    'nu': np.array([0.2, 0.05]),
                                    'rho': np.array([1.0]),
                                    'a': np.array([0.1]),
                                    'b': np.array([0.01])}
        if first_sigma:
            sigma_0 = copy.deepcopy(self.pars['sigma'])
        if self.ind_update:
            for name in self.to_fit:
                if np.sum(self.pars['fitting'][name]) > 1:
                    sigma_0[name] = np.diag(sigma_0[name][self.pars['fitting'][name]])
            for name in self.par_names:
                if name not in self.to_fit:
                    del sigma_0[name]

        tmp = np.tile(self.mean_exits[:(self.data_comp_id[0] - 1)], [len(self.infected_farms), 1])
        tmp[:, self.rand_exit_time_comp_id[self.rand_exit_time_comp_id <(self.data_comp_id[0] - 1)]] = np.round(self.rand_exits)

        self.first_exposed = np.min(self.report_day - np.sum(tmp, axis=1))
        self.season_times = self.update_season_times()
        self.exposure_rate = self.update_exposure_rate()

        self.pars_chain = [dict() for _ in range(self.max_iter + 1)]
        pars_old = dict()
        pars_old_full = dict()
        pars_new = dict()

        for name in self.par_names:
            self.pars_chain[0][name] = copy.deepcopy(self.pars['value'][name])
        self.infected_chain = [None for _ in range(self.max_iter + 1)]
        self.infected_chain[0] = self.infected_farms
        self.exit_chain = [None for _ in range(self.max_iter + 1)]
        self.exit_chain[0] = self.rand_exits.flatten()
        self.num_occult_chain = np.zeros(self.max_iter + 1)
        self.mean_exit_chain = np.zeros(self.max_iter + 1)
        self.mean_exit_chain[0] = np.mean(self.rand_exits)
        self.neg_log_like_chain = np.zeros(self.max_iter + 1)
        self.neg_log_post_chain = np.zeros(self.max_iter + 1)
        self.neg_log_like_chain[0] = self.get_neg_log_like()
        self.neg_log_post_chain[0] = self.neg_log_like_chain[0] + self.get_neg_log_prior()
        current_neg_log_post = self.neg_log_post_chain[0]

        # Begin MCMC iterations
        for iter in range(start_iter, self.max_iter):
            print(iter)
            self.chain_string = (self.data.date_start.strftime('%Y%m%d') + '_' + self.data.date_end.strftime('%Y%m%d') +
                                 '_' + str(iter + 1) + '_t' + str(self.transmission_type) + '_' + self.kernel_type +
                                 '_c' + str(self.combine) + '_s' + str(self.spatial) + '_')
            if self.cull_data:
                self.chain_string += 'cull' + str(self.cull_data) + '_'
            if self.data.select_region is not None:
                self.chain_string += self.data.region_names[self.data.select_region].replace(" ", "_") + '_'
            # First iterations: update all parameters separately
            if iter < first_iter:
                for name in self.to_fit:
                    for i in range(np.sum(self.pars['fitting'][name])):
                        # Propose new parameter values
                        par_old_full = self.pars['value'][name][np.where(self.pars['fitting'][name])[0][i]]
                        par_old = self.pars['current'][name][i]
                        par_new = par_old + np.random.normal(0, np.sqrt(self.pars['sigma'][name][np.where(self.pars['fitting'][name])[0][i]]))
                        self.pars['current'][name][i] = par_new
                        if self.log_update:
                            if np.array(self.pars['prior_type'][name])[self.pars['fitting'][name]][i] == 'gamma':
                                par_new_full = np.exp(par_new)
                                neg_log_jacob = par_old - par_new
                            elif np.array(self.pars['prior_type'][name])[self.pars['fitting'][name]][i] == 'beta':
                                par_new_full = np.exp(par_new) / (1 + np.exp(par_new))
                                neg_log_jacob = np.log(par_old_full) - np.log(par_new_full) + np.log(1-par_old_full) - np.log(1-par_new_full)
                            if name == 'gamma':
                                if (i > 0) or not (self.pars['fitting']['gamma'][0]):
                                    par_new_full *= self.pars['value'][name][0]
                            elif (name == 'xi' or name == 'zeta') and self.combine:
                                if 'gamma' in self.to_fit:
                                    par_new_full *= 1 / self.pars['value']['gamma'][0]
                        else:
                            neg_log_jacob = 0
                        # Update model parameters, seasonality and exposure rate
                        self.pars['value'][name][np.where(self.pars['fitting'][name])[0][i]] = par_new_full
                        if name == 'gamma' and self.combine and (i == 0) and (self.pars['fitting']['gamma'][0]):
                            if 'xi' in self.to_fit:
                                self.pars['value']['xi'][self.pars['fitting']['xi']] *= par_old_full / par_new_full
                            if 'zeta' in self.to_fit:
                                self.pars['value']['zeta'][self.pars['fitting']['zeta']] *= par_old_full / par_new_full
                        if name == 'nu':
                            self.season_times = self.update_season_times()
                        self.exposure_rate = self.update_exposure_rate()
                        # Calculate new likelihood and posterior probability
                        new_neg_log_like = self.get_neg_log_like()
                        new_neg_log_post = new_neg_log_like + self.get_neg_log_prior()
                        # Accept or reject the new parameter values
                        if current_neg_log_post - new_neg_log_post - neg_log_jacob > np.log(np.random.uniform()):
                            print([name + '_' + str(i), 'accept', par_old_full, par_new_full, current_neg_log_post,
                                   new_neg_log_post])
                            current_neg_log_post = new_neg_log_post
                            self.pars['sigma'][name][np.where(self.pars['fitting'][name])[0][i]] *= 1.4
                            accept += 1 / self.n_to_fit
                        else:
                            print([name + '_' + str(i), 'reject', par_old_full, par_new_full, current_neg_log_post,
                                   new_neg_log_post])
                            self.pars['sigma'][name][np.where(self.pars['fitting'][name])[0][i]] *= (1.4 ** -0.7857143)
                            self.pars['current'][name][i] = par_old
                            self.pars['value'][name][np.where(self.pars['fitting'][name])[0][i]] = par_old_full
                            if (name == 'gamma') and (i == 0) and (self.pars['fitting']['gamma'][0]) and self.combine:
                                if 'xi' in self.to_fit:
                                    self.pars['value']['xi'][self.pars['fitting']['xi']] *= par_new_full / par_old_full
                                if 'zeta' in self.to_fit:
                                    self.pars['value']['zeta'][self.pars['fitting']['zeta']] *= par_new_full / par_old_full
                            if name == 'nu':
                                self.season_times = self.update_season_times()
                            self.exposure_rate = self.update_exposure_rate()
            else:
                if self.first_sigma and first_iter == iter:
                    self.pars['sigma'] = copy.deepcopy(sigma_0)
                neg_log_jacob = 0
                if self.ind_update:
                    # Update all parameters together, proposing by parameter
                    for name in self.to_fit:
                        if np.sum(self.pars['fitting'][name]) > 1:
                            sd = np.sqrt(np.diag(self.pars['sigma'][name]))
                            den = np.outer(sd, sd)
                            den[den == 0] = 1e-16
                            den[np.isnan(den)] = 1e-16
                            corr = self.pars['sigma'][name] / den
                            corr[self.pars['sigma'][name] == 0] = 0
                        else:
                            sd = np.sqrt(self.pars['sigma'][name])
                            if np.isnan(sd) or sd == 0:
                                sd = 1e-16
                        pars_old[name] = copy.deepcopy(self.pars['current'][name])
                        pars_old_full[name] = copy.deepcopy(self.pars['value'][name][self.pars['fitting'][name]])
                        # Propose new parameters
                        if len(pars_old[name]) > 1:
                            pars_new[name] = (pars_old[name] + np.random.multivariate_normal(np.zeros(
                                len(pars_old[name])), corr) * (1 + (iter % 2) * (self.lambda_iter - 1))
                                              * 2.38 * sd / np.sqrt(len(sd)))
                        else:
                            pars_new[name] = pars_old[name] + np.random.normal(0, (1 + (iter % 2) *
                                                                                   (self.lambda_iter - 1)) * 2.38 * sd)
                        self.pars['current'][name] = pars_new[name]
                else:
                    raise ValueError("Joint update not yet implemented.")
                # Update model parameters, seasonality and exposure rate
                pars_new_full = copy.deepcopy(pars_new)
                for name in self.to_fit:
                    if self.log_update:
                        for i in range(np.sum(self.pars['fitting'][name])):
                            if np.array(self.pars['prior_type'][name])[self.pars['fitting'][name]][i] == 'gamma':
                                pars_new_full[name][i] = np.exp(pars_new[name][i])
                            elif np.array(self.pars['prior_type'][name])[self.pars['fitting'][name]][i] == 'beta':
                                pars_new_full[name][i] = np.exp(pars_new[name][i]) / (1 + np.exp(pars_new[name][i]))
                    if name == 'gamma':
                        pars_new_full[name] = np.insert(pars_new_full[name][0] * pars_new_full[name][1:], 0, pars_new_full[name][0])
                    elif (name == 'xi' or name == 'zeta') and ('gamma' in self.to_fit) and self.combine:
                        pars_new_full[name] *= 1 / pars_new_full['gamma'][0]
                    self.pars['value'][name][self.pars['fitting'][name]] = pars_new_full[name]
                if self.log_update:
                    for name in self.to_fit:
                        for i in range(len(pars_old[name])):
                            if np.array(self.pars['prior_type'][name])[self.pars['fitting'][name]][i] == 'gamma':
                                neg_log_jacob += pars_old[name][i] - pars_new[name][i]
                            elif np.array(self.pars['prior_type'][name])[self.pars['fitting'][name]][i] == 'beta':
                                np.log(pars_old_full[name][i]) - np.log(pars_new_full[name][i]) + np.log(
                                    1 - pars_old_full[name][i]) - np.log(1 - pars_new_full[name][i])
                self.season_times = self.update_season_times()
                self.exposure_rate = self.update_exposure_rate()
                # Calculate new likelihood and posterior probability
                new_neg_log_like = self.get_neg_log_like()
                new_neg_log_post = new_neg_log_like + self.get_neg_log_prior()
                print(self.pars['value'])
                iter_2 = iter - first_iter + 1
                # Accept or reject the new parameter values
                if current_neg_log_post - new_neg_log_post - neg_log_jacob > np.log(np.random.uniform()):
                    print(['accept', current_neg_log_post, new_neg_log_post])
                    current_neg_log_post = new_neg_log_post
                    accept += 1
                    if iter % 2 == 1:
                        self.lambda_iter *= (1 + self.lambda_rate / (self.lambda_rate + iter_2))
                else:
                    print(['reject', current_neg_log_post, new_neg_log_post])
                    self.pars['current'] = copy.deepcopy(pars_old)
                    if iter % 2 == 1:
                        self.lambda_iter *= ((1 + self.lambda_rate / (self.lambda_rate + iter_2)) ** -0.305483)
                    for name in self.to_fit:
                        self.pars['value'][name][self.pars['fitting'][name]] = copy.deepcopy(pars_old_full[name])
                    self.season_times = self.update_season_times()
                    self.exposure_rate = self.update_exposure_rate()
                # Update sigma values
                if self.ind_update:
                    new_mu = dict()
                    ss_pars = dict()
                    ss_mu = dict()
                    ss_new_mu = dict()
                    for name in self.to_fit:
                        new_mu[name] = (iter_2 / (iter_2 + 1)) * self.pars['mu'][name] + self.pars['current'][name] / (
                                    iter_2 + 1)
                        if self.pars['length'][name] > 1:
                            ss_pars[name] = np.outer(self.pars['current'][name], self.pars['current'][name])
                        else:
                            ss_pars[name] = self.pars['current'][name] ** 2
                        if self.pars['length'][name] > 1:
                            ss_mu[name] = np.outer(self.pars['mu'][name], self.pars['mu'][name])
                            ss_new_mu[name] = np.outer(new_mu[name], new_mu[name])
                        else:
                            ss_mu[name] = self.pars['mu'][name] ** 2
                            ss_new_mu[name] = new_mu[name] ** 2
                        self.pars['mu'][name] = copy.deepcopy(new_mu[name])
                        self.pars['sigma'][name] = ((iter_2 - 1 + self.cov_rate) * self.pars['sigma'][name] + iter_2 * ss_mu[name] - (iter_2 + 1) * ss_new_mu[name] + ss_pars[name]) / (iter_2 + self.cov_rate)
                        self.pars['sigma'][name] = self.symmetric_pos_def(self.pars['sigma'][name])
                else:
                    raise ValueError("Joint update not yet implemented.")
            # Update infected farms
            n_updates = np.floor(self.prop_updates * len(self.infected_farms)).astype(int)
            # Three events types (change infection time, add occult infection, remove occult infection)
            event_type = np.random.randint(0, 3, n_updates)
            event_types = np.bincount(event_type, minlength=3)
            new_exit_times = np.zeros(n_updates)
            new_exit_times[event_type == 0] = np.random.gamma(self.pars['value']['a'], self.pars['value']['b'], event_types[0])
            if (self.pars['value']['a'] == 4) & (self.pars['value']['b'] == 2):
                new_exit_times[event_type == 1] = stats.truncnorm.rvs(-0.6567 / 8.17, np.inf, loc=0.6567, scale=8.17, size=np.sum(event_type == 1))
            else:
                raise ValueError("Need to edit occult infection times for given a and b values")
            for i, event in enumerate(event_type):
                if event == 0:
                    #update existing infections
                    update_idx = np.random.randint(0, len(self.infected_farms))
                    self.old_exit_time = copy.deepcopy(self.rand_exits[update_idx])
                    if update_idx > len(self.data_farms):
                        new_exit_times[i] = stats.truncnorm.rvs(-0.6567 / 8.17, np.inf, loc=0.6567, scale=8.17)
                        proposal = (stats.gamma.cdf(self.old_exit_time,  a=self.pars['value']['a'], scale=self.pars['value']['b']) /
                                    stats.gamma.cdf(new_exit_times[i], a=self.pars['value']['a'], scale=self.pars['value']['b']))
                    else:
                        proposal = (stats.gamma.pdf(self.old_exit_time,  a=self.pars['value']['a'], scale=self.pars['value']['b']) /
                                    stats.gamma.pdf(new_exit_times[i], a=self.pars['value']['a'], scale=self.pars['value']['b']))
                    self.rand_exits[update_idx] = new_exit_times[i]
                    tmp = np.tile(self.mean_exits[:(self.data_comp_id[0] - 1)], [len(self.infected_farms), 1])
                    tmp[:, self.rand_exit_time_comp_id[self.rand_exit_time_comp_id < (self.data_comp_id[0] - 1)]] = np.round(
                        self.rand_exits)
                    new_exposed = self.report_day[update_idx] - np.sum(tmp[update_idx, :])
                    old_exposed = self.first_exposed
                    if np.min(self.report_day - np.sum(tmp, axis=1)) != old_exposed:
                        if new_exposed < old_exposed:
                            self.first_exposed = new_exposed
                        else:
                            self.first_exposed = np.min(self.report_day - np.sum(tmp, axis=1))
                        self.season_times = self.update_season_times()
                    self.exposure_rate = self.update_exposure_rate(update_type='exits', idx=update_idx)
                    new_neg_log_like = self.get_neg_log_like()
                    new_neg_log_post = new_neg_log_like + self.get_neg_log_prior()
                    if current_neg_log_post - new_neg_log_post - np.log(proposal) > np.log(np.random.uniform()):
                        print([event, 'accept', current_neg_log_post, new_neg_log_post])
                        current_neg_log_post = new_neg_log_post
                        accept_exit += 1
                    else:
                        print([event, 'reject', current_neg_log_post, new_neg_log_post])
                        self.rand_exits[update_idx] = copy.deepcopy(self.old_exit_time)
                        self.old_exit_time = np.array([new_exit_times[i]])
                        self.first_exposed = old_exposed
                        self.season_times = self.update_season_times()
                        self.exposure_rate = self.update_exposure_rate(update_type='exits', idx=update_idx)
                elif event == 1:
                    # add occult infection
                    new_farm = np.random.choice(np.setdiff1d(range(self.data.n_farms), np.append(self.infected_farms, self.past_infected_farms)))
                    self.infected_farms = np.append(self.infected_farms, new_farm)
                    self.rand_exits = np.append(self.rand_exits, [[new_exit_times[i]]], axis=0)
                    self.report_day = np.append(self.report_day, self.end_day)
                    old_exposed = self.first_exposed
                    tmp = np.tile(self.mean_exits[:(self.data_comp_id[0] - 1)], [len(self.infected_farms), 1])
                    tmp[:, self.rand_exit_time_comp_id[self.rand_exit_time_comp_id < (self.data_comp_id[0] - 1)]] = np.round(
                        self.rand_exits)
                    new_exposed = self.end_day - np.sum(tmp[-1, :])
                    if new_exposed < old_exposed:
                        self.first_exposed = new_exposed
                        self.season_times = self.update_season_times()
                    self.exposure_rate = self.update_exposure_rate(update_type='add', idx=len(self.infected_farms) - 1)
                    new_neg_log_like = self.get_neg_log_like()
                    new_neg_log_post = new_neg_log_like + self.get_neg_log_prior()
                    proposal = (self.data.n_farms - (len(self.infected_farms) + len(self.past_infected_farms)) + 1) / (
                            (len(self.infected_farms) - len(self.data_farms)) * stats.truncnorm.pdf(new_exit_times[i], -0.6567 / 8.17,
                                                                                    np.inf, loc=0.6567, scale=8.17))
                    if current_neg_log_post - new_neg_log_post  + np.log(proposal) > np.log(np.random.uniform()):
                        print([event, 'accept', current_neg_log_post, new_neg_log_post])
                        current_neg_log_post = new_neg_log_post
                        accept_add += 1
                    else:
                        print([event, 'reject', current_neg_log_post, new_neg_log_post])
                        if self.first_exposed != old_exposed:
                            self.season_times = self.update_season_times()
                            self.first_exposed = old_exposed
                        self.exposure_rate = self.update_exposure_rate(update_type='remove', idx=len(self.infected_farms) - 1)
                        self.infected_farms = self.infected_farms[:-1]
                        self.rand_exits = self.rand_exits[:-1]
                        self.report_day = self.report_day[:-1]
                elif (event == 2) and (len(self.infected_farms) > len(self.data_farms)):
                    # remove occult infection
                    remove_idx = np.random.randint(len(self.data_farms), len(self.infected_farms))
                    remove_farm = copy.deepcopy(self.infected_farms[remove_idx])
                    remove_exit_time = copy.deepcopy(self.rand_exits[remove_idx])
                    old_exposed = self.first_exposed
                    tmp = np.tile(self.mean_exits[:(self.data_comp_id[0] - 1)], [len(self.infected_farms), 1])
                    tmp[:, self.rand_exit_time_comp_id[self.rand_exit_time_comp_id < (self.data_comp_id[0] - 1)]] = np.round(
                        self.rand_exits)
                    new_exposed = np.min(np.delete(self.report_day, remove_idx) - np.sum(np.delete(tmp, remove_idx, axis=0), axis=1))
                    if new_exposed > old_exposed:
                        self.first_exposed = new_exposed
                        self.season_times = self.update_season_times()
                    self.exposure_rate = self.update_exposure_rate(update_type='remove', idx=remove_idx)
                    self.infected_farms = np.delete(self.infected_farms, remove_idx)
                    self.rand_exits = np.delete(self.rand_exits, remove_idx)[:, np.newaxis]
                    self.report_day = np.delete(self.report_day, remove_idx)
                    new_neg_log_like = self.get_neg_log_like()
                    new_neg_log_post = new_neg_log_like + self.get_neg_log_prior()
                    proposal = ((len(self.infected_farms) - len(self.data_farms) + 1) *
                                stats.truncnorm.pdf(remove_exit_time, -0.6567 / 8.17, np.inf, loc=0.6567, scale=8.17)) \
                               / (self.data.n_farms - len(self.infected_farms))
                    if current_neg_log_post - new_neg_log_post + np.log(proposal) > np.log(np.random.uniform()):
                        print([event, 'accept', current_neg_log_post, new_neg_log_post])
                        current_neg_log_post = new_neg_log_post
                        accept_rem += 1
                    else:
                        print([event, 'reject', current_neg_log_post, new_neg_log_post])
                        if self.first_exposed != old_exposed:
                            self.season_times = self.update_season_times()
                            self.first_exposed = old_exposed
                        self.infected_farms = np.insert(self.infected_farms, remove_idx, remove_farm)
                        self.rand_exits = np.insert(self.rand_exits, remove_idx, remove_exit_time)[:, np.newaxis]
                        self.report_day = np.insert(self.report_day, remove_idx, self.end_day)
                        self.exposure_rate = self.update_exposure_rate(update_type='add', idx=remove_idx)
            update_exit += event_types[0]
            update_add += event_types[1]
            update_rem += event_types[2]
            for name in self.to_fit:
                self.pars_chain[iter + 1][name] = copy.deepcopy(self.pars['value'][name])
            self.neg_log_like_chain[iter + 1] = current_neg_log_post
            self.neg_log_post_chain[iter + 1] = current_neg_log_post
            self.num_occult_chain[iter + 1] = len(self.infected_farms) - len(self.data_farms)
            self.infected_chain[iter + 1] = self.infected_farms
            self.exit_chain[iter + 1] = self.rand_exits.flatten()
            self.mean_exit_chain[iter + 1] = np.mean(self.rand_exits)
            if np.isin(iter + 1, save_tmp):
                if self.save:
                    self.save_chain(chain_num, save_tmp=iter + 1)
                self.acceptance = accept / self.max_iter
                if update_exit > 0:
                    self.acceptance_exit = accept_exit / update_exit
                    self.updates_exit = update_exit
                if update_add > 0:
                    self.acceptance_add = accept_add / update_add
                    self.updates_add = update_add
                if update_rem > 0:
                    self.acceptance_rem = accept_rem / update_rem
                    self.updates_rem = update_rem
        self.acceptance = accept / self.max_iter
        if update_exit > 0:
            self.acceptance_exit = accept_exit / update_exit
            self.updates_exit = update_exit
        if update_add > 0:
            self.acceptance_add = accept_add / update_add
            self.updates_add = update_add
        if update_rem > 0:
            self.acceptance_rem = accept_rem / update_rem
            self.updates_rem = update_rem
        if self.save:
            self.save_chain(chain_num)

    def simulate_model(self, reps=1000, with_post=True, max_days=None, pars_sim=None, future_info=100, initial_exposure=1, chains=None, start_day=None, end_day=None, cond_sample=True, include_total=False, biosecurity_level=None, biosecurity_duration=None, biosecurity_zone=None):
        """Run projections of the model."""
        self.reps = reps
        self.with_post = with_post
        if self.with_post:
            self.post_idx = np.random.choice(range(self.neg_log_post_post.shape[0]), size=reps, replace=True)
        if start_day is None:
            self.start_day = 0
        else:
            self.start_day = start_day
        if self.start_day < 0:
            raise ValueError("start_day must be greater than or equal to data start day.")
        if end_day is None:
            self.end_day = self.data.end_day
        else:
            self.end_day = end_day
        if self.end_day > self.data.end_day:
            raise ValueError("end_day must be less than or equal to data end day.")
        if max_days is None:
            self.max_days = self.end_day - self.start_day + 1
        else:
            self.max_days = max_days
        if self.with_post:
            if pars_sim is not None:
                raise ValueError("pars_sim must be None when with_post is True.")
        else:
            if pars_sim is None:
                raise ValueError("pars_sim must be provided when with_post is False.")
            else:
                self.pars_sim = pars_sim
        self.future_info = future_info
        self.initial_exposure = initial_exposure
        self.cond_sample = cond_sample
        if chains is None:
            self.chains = np.arange(self.pars_chains['epsilon'].shape[0])
        else:
            self.chains = chains
        self.time = np.arange(self.start_day, self.end_day + 1)
        self.exposure_day = -1e5 * np.ones((reps, self.data.n_farms), dtype=int)
        self.background = np.zeros((reps, self.data.n_farms))
        self.include_total = include_total
        self.biosecurity_level = biosecurity_level
        self.biosecurity_duration = biosecurity_duration
        self.biosecurity_zone = biosecurity_zone
        if not ((self.biosecurity_level is not None and self.biosecurity_duration is not None and self.biosecurity_zone is not None) or
                (self.biosecurity_level is None and self.biosecurity_duration is None and self.biosecurity_zone is None)):
                raise ValueError("biosecurity_level, biosecurity_duration and biosecurity_zone must all be provided.")
        elif self.biosecurity_level is not None:
            self.biosecurity = True
        else:
            self.biosecurity = False

        self.farm_status_reps = np.zeros((self.end_day - self.start_day + 1, self.n_comps, self.reps), dtype=int)
        self.total_infected = np.zeros(self.reps)
        self.notified_day = -1e5 * np.ones((reps, self.data.n_farms), dtype=int)

        all_times = (np.arange(self.end_day + 1) + self.data.date_start.timetuple().tm_yday) % 365
        # Simulate multiple replicates
        for rep in range(reps):
            if self.biosecurity:
                self.biosecurity_times = np.inf * np.ones(self.data.n_farms)
            self.season_times = self.update_season_times(all_times)[:-1]
            self.farm_status = np.zeros((self.end_day - self.start_day + 1, self.data.n_farms), dtype=int)
            self.farm_status_days = np.zeros(self.data.n_farms, dtype=int)
            print('Simulating rep ' + str(rep + 1) + ' of ' + str(self.reps) + '...')
            self.all_exits = np.tile(self.mean_exits, [self.data.n_farms, 1])
            if self.cull_data:
                self.all_exits[:, self.data_comp_id[0] - 1] = np.random.choice(self.data.cull_times[self.data.report_day >= 0], self.data.n_farms)
            if self.with_post:
                fit_a, _, fit_b = stats.gamma.fit(self.exit_post[self.post_idx[rep], :len(self.data_farms)], floc=0)
                for i in range(len(self.rand_exit_time_comp_id)):
                    self.all_exits[:, self.rand_exit_time_comp_id[i]] = np.round(
                        np.random.gamma(fit_a, fit_b, self.data.n_farms))
                for name in self.to_fit:
                    self.pars['value'][name] = self.pars_post[name][self.post_idx[rep]]
                self.infected_farms = self.infected_post[self.post_idx[rep]]
                self.rand_exits = self.exit_post[self.post_idx[rep]]
                self.all_exits[self.infected_post[self.post_idx[rep]][~np.isinf(self.infected_post[self.post_idx[rep]])].astype(int), self.rand_exit_time_comp_id] = np.round(self.rand_exits[~np.isinf(self.rand_exits)]).astype(int)
                tmp = np.tile(self.mean_exits, [np.sum(~np.isinf(self.infected_post[self.post_idx[rep]])), 1])
                tmp[:, self.rand_exit_time_comp_id[self.rand_exit_time_comp_id < (self.data_comp_id[0] - 1)]] = np.round(self.rand_exits[~np.isinf(self.rand_exits)]).astype(int)[:, np.newaxis]
                tmp = np.insert(np.cumsum(tmp, axis=1), 0, 0, axis=1)
                tmp -= tmp[:, self.data_comp_id - 1].reshape((-1, 1))
                tmp[:len(self.data_farms), :] += self.report_day[:, np.newaxis]
                tmp[len(self.data_farms):, :] += self.end_day
                self.non_susceptible_farms = self.infected_post[self.post_idx[rep]][~np.isinf(self.infected_post[self.post_idx[rep]])][(tmp[:, 0] <= 0) & (tmp[:, self.data_comp_id[0] - 1] < self.future_info)].astype(int)
                tmp0 = tmp[(tmp[:, 0] <= 0) & (tmp[:, self.data_comp_id[0] - 1] < self.future_info), :]
                self.farm_status[0, self.non_susceptible_farms] = np.array([np.argmax(row > 0) for row in tmp0])
                tmp2 = -np.array([row[np.argmax(row > 0) - 1] for row in tmp0])
                self.farm_status_days[self.non_susceptible_farms] = tmp2
                if (self.transmission_type >= 3) and (self.data.past_date_start < self.data.date_start):
                    self.farm_status[start_day, self.past_infected_farms] = self.n_comps - 1
                    self.farm_status_days[self.past_infected_farms] = -self.data.report_day[(self.data.report_day <= -self.mean_exits[-1]) & (self.data.report_day >= -self.data.past_start_day)] - self.mean_exits[-1]
            else:
                self.pars['value'] = self.pars_sim
                self.exposed_farms = np.random.choice(range(self.data.n_farms), size=self.initial_exposure, replace=False)
                self.farm_status[0, self.exposed_farms] = 1
                for i in range(len(self.rand_exit_time_comp_id)):
                    self.all_exits[:, self.rand_exit_time_comp_id[i]] = np.round(
                        np.random.gamma(self.pars['value']['a'], self.pars['value']['b'], self.data.n_farms))
            if self.biosecurity:
                self.biosecurity_times[self.farm_status[0] == 4] = self.farm_status_days[self.farm_status[0] == 4]
            self.exposure_day[rep, self.farm_status[0, :] == 1] = - self.farm_status_days[self.farm_status[0, :] == 1]
            for i in range(self.n_comps - 2):
                self.exposure_day[rep, self.farm_status[0, :] == i + 2] = - self.farm_status_days[self.farm_status[0, :] == i + 2] - np.cumsum(self.all_exits[self.farm_status[0, :] == i + 2, :], axis=1)[:, i]
            if not self.transmission_type == 0:
                self.sus = np.sum(np.array(self.pars['value']['zeta'])[:, np.newaxis] * (
                        self.data.pop_over_mean ** np.array(self.pars['value']['phi'])[:, np.newaxis]), axis=0)
                self.max_sus_grid = np.zeros(self.data.n_grids)
                for i in range(self.data.n_grids):
                    self.max_sus_grid[i] = np.max(self.sus[self.data.farm_grid == i])
                self.transmission = self.pars['value']['gamma'][0] * np.sum(
                    np.array(self.pars['value']['xi'])[:, np.newaxis] * (
                            self.data.pop_over_mean ** np.array(
                        self.pars['value']['psi'])[:, np.newaxis]), axis=0)
                self.transmission_scale = self.pars['value']['gamma'] / self.pars['value']['gamma'][0]
                if self.transmission_type >= 3:
                    self.transmission_scale = np.append(self.transmission_scale, self.transmission_scale[-1])
                self.max_trans_grid = np.zeros(self.data.n_grids)
                for i in range(self.data.n_grids):
                    self.max_trans_grid[i] = np.max(self.transmission[self.data.farm_grid == i])
                self.max_trans_grid = self.max_trans_grid * np.max(self.transmission_scale)
                self.u_ab = 1 - np.exp(
                    -self.max_sus_grid * self.max_trans_grid[:, np.newaxis] * self.kernel(self.data.dist2))
                self.max_rate_grid = self.max_sus_grid * self.kernel(self.data.dist2)
            self.background[rep, self.farm_status[0, :] != 0] = 1
            # Simulate forward in time
            for t in range(self.start_day, self.end_day):
                time_in_year = (self.data.date_start.timetuple().tm_yday + t) % 365
                season_time = self.update_season_times(time_in_year)
                expose_event = 1 - np.exp(-self.pars['value']['epsilon'] * season_time) > np.random.rand(self.data.n_farms)
                self.background[rep, expose_event] = 1
                other_events = np.zeros(self.data.n_farms)
                farm_status_t = self.farm_status[t, :]
                infected_farms = np.where((0 < farm_status_t) & (farm_status_t < self.n_comps - 1))[0]
                move_comp_check = self.farm_status_days[infected_farms] >= self.all_exits[
                    infected_farms, farm_status_t[infected_farms] - 1] - 1
                multiple_move_check = ((farm_status_t[infected_farms] == self.rand_exit_time_comp_id[0])
                                       & (self.all_exits[infected_farms, self.rand_exit_time_comp_id[0]] == 0))
                other_events[infected_farms[move_comp_check]] = 1
                other_events[infected_farms[move_comp_check & multiple_move_check]] = 2
                if self.biosecurity:
                    biosecurity_sus = (self.biosecurity_times < self.biosecurity_duration) * self.biosecurity_level + (self.biosecurity_times >= self.biosecurity_duration)
                if (self.transmission_type == 1) or (self.transmission_type == 2) or (self.transmission_type > 4):
                    season_time = 1
                if self.cond_sample and not self.transmission_type == 0:
                    infectious_farms = np.where(np.isin(farm_status_t, self.inf_comps_id))[0]
                    if self.transmission_type >= 3:
                        max_decay_days = np.where(np.exp(-(1 / self.pars['value']['rho']) * np.arange(500)) > 1e-20)[0][-1]
                        infectious_farms = np.append(infectious_farms, np.where((farm_status_t == self.n_comps - 1) & (self.farm_status_days <= max_decay_days))[0])
                    infectious_grids, inf_in_grid = np.unique(self.data.farm_grid[infectious_farms], return_counts=True)
                    sus_farms = farm_status_t == 0
                    sus_in_grid = np.bincount(self.data.farm_grid[sus_farms], minlength=self.data.n_grids)
                    sus_grids = np.where(sus_in_grid > 0)[0]
                    for a_i, a in enumerate(infectious_grids):
                        N_a = infectious_farms[self.data.farm_grid[infectious_farms] == a]
                        n_a = len(N_a)
                        w_ab = 1 - (1 - self.u_ab[a, :]) ** n_a
                        for b in sus_grids[sus_grids != a]:
                            n_b = sus_in_grid[b]
                            n_sample = np.random.binomial(n_b, w_ab[b])
                            if n_sample > 0:
                                N_b = np.where((self.data.farm_grid == b) & sus_farms)[0]
                                N_sample = np.random.choice(N_b, n_sample, replace=False)
                                K_ij = self.kernel((self.data.location_x[N_a][:, np.newaxis] - self.data.location_x[N_sample]) ** 2 + \
                                                 (self.data.location_y[N_a][:, np.newaxis] - self.data.location_y[N_sample]) ** 2)
                                p_ij = np.zeros((n_a, n_sample))
                                if self.biosecurity:
                                    s_bio = self.sus[N_sample] * biosecurity_sus[N_sample]
                                else:
                                    s_bio = self.sus[N_sample]
                                p_ij[farm_status_t[N_a] != (self.n_comps - 1), :] = 1 - np.exp(-s_bio * self.transmission_scale[farm_status_t[N_a[farm_status_t[N_a] != (self.n_comps - 1)]] - self.inf_comps_id[0]][:, np.newaxis] *
                                                  self.transmission[N_a[farm_status_t[N_a] != (self.n_comps - 1)]][:, np.newaxis] * season_time * K_ij[farm_status_t[N_a] != (self.n_comps - 1)])
                                if self.transmission_type >= 3:
                                    days_since = np.exp(-(1/self.pars['value']['rho']) * (1 + self.farm_status_days[N_a[farm_status_t[N_a] == (self.n_comps - 1)]]))[:, np.newaxis]
                                    p_ij[farm_status_t[N_a] == (self.n_comps - 1), :] = 1 - np.exp(-s_bio * self.transmission_scale[
                                                                                farm_status_t[N_a[farm_status_t[N_a] == (self.n_comps - 1)]] - self.inf_comps_id[0]][:, np.newaxis] * self.transmission[N_a[farm_status_t[N_a] == (self.n_comps - 1)]][:,
                                                      np.newaxis] * days_since * season_time * K_ij[farm_status_t[N_a] == (self.n_comps - 1)])
                                p_aj = 1 - np.prod(1 - p_ij, axis=0)
                                exp_mask = N_sample[np.random.rand(n_sample) < p_aj / w_ab[b]]
                                expose_event[exp_mask] = True
                                self.background[rep, exp_mask] = 2
                        N_b = np.where((self.data.farm_grid == a) & sus_farms)[0]
                        n_b = sus_in_grid[a]
                        K_ij = self.kernel(
                            (self.data.location_x[N_a][:, np.newaxis] - self.data.location_x[N_b]) ** 2 + \
                            (self.data.location_y[N_a][:, np.newaxis] - self.data.location_y[N_b]) ** 2)
                        p_ij = np.zeros((n_a, n_b))
                        if self.biosecurity:
                            s_bio = self.sus[N_b] * biosecurity_sus[N_b]
                        else:
                            s_bio = self.sus[N_b]
                        p_ij[farm_status_t[N_a] != (self.n_comps - 1), :] = 1 - np.exp(
                            -s_bio * self.transmission_scale[farm_status_t[N_a[
                                farm_status_t[N_a] != (self.n_comps - 1)]] - self.inf_comps_id[0]][:,
                                                  np.newaxis] *
                            self.transmission[N_a[farm_status_t[N_a] != (self.n_comps - 1)]][:,
                            np.newaxis] * season_time * K_ij[
                                farm_status_t[N_a] != (self.n_comps - 1)])
                        if self.transmission_type >= 3:
                            days_since = np.exp(-(1/self.pars['value']['rho']) * (1 + self.farm_status_days[
                                N_a[farm_status_t[N_a] == (self.n_comps - 1)]]))[:, np.newaxis]
                            p_ij[farm_status_t[N_a] == (self.n_comps - 1), :] = 1 - np.exp(
                                -s_bio * self.transmission_scale[farm_status_t[N_a[farm_status_t[N_a] == (self.n_comps - 1)]] - self.inf_comps_id[0]][:, np.newaxis] * self.transmission[N_a[farm_status_t[N_a] == (self.n_comps - 1)]][:, np.newaxis] * days_since * season_time *
                                K_ij[farm_status_t[N_a] == (self.n_comps - 1)])
                        p_aj = 1 - np.prod(1 - p_ij, axis=0)
                        exp_mask = N_b[np.random.rand(n_b) < p_aj]
                        expose_event[exp_mask] = True
                        self.background[rep, exp_mask] = 2
                elif not self.transmission_type == 0:
                    raise ValueError("Only conditional sampling is implemented.")
                self.farm_status_days[np.where((farm_status_t > 0) & (farm_status_t <= self.n_comps - 1))[0]] += 1
                self.farm_status[t + 1, :] = farm_status_t
                self.farm_status[t + 1, (farm_status_t == 0) & expose_event] = 1
                self.farm_status[t + 1, other_events == 1] += 1
                self.farm_status[t + 1, other_events == 2] += 2
                self.farm_status_days[other_events > 0] = 0
                self.farm_status_days[(farm_status_t == 0) & expose_event] = 1
                self.exposure_day[rep, expose_event] = t
                if self.biosecurity:
                    self.biosecurity_times[~np.isinf(self.biosecurity_times)] += 1
                    new_biosecurity = np.where((self.farm_status[t + 1, :] == 4) & (self.farm_status[t, :] == 3))[0]
                    if len(new_biosecurity) > 0:
                        self.add_biosecurity(new_biosecurity)

            self.farm_status_reps[:, :, rep] = np.apply_along_axis(np.bincount, 1, self.farm_status, minlength=self.n_comps)
            self.total_infected[rep] = self.data.n_farms - np.sum(farm_status_t == 0)
            self.notified_day[rep, self.exposure_day[rep, :] > -1e5] = self.exposure_day[rep, self.exposure_day[rep, :] > -1e5] + self.mean_exits[0] + self.all_exits[self.exposure_day[rep, :] > -1e5, 1]

    def add_biosecurity(self, farms):
        """Add biosecurity to premises."""
        if isinstance(self.biosecurity_zone, (int, float)):
            # Get a ring around self.
            farms_in = np.hstack([self.data.location_tree.query_ball_point(
                (self.data.location_x[f], self.data.location_y[f]), self.biosecurity_zone
            ) for f in farms])
        elif self.biosecurity_zone == 'region':
            farms_in = np.where(np.isin(self.data.region, self.data.region[farms]))[0]
        elif self.biosecurity_zone == 'county':
            farms_in = np.where(np.isin(self.data.county, self.data.county[farms]))[0]
        else:
            raise ValueError("biosecurity_zone must be 'region', 'county' or a number.")
        self.biosecurity_times[farms_in] = 0


    def save_chain(self, chain_num, save_tmp=None, directory='../outputs/'):
        """Save outputs from the MCMC."""
        if save_tmp is None:
            total_iter = self.max_iter
        else:
            total_iter = save_tmp
        accept = np.array([self.acceptance, self.acceptance_exit, self.acceptance_add, self.acceptance_rem, self.updates_exit, self.updates_add, self.updates_rem])
        np.savetxt(directory + 'acceptance_rate_' + self.chain_string + str(chain_num) + '.txt', accept, fmt='%12.6f')
        tmp0 = np.zeros([self.n_to_fit, total_iter + 1])
        for j in range(total_iter + 1):
            name_i = 0
            for name in self.to_fit:
                for i in range(np.sum(self.pars['fitting'][name])):
                    tmp0[name_i + i, j] = self.pars_chain[j][name][np.where(self.pars['fitting'][name])[0][i]]
                name_i += np.sum(self.pars['fitting'][name])
        self.pars_chain_save = tmp0

        tmp1 = np.inf * np.ones((len(self.data_farms) + np.max(self.num_occult_chain).astype(int), total_iter + 1))
        for j, d in enumerate(self.infected_chain[:total_iter + 1]):
            tmp1[:len(d), j] = d
        self.infected_chain_save = tmp1

        tmp2 = np.inf * np.ones((len(self.data_farms) + np.max(self.num_occult_chain).astype(int), total_iter + 1))
        for j, d in enumerate(self.exit_chain[:total_iter + 1]):
            tmp2[:len(d), j] = d
        self.exit_chain_save = tmp2

        np.save(directory + 'chain_' + self.chain_string + str(chain_num) + '.npy', self.pars_chain_save)
        np.save(directory + 'infected_chain_' + self.chain_string + str(chain_num) + '.npy', self.infected_chain_save)
        np.save(directory + 'exit_chain_' + self.chain_string + str(chain_num) + '.npy', self.exit_chain_save)
        np.save(directory + 'num_occult_chain_' + self.chain_string + str(chain_num) + '.npy', self.num_occult_chain[:total_iter + 1])
        np.save(directory + 'mean_exit_chain_' + self.chain_string + str(chain_num) + '.npy', self.mean_exit_chain[:total_iter + 1])
        np.save(directory + 'neg_log_like_chain_' + self.chain_string + str(chain_num) + '.npy', self.neg_log_like_chain[:total_iter + 1])
        np.save(directory + 'neg_log_post_chain_' + self.chain_string + str(chain_num) + '.npy', self.neg_log_post_chain[:total_iter + 1])

    def load_chains(self, chain_nums, max_iter, to_fit=None, directory='../outputs/'):
        """Load the MCMC chains."""
        self.max_iter = max_iter
        self.chain_string = (self.data.date_start.strftime('%Y%m%d') + '_' + self.data.date_end.strftime('%Y%m%d') +
             '_' + str(self.max_iter) + '_t' + str(self.transmission_type) + '_' + self.kernel_type +
             '_c' + str(self.combine) + '_s' + str(self.spatial) + '_')
        if self.cull_data:
            self.chain_string += 'cull' + str(self.cull_data) + '_'
        if self.data.select_region is not None:
            self.chain_string += self.data.region_names[self.data.select_region].replace(" ", "_") + '_'
        if to_fit is None:
            if self.spatial:
                if self.transmission_type == 0:
                    self.to_fit = ['epsilon']
                elif (self.transmission_type == 1) or (self.transmission_type == 2):
                    self.to_fit = ['epsilon', 'gamma', 'delta', 'psi', 'phi', 'xi', 'zeta']
                elif (self.transmission_type == 3) or (self.transmission_type == 4):
                    self.to_fit = ['epsilon', 'gamma', 'delta', 'psi', 'phi', 'xi', 'zeta', 'rho']
                elif self.transmission_type == 5:
                    self.to_fit = ['epsilon', 'gamma', 'delta', 'psi', 'phi', 'xi', 'zeta']
            else:
                if self.transmission_type == 0:
                    self.to_fit = ['epsilon', 'nu']
                elif (self.transmission_type == 1) or (self.transmission_type == 2):
                    self.to_fit = ['epsilon', 'gamma', 'delta', 'psi', 'phi', 'xi', 'zeta', 'nu']
                elif (self.transmission_type == 3) or (self.transmission_type == 4):
                    self.to_fit = ['epsilon', 'gamma', 'delta', 'psi', 'phi', 'xi', 'zeta', 'nu', 'rho']
                elif self.transmission_type == 5:
                    self.to_fit = ['epsilon', 'gamma', 'delta', 'psi', 'phi', 'xi', 'zeta', 'nu']
        else:
            self.to_fit = to_fit
        for name in self.to_fit:
            if name not in self.par_names:
                raise ValueError("Invalid parameter name: " + name + " not in par_names.")
            else:
                if (name == 'xi') or (name == 'zeta'):
                    self.pars['fitting'][name] = np.array([False] + [True] * (self.pars['length'][name] - 1))
                else:
                    self.pars['fitting'][name] = np.array([True] * self.pars['length'][name])
        self.n_to_fit = 0
        for name in self.par_names:
            self.n_to_fit += np.sum(self.pars['fitting'][name])


        self.pars_chains = np.zeros((len(chain_nums), self.max_iter + 1, self.n_to_fit))
        self.infected_chains = [None for _ in chain_nums]
        self.exit_chains = [None for _ in chain_nums]
        self.mean_exit_chains = np.zeros((len(chain_nums), self.max_iter + 1))
        self.num_occult_chains = np.zeros((len(chain_nums), self.max_iter + 1))
        self.neg_log_like_chains = np.zeros((len(chain_nums), self.max_iter + 1))
        self.neg_log_post_chains = np.zeros((len(chain_nums), self.max_iter + 1))
        for i, chain in enumerate(chain_nums):
            self.pars_chains[i, :, :] = np.load(directory + 'chain_' + self.chain_string + str(chain) + '.npy').T
            self.infected_chains[i] = np.load(directory + 'infected_chain_' + self.chain_string + str(chain) + '.npy').T
            self.exit_chains[i] = np.load(directory + 'exit_chain_' + self.chain_string + str(chain) + '.npy').T
            self.num_occult_chains[i, :] = np.load(directory + 'num_occult_chain_' + self.chain_string + str(chain) + '.npy')
            self.mean_exit_chains[i, :] = np.load(directory + 'mean_exit_chain_' + self.chain_string + str(chain) + '.npy')
            self.neg_log_like_chains[i, :] = np.load(directory + 'neg_log_like_chain_' + self.chain_string + str(chain) + '.npy')
            self.neg_log_post_chains[i, :] = np.load(directory + 'neg_log_post_chain_' + self.chain_string + str(chain) + '.npy')

        tmp = {name: np.zeros((len(chain_nums), self.max_iter + 1, np.sum(self.pars['fitting'][name]))) for name in self.to_fit}
        for i, sublist in enumerate(self.pars_chains):
            for j, d in enumerate(sublist):
                name_start = 0
                for name in self.to_fit:
                    name_end = name_start + np.sum(self.pars['fitting'][name])
                    tmp[name][i, j, :] = d[name_start:name_end]
                    name_start = name_end
        self.pars_chains = tmp

        tmp1 = np.inf * np.ones((len(chain_nums), self.max_iter + 1, len(self.data_farms) + np.max(self.num_occult_chains).astype(int)))
        tmp2 = np.inf * np.ones((len(chain_nums), self.max_iter + 1, len(self.data_farms) + np.max(self.num_occult_chains).astype(int)))
        for i in range(len(chain_nums)):
            tmp1[i, :, :self.infected_chains[i].shape[1]] = self.infected_chains[i]
            tmp2[i, :, :self.exit_chains[i].shape[1]] = self.exit_chains[i]
        self.infected_chains = tmp1
        self.exit_chains = tmp2

    def save_projections(self, directory='../outputs/', max_iter=None, save_full=False):
        """Save projections."""
        if max_iter is not None:
            self.max_iter = max_iter
        date_start = self.data.date_start + pd.DateOffset(days=int(self.start_day))
        date_end = self.data.date_start + pd.DateOffset(days=int(self.end_day))
        self.chain_string = (date_start.strftime('%Y%m%d') + '_' + date_end.strftime('%Y%m%d') +
                             '_' + str(self.max_iter) + '_' + 't' + str(self.transmission_type) + '_' + self.kernel_type
                             + '_c' + str(self.combine) + '_s' + str(self.spatial) + '_')
        if self.cull_data:
            self.chain_string += 'cull' + str(self.cull_data) + '_'
        if self.data.select_region is not None:
            self.chain_string += self.data.region_names[self.data.select_region].replace(" ", "_") + '_'
        if self.biosecurity:
            self.chain_string += 'bl' + str(self.biosecurity_level) + '_' + 'bd' + str(self.biosecurity_duration) + '_' + 'bz' + str(self.biosecurity_zone) + '_'
        np.save(directory + 'proj_' + self.chain_string + 'reps_' + str(self.reps) + '.npy', self.farm_status_reps)
        if save_full:
            np.save(directory + 'notified_' + self.chain_string + 'reps_' + str(self.reps) + '.npy', self.notified_day)
        else:
            notif_times = self.notified_day[self.notified_day > -1e5]
            notif_times_rep = np.where(self.notified_day > -1e5)[0]
            notif_times_farm = np.where(self.notified_day > -1e5)[1]
            time_to_notif = self.notified_day[self.exposure_day > -1e5] - (self.exposure_day[self.exposure_day > -1e5] + self.mean_exits[0])
            np.save(directory + 'notified_times_' + self.chain_string + 'reps_' + str(self.reps) + '.npy',
                    notif_times)
            np.save(directory + 'notified_rep_' + self.chain_string + 'reps_' + str(self.reps) + '.npy',
                    notif_times_rep)
            np.save(directory + 'notified_farm_' + self.chain_string + 'reps_' + str(self.reps) + '.npy',
                    notif_times_farm)
            np.save(directory + 'time_to_notified_' + self.chain_string + 'reps_' + str(self.reps) + '.npy',
                    time_to_notif)
        np.save(directory + 'post_idx_' + self.chain_string + 'reps_' + str(self.reps) + '.npy', self.post_idx)

    def load_projections(self, reps, max_iter, date_start, date_end, directory='../outputs/', include_total=False, save_full=False, biosecurity_level=None, biosecurity_duration=None, biosecurity_zone=None):
        """Load projections."""
        self.max_iter = max_iter
        self.reps = reps
        self.include_total = include_total
        self.date_start = date_start
        self.date_end = date_end
        self.start_day = (self.date_start - self.data.date_start).days
        self.end_day = (self.date_end - self.data.date_start).days
        self.max_days = self.end_day - self.start_day + 1
        self.chain_string = (self.date_start.strftime('%Y%m%d') + '_' + self.date_end.strftime('%Y%m%d') +
                             '_' + str(self.max_iter) + '_' + 't' + str(self.transmission_type) + '_' + self.kernel_type
                             + '_c' + str(self.combine) + '_s' + str(self.spatial) + '_')
        if self.cull_data:
            self.chain_string += 'cull' + str(self.cull_data) + '_'
        if self.data.select_region is not None:
            self.chain_string += self.data.region_names[self.data.select_region].replace(" ", "_") + '_'
        if (biosecurity_level is not None) and (biosecurity_duration is not None) and (biosecurity_zone is not None):
            self.chain_string += 'bl' + str(biosecurity_level) + '_' + 'bd' + str(biosecurity_duration) + '_' + 'bz' + str(biosecurity_zone) + '_'
        elif not ((biosecurity_level is None) and (biosecurity_duration is None) and (biosecurity_zone is None)):
            raise ValueError("Must specify all biosecurity parameters.")
        self.farm_status_reps = np.load(directory + 'proj_' + self.chain_string + 'reps_' + str(reps) + '.npy')
        if save_full:
            self.notified_day = np.load(directory + 'notified_' + self.chain_string + 'reps_' + str(reps) + '.npy')
        else:
            self.notified_day = -1e5 * np.ones((reps, self.data.n_farms))
            self.notif_times = np.load(directory + 'notified_times_' + self.chain_string + 'reps_' + str(reps) + '.npy')
            self.notif_times_rep = np.load(directory + 'notified_rep_' + self.chain_string + 'reps_' + str(reps) + '.npy')
            self.notif_times_farm = np.load(directory + 'notified_farm_' + self.chain_string + 'reps_' + str(reps) + '.npy')
            self.time_to_notif = np.load(directory + 'time_to_notified_' + self.chain_string + 'reps_' + str(reps) + '.npy')
            self.notified_day[self.notif_times_rep, self.notif_times_farm] = self.notif_times
        self.post_idx = np.load(directory + 'post_idx_' + self.chain_string + 'reps_' + str(reps) + '.npy')

    def format_posts(self, chains=None, values_per_chain=1000, burn_in=1000):
        """Get posterior from the chains."""
        if chains is None:
            chains = np.arange(self.neg_log_like_chains.shape[0])
        if self.burn_in is None:
            self.burn_in = burn_in
            print('Burn in set to ' + str(self.burn_in))
        choose_samples = np.arange(self.burn_in + 1, self.max_iter + 1,
                                        (self.max_iter - self.burn_in - 1) / (values_per_chain - 1)).astype(int)
        self.neg_log_post_post = self.neg_log_post_chains[np.ix_(chains, choose_samples)].flatten()
        self.neg_log_like_post = self.neg_log_like_chains[np.ix_(chains, choose_samples)].flatten()
        self.mean_exit_post = self.mean_exit_chains[np.ix_(chains, choose_samples)].flatten()
        self.num_occult_post = self.num_occult_chains[np.ix_(chains, choose_samples)].flatten()
        self.pars_post = {name: np.tile(self.pars['value'][name], [len(chains) * values_per_chain, 1]) for name in self.par_names}
        for name in self.to_fit:
            self.pars_post[name][:, np.where(self.pars['fitting'][name])[0]] = self.pars_chains[name][np.ix_(chains, choose_samples, np.arange(np.sum(self.pars['fitting'][name])))].reshape(len(chains) * len(choose_samples), np.sum(self.pars['fitting'][name]))
        self.infected_post = self.infected_chains[np.ix_(chains, choose_samples, np.arange(self.infected_chains.shape[2]))].reshape(len(chains) * len(choose_samples), self.infected_chains.shape[2])
        self.exit_post = self.exit_chains[np.ix_(chains, choose_samples, np.arange(self.exit_chains.shape[2]))].reshape(len(chains) * len(choose_samples), self.exit_chains.shape[2])

    def get_neg_log_like(self):
        """Calculate the negative log likelihood of the model."""
        for name in self.to_fit:
            if np.sum(self.pars['value'][name] < 0) > 0:
                return np.inf
        log_like = 0
        tmp = np.tile(self.mean_exits[:(self.data_comp_id[0] - 1)], [len(self.infected_farms), 1])
        tmp[:, self.rand_exit_time_comp_id[self.rand_exit_time_comp_id < (self.data_comp_id[0] - 1)]] = np.round(
            self.rand_exits)
        exposure_times = self.report_day - np.sum(tmp, axis=1) - self.first_exposed
        for i in range(len(self.infected_farms)):
            log_like += np.log(self.exposure_rate[exposure_times[i], self.infected_farms[i]])
            log_like += -np.sum(self.exposure_rate[:exposure_times[i], self.infected_farms[i]])
        log_like += -np.sum(self.exposure_rate[:, np.setdiff1d(range(self.data.n_farms), np.append(self.infected_farms, self.past_infected_farms))])
        notif_like = stats.gamma.pdf(self.rand_exits[:len(self.data_farms)], self.pars['value']['a'], scale=self.pars['value']['b'])
        log_like += np.sum(np.log(notif_like))
        if len(self.data_farms) < len(self.infected_farms):
            log_like += np.sum(np.log(1 - stats.gamma.cdf(self.rand_exits[len(self.data_farms):], self.pars['value']['a'], scale=self.pars['value']['b'])))
        return -log_like

    def get_neg_log_prior(self):
        """Calculate the negative log prior of the model."""
        log_prior = 0
        for name in self.to_fit:
            for i in range(np.sum(self.pars['fitting'][name])):
                if self.pars['prior_type'][name][np.where(self.pars['fitting'][name])[0][i]] == 'gamma':
                    log_prior += stats.gamma.logpdf(self.pars['value'][name][np.where(self.pars['fitting'][name])[0][i]], a=self.pars['prior_pars'][name][np.where(self.pars['fitting'][name])[0][i], 0], scale=self.pars['prior_pars'][name][np.where(self.pars['fitting'][name])[0][i], 1])
                elif self.pars['prior_type'][name][np.where(self.pars['fitting'][name])[0][i]] == 'beta':
                    log_prior += stats.beta.logpdf(self.pars['value'][name][np.where(self.pars['fitting'][name])[0][i]], a=self.pars['prior_pars'][name][np.where(self.pars['fitting'][name])[0][i], 0], b=self.pars['prior_pars'][name][np.where(self.pars['fitting'][name])[0][i], 1])
        return -log_prior

    def update_season_times(self, time_in_year=None):
        """Get the season time."""
        if time_in_year is None:
            time_in_year = (np.arange(self.first_exposed, self.end_day + 1) + self.data.date_start.timetuple().tm_yday) % 365
        if self.spatial:
            quarter = time_in_year
            if type(quarter) == int:
                if quarter < 90:
                    quarter = 0
                elif (quarter >= 90) & (quarter < 181):
                    quarter = 1
                elif (quarter >= 180) & (quarter < 272):
                    quarter = 2
                else:
                    quarter = 3
            else:
                quarter[quarter < 90] = 0
                quarter[(quarter >= 90) & (quarter < 181)] = 1
                quarter[(quarter >= 180) & (quarter < 272)] = 2
                quarter[quarter >= 272] = 3
            return quarter
        else:
            return np.exp(-self.pars['value']['nu'][0] * (1 + np.cos(2 * np.pi * (time_in_year / 365 -
                                                                              self.pars['value']['nu'][1]))))

    def update_exposure_rate(self, update_type='pars', idx=None):
        """Calculate the exposure rate."""
        if update_type == 'pars':
            # Update if parameter values have changed
            exposure_rate = np.repeat(self.pars['value']['epsilon'] * self.season_times[:, np.newaxis],
                                       self.data.n_farms, axis=1)
            if self.transmission_type > 0:
                sus = np.sum(np.array(self.pars['value']['zeta'])[:, np.newaxis] * (
                        self.data.pop_over_mean ** np.array(self.pars['value']['phi'])[:, np.newaxis]), axis=0)
                transmission = self.pars['value']['gamma'][0] * np.sum(
                    np.array(self.pars['value']['xi'])[:, np.newaxis] * (
                                self.data.pop_over_mean[:, self.infected_farms] ** np.array(
                            self.pars['value']['psi'])[:, np.newaxis]), axis=0)
                transmission_scale = self.pars['value']['gamma'] / self.pars['value']['gamma'][0]
                farms_kernel = self.kernel(np.sum((self.xy_farms - self.xy_farms[:, :, self.infected_farms].reshape(2, len(self.infected_farms), 1)) ** 2, axis=0))
                tmp = np.tile(self.mean_exits[:self.data_comp_id[0]], [len(self.infected_farms), 1])
                tmp[:, self.rand_exit_time_comp_id[self.rand_exit_time_comp_id < (self.data_comp_id[0] - 1)]] = np.round(
                    self.rand_exits)
                if self.cull_data:
                    tmp[:self.data.n_data, -1] = self.data.cull_times[self.data.report_day >= 0]
                transitions = self.report_day[:, np.newaxis] + np.insert(np.cumsum(tmp, axis=1), 0, 0, axis=1) - np.cumsum(
                    tmp, axis=1)[:, (self.data_comp_id[0] - 2)][:, np.newaxis] - self.first_exposed
                transitions[len(self.data_farms):, self.data_comp_id[0] - 1] += 1
                for i in range(len(self.infected_farms)):
                    for j, inf_type in enumerate(self.inf_comps_id):
                        new_exposures = (transmission[i] * transmission_scale[j] * sus * farms_kernel[i, :])
                        if (self.transmission_type == 2) or (self.transmission_type == 4):
                            new_exposures = new_exposures[np.newaxis, :] * self.season_times[transitions[i, j + 1]:transitions[i, j + 2], np.newaxis]
                        exposure_rate[transitions[i, j + 1]:transitions[i, j + 2], :] += new_exposures
                    if self.transmission_type >= 3:
                        new_exposures_decay = transmission[i] * transmission_scale[-1] * sus * farms_kernel[i, :]
                        decay = (np.exp(
                            -np.arange(1, 1 + exposure_rate.shape[0] - transitions[i, -1]) / self.pars['value']['rho']))
                        if self.transmission_type == 4:
                            decay *= self.season_times[transitions[i, -1]:]
                        decay = decay[decay > 1e-15][:, np.newaxis]
                        new_exposures_decay = new_exposures_decay * decay
                        exposure_rate[transitions[i, -1]:(transitions[i, -1] + len(decay)), :] += new_exposures_decay
        elif self.transmission_type > 0:
            # Update if the number of infected farms or time to notification has changed
            sus = np.sum(np.array(self.pars['value']['zeta'])[:, np.newaxis] * (
                    self.data.pop_over_mean ** np.array(self.pars['value']['phi'])[:, np.newaxis]), axis=0)
            transmission = self.pars['value']['gamma'][0] * np.sum(
                np.array(self.pars['value']['xi'])[:, np.newaxis] * (
                        self.data.pop_over_mean[:, self.infected_farms[idx]][:, np.newaxis] ** np.array(
                    self.pars['value']['psi'])[:, np.newaxis]), axis=0)
            transmission_scale = self.pars['value']['gamma'] / self.pars['value']['gamma'][0]
            farms_kernel = self.kernel(
                np.sum((self.xy_farms - self.xy_farms[:, :, self.infected_farms[idx]].reshape(2, 1, 1)) ** 2, axis=0))
            new_exposures = transmission * sus * farms_kernel
            if update_type == 'exits':
                new_inf_col = self.report_day[idx] - (np.round(self.rand_exits[idx, 0]).astype(int) + self.first_exposed)
                old_inf_col = self.report_day[idx] - (np.round(self.old_exit_time[0]).astype(int) + self.first_exposed)
                if new_inf_col < old_inf_col:
                    if self.season_times.shape[0] > self.exposure_rate.shape[0]:
                        exposure_rate = np.vstack((np.repeat(self.pars['value']['epsilon'] * self.season_times[:(self.season_times.shape[0] - self.exposure_rate.shape[0]), np.newaxis],
                                                            self.data.n_farms, axis=1), self.exposure_rate))
                    else:
                        exposure_rate = copy.deepcopy(self.exposure_rate)
                    if (self.transmission_type == 2) or (self.transmission_type == 4):
                        exposure_rate[new_inf_col:old_inf_col, :] += new_exposures * self.season_times[new_inf_col:old_inf_col, np.newaxis]
                    else:
                        exposure_rate[new_inf_col:old_inf_col, :] += new_exposures
                else:
                    if old_inf_col < 0:
                        new_old_inf_col = 0
                    else:
                        new_old_inf_col = old_inf_col
                    if self.season_times.shape[0] < self.exposure_rate.shape[0]:
                        exposure_rate = copy.deepcopy(self.exposure_rate[(self.exposure_rate.shape[0]-self.season_times.shape[0]):, :])
                    else:
                        exposure_rate = copy.deepcopy(self.exposure_rate)
                    if (self.transmission_type == 2) or (self.transmission_type == 4):
                        exposure_rate[new_old_inf_col:new_inf_col, :] -= new_exposures * self.season_times[new_old_inf_col:new_inf_col, np.newaxis]
                    else:
                        exposure_rate[new_old_inf_col:new_inf_col, :] -= new_exposures
                if np.sum(exposure_rate <= 0) > 0:
                    print('error')
            elif update_type == 'add':
                new_inf_col = self.report_day[idx] - (np.round(self.rand_exits[idx, 0]).astype(int) + self.first_exposed)
                if self.season_times.shape[0] > self.exposure_rate.shape[0]:
                    exposure_rate = np.vstack((np.repeat(self.pars['value']['epsilon'] * self.season_times[:(self.season_times.shape[0] - self.exposure_rate.shape[0])],
                                                        self.data.n_farms, axis=1), self.exposure_rate))
                else:
                    exposure_rate = copy.deepcopy(self.exposure_rate)
                if (self.transmission_type == 2) or (self.transmission_type == 4):
                    exposure_rate[new_inf_col:, :] += new_exposures * self.season_times[new_inf_col:, np.newaxis]
                else:
                    exposure_rate[new_inf_col:, :] += new_exposures
                if np.sum(exposure_rate <= 0) > 0:
                    print('error')
            elif update_type == 'remove':
                new_inf_col = self.report_day[idx] - (np.round(self.rand_exits[idx, 0]).astype(int) + self.first_exposed)
                if self.season_times.shape[0] < self.exposure_rate.shape[0]:
                    exposure_rate = copy.deepcopy(self.exposure_rate[(self.exposure_rate.shape[0] - self.season_times.shape[0]):, :])
                else:
                    exposure_rate = copy.deepcopy(self.exposure_rate)
                if (self.transmission_type == 2) or (self.transmission_type == 4):
                    exposure_rate[new_inf_col:, :] -= new_exposures * self.season_times[new_inf_col:, np.newaxis]
                else:
                    exposure_rate[new_inf_col:, :] -= new_exposures
                if np.sum(exposure_rate <= 0) > 0:
                    print('error')
        else:
            exposure_rate = self.exposure_rate
        if np.sum(exposure_rate <= 0) > 0:
            print('error')
        return exposure_rate

    def symmetric_pos_def(self, matrix):
        """Make a matrix symmetric positive definite"""
        if isinstance(matrix, (list, np.ndarray)) and len(matrix) > 1:
            matrix = (matrix + matrix.T) / 2
            e_vals, e_vecs = np.linalg.eig(matrix)
            if sum(e_vals < 0) != 0:
                s_vals = sum(e_vals[e_vals < 0]) * 2
                t_vals = (s_vals * s_vals * 100) + 1
                if e_vals[e_vals > 0].size == 0:
                    return matrix
                else:
                    p_vals = min(e_vals[e_vals > 0])
                n_vals = e_vals[e_vals < 0]
                nn_vals = p_vals * (s_vals - n_vals) * (s_vals - n_vals) / t_vals
                d_vals = np.copy(e_vals)
                d_vals[d_vals <= 0] = nn_vals
                d_vals = np.diag(d_vals)
                matrix = np.matmul(np.matmul(e_vecs, d_vals), e_vecs.T)
        return matrix

    def plot_figure_1(self, by_region=False, new=True):
        if by_region:
            fig, ax = plt.subplots(3, 6, figsize=(14, 6))
            ax_left = plt.subplot2grid((3, 6), (0, 0), rowspan=3, colspan=2)
            ax[0, 0].axis('off')
            ax[1, 0].axis('off')
            ax[2, 0].axis('off')
            ax[0, 1].axis('off')
            ax[1, 1].axis('off')
            ax[2, 1].axis('off')
        else:
            fig, ax = plt.subplots(1, 3, figsize=(12, 4))
            ax_left = plt.subplot2grid((1, 3), (0, 0), rowspan=1, colspan=1)
            ax[0].axis('off')
        cmap_name = 'autumn'
        farms_in_region_cases = self.data.matched_farm
        farms_in_region = np.arange(self.data.n_farms)
        all_uk = self.data.polygons.dissolve()

        times_id = np.array([np.where(self.data.matched_farm == value)[0][0] for value in farms_in_region_cases])
        times_tmp = self.data.report_day[times_id]
        times = times_tmp[times_tmp >= 0]
        farms_in_region_cases_times = farms_in_region_cases[times_tmp >= 0]
        times_id = times_id[times_tmp >= 0]
        scat_farms = ax_left.scatter(1000 * self.data.location_x[farms_in_region],
                                 1000 * self.data.location_y[farms_in_region], s=0.1, c='#808080')
        if by_region:
            self.data.polygons.plot(ax=ax_left, lw=1, edgecolor='k', facecolor='none')
        else:
            all_uk.plot(ax=ax_left, lw=1, edgecolor='k', facecolor='none')
        scat_cases = ax_left.scatter(1000 * self.data.location_x[farms_in_region_cases_times],
                                 1000 * self.data.location_y[farms_in_region_cases_times], s=12, c=times,
                                 edgecolor='black', linewidths=0.5, cmap=cmap_name)
        ax_left.set_xticks([])
        ax_left.set_yticks([])

        first_day = self.data.date_start.replace(day=1)
        date_range = pd.date_range(start=first_day, end=self.data.date_end, freq='MS')
        dates = date_range.strftime('%b %Y')
        days_in_each_month = [calendar.monthrange(date.year, date.month)[1] for date in date_range]
        if new:
            day_breaks = np.arange(-5,367,7)
            ylab = 'Weekly number of notified premises'
        else:
            day_breaks = np.insert(np.cumsum(days_in_each_month), 0, 0)
            ylab = 'Monthly notified premises'
        if by_region:
            data_hist = np.histogram(self.data.report_day, bins=day_breaks)[0]
            bar_color = np.clip((day_breaks[:-1] + day_breaks[1:]) / 2, 0, np.max(times)) / np.max(times)
            cmap = cm.get_cmap(cmap_name, (np.max(times) * 2 + 1))
            colors = cmap(np.arange(np.max(times) * 2 + 1))
            for j in range(data_hist.shape[0]):
                ax[0,2].bar((day_breaks[j] + day_breaks[j + 1]) / 2, data_hist[j], np.diff(day_breaks)[j],
                             color=colors[
                                 (np.clip((day_breaks[:-1] + day_breaks[1:]) / 2, 0, np.max(times))[j] * 2).astype(
                                     int)],
                             edgecolor='black', label='Data')
            ax[0,2].set_xticks((day_breaks[:-1:11] + day_breaks[1::11]) / 2)
            ax[0,2].set_xticklabels(dates[::11])
            ax[0,2].set_xlim([day_breaks[0] - 5, day_breaks[-1] + 5])
            ax[0,2].set_title('Great Britain (all regions)')
            for i in range(11):
                data_hist = np.histogram(self.data.report_day[self.data.region[self.data.matched_farm] == i], bins=day_breaks)[0]
                bar_color = np.clip((day_breaks[:-1] + day_breaks[1:]) / 2, 0, np.max(times)) / np.max(times)
                cmap = cm.get_cmap(cmap_name, (np.max(times) * 2 + 1))
                colors = cmap(np.arange(np.max(times) * 2 + 1))
                for j in range(data_hist.shape[0]):
                    ax[(i + 1) // 4, (i + 1) % 4 + 2].bar((day_breaks[j] + day_breaks[j + 1]) / 2, data_hist[j], np.diff(day_breaks)[j],
                                 color=colors[
                                     (np.clip((day_breaks[:-1] + day_breaks[1:]) / 2, 0, np.max(times))[j] * 2).astype(
                                         int)],
                                 edgecolor='black', label='Data')
                ax[(i + 1) // 4, (i + 1) % 4 + 2].set_xticks((day_breaks[:-1:11] + day_breaks[1::11]) / 2)
                ax[(i + 1) // 4, (i + 1) % 4 + 2].set_xticklabels(dates[::11])
                ax[(i + 1) // 4, (i + 1) % 4 + 2].set_xlim([day_breaks[0] - 5, day_breaks[-1] + 5])
                ax[(i + 1) // 4, (i + 1) % 4 + 2].set_xlim([day_breaks[0] - 5, day_breaks[-1] + 5])
                ax[(i + 1) // 4, (i + 1) % 4 + 2].set_ylim(bottom=0)
                if np.sum(data_hist) == 0:
                    ax[(i + 1) // 4, (i + 1) % 4 + 2].set_ylim([0, 1])
                ax[(i + 1) // 4, (i + 1) % 4 + 2].set_title(self.data.region_names[i])
                ax[(i + 1) // 4, (i + 1) % 4 + 2].yaxis.set_major_locator(MaxNLocator(integer=True))
        else:
            data_hist = np.histogram(self.data.report_day, bins=day_breaks)[0]

            bar_color = np.clip((day_breaks[:-1] + day_breaks[1:]) / 2, 0, np.max(times)) / np.max(times)
            cmap = cm.get_cmap(cmap_name, (np.max(times) * 2 + 1))
            colors = cmap(np.arange(np.max(times) * 2 + 1))
            ax_right = plt.subplot2grid((1, 3), (0, 1), rowspan=1, colspan=2)
            for j in range(data_hist.shape[0]):
                ax_right.bar((day_breaks[j] + day_breaks[j + 1]) / 2, data_hist[j], np.diff(day_breaks)[j], color=colors[
                    (np.clip((day_breaks[:-1] + day_breaks[1:]) / 2, 0, np.max(times))[j] * 2).astype(int)],
                        edgecolor='black', label='Data')
            if new:
                ax_right.set_xticks(day_breaks[1:-1:2] + 3.5)
                ax_right.set_xticklabels(np.append(np.arange(40,54,2),np.arange(2,40,2)))
                ax_right.set_ylabel(ylab)
                ax_right.set_xlim([day_breaks[0] - 5, day_breaks[-1] + 5])
                ax_right.set_xlabel('Week number')
                ax_top = ax_right.secondary_xaxis('top')
                ax_top.set_xticks(np.insert(np.cumsum(days_in_each_month),0,0)[::2])
                ax_top.set_xticklabels(np.append(dates[::2], 'Oct 2023'))
            else:
                ax_right.set_xticks((day_breaks[:-1:2] + day_breaks[1::2]) / 2)
                ax_right.set_xticklabels(dates[::2])
                ax_right.set_ylabel(ylab)
                ax_right.set_xlim([day_breaks[0] - 5, day_breaks[-1] + 5])
            ax[1].axis('off')
            ax[2].axis('off')
        plt.tight_layout()
        #
        ax_left_pos = ax_left.get_position()
        ax_left.set_position([ax_left_pos.x0 - ax_left_pos.width*0.25, ax_left_pos.y0 - ax_left_pos.height*0.255, ax_left_pos.width*1.5, ax_left_pos.height*1.45])
        ax_right_pos = ax_right.get_position()
        ax_right.set_position([ax_right_pos.x0, ax_right_pos.y0, ax_right_pos.width, ax_right_pos.height])
        # get axes location od ax1
        ax1_pos = ax_left.get_position()
        ax_left.set_ylim(0, 1.08e6)
        inset_x = ax1_pos.width * 8.6e4 / (ax_left.get_xlim()[1] - ax_left.get_xlim()[0])
        inset_y = ax1_pos.height * 1.3e5 / (ax_left.get_ylim()[1] - ax_left.get_ylim()[0])
        inset_ax = fig.add_axes(
            [ax1_pos.x0 + ax1_pos.width - inset_x, ax1_pos.y0 + ax1_pos.height - inset_y - 0.094, inset_x, inset_y])
        all_uk2 = self.data.polygons.dissolve()
        all_uk2.plot(ax=inset_ax, lw=1, edgecolor='k', facecolor='none')
        inset_ax.set_xlim([3.9e5, 4.76e5])
        inset_ax.set_ylim([1.1e6, 1.23e6])
        inset_ax.set_xticks([])
        inset_ax.set_yticks([])

        scat_farms = inset_ax.scatter(1000 * self.data.location_x[farms_in_region],
                                      1000 * self.data.location_y[farms_in_region], s=0.1, c='k')
        scat_cases = inset_ax.scatter(1000 * self.data.location_x[farms_in_region_cases_times],
                                      1000 * self.data.location_y[farms_in_region_cases_times], s=20, c=times,
                                      edgecolor='black', linewidths=0.8, cmap=cmap_name)
        if by_region:
            fig.text(0.03, 0.965, 'A', ha='center', va='center', fontweight='bold', fontsize=20)
            fig.text(0.31, 0.5, 'Number of premises with infection', ha='center', va='center', fontsize=12, rotation=90)
        else:
            fig.text(0.025, 0.965, 'A', ha='center', va='center', fontweight='bold', fontsize=20)
        fig.text(0.302, 0.965, 'B', ha='center', va='center', fontweight='bold', fontsize=20)

    def plot_figure_2(self, c_i=95, birds=False, print_fig=None, best=10, rain_cloud=True):
        day_breaks = np.arange(-5, 367, 7)
        first_day = self.data.date_start.replace(day=1)
        date_range = pd.date_range(start=first_day, end=self.data.date_end + pd.DateOffset(days=1), freq='MS')
        dates = date_range.strftime('%b %Y')
        days_in_each_month = [calendar.monthrange(date.year, date.month)[1] for date in date_range]

        if birds:
            data_hist_bird = np.histogram(self.report_day, bins=day_breaks, weights=np.sum(self.data.species_pop[:, self.infected_farms], axis=0))[0]
            model_hist_bird = np.zeros((self.notified_day.shape[0], len(day_breaks) - 1))
            # Loop over rows (axis=0) and compute histograms efficiently
            for i in range(self.notified_day.shape[0]):
                valid_indices = self.notified_day[i, :] > -1e5  # Filter valid indices for the current row
                valid_days = self.notified_day[i, valid_indices]  # Extract valid days
                valid_weights = np.sum(self.data.species_pop[:, valid_indices], axis=0)  # Compute weights for valid indices
                # Compute histogram for the current row
                model_hist_bird[i, :] = np.histogram(valid_days, bins=day_breaks, weights=valid_weights)[0]
        data_hist = np.histogram(self.report_day, bins=day_breaks)[0]
        model_hist = np.apply_along_axis(lambda a: np.histogram(a, bins=day_breaks)[0], 1, self.notified_day)


        if best is not None:
            model_hist_ci = copy.deepcopy(model_hist)
            model_hist = model_hist[np.argsort(np.sum((model_hist-data_hist) ** 2, axis=1))[:best], :]
            if birds:
                model_hist_bird_ci = copy.deepcopy(model_hist_bird)
                model_hist_bird = model_hist_bird[np.argsort(np.sum((model_hist_bird-data_hist_bird) ** 2, axis=1))[:best], :]

        if birds:
            fig, ax = plt.subplots(2, 3, figsize=(18, 10))
            ax1 = ax[0, 0]
            ax2 = ax[0, 1]
            ax3 = ax[0, 2]
            ax4 = ax[1, 0]
            ax5 = ax[1, 1]
            ax6 = ax[1, 2]
        else:
            fig, ax = plt.subplots(2, 2, figsize=(12, 6), gridspec_kw={'height_ratios': [1, 1.3], 'width_ratios': [1.8, 1]}, constrained_layout=True)
            gs = ax[0, 0].get_gridspec()
            ax[0, 1].remove()
            ax[1, 1].remove()
            ax1 = ax[0, 0]
            ax2 = ax[1, 0]
            ax3 = fig.add_subplot(gs[:, 1])

        ax1.fill_between(np.arange(len(data_hist)), np.percentile(model_hist_ci, (100 - c_i) / 2, axis=0),
                             np.percentile(model_hist_ci, 100 - (100 - c_i) / 2, axis=0), alpha=0.2, color='#1f77b4', edgecolor='none', label='Model')
        for row in model_hist[:-1]:
            ax1.plot(np.arange(len(data_hist)), row, color='#1f77b4', alpha=0.3)
        ax1.plot(np.arange(len(data_hist)), model_hist[-1], color='#1f77b4', alpha=0.3)
        ax1.set_xlim([-3/7, len(data_hist) - 4/7])

        ax1.plot(np.arange(len(data_hist)), data_hist, color='red', marker='x', label='Data',
                linestyle='None', zorder=1000)
        ax1.set_xticks(np.arange(1,52,4))
        ax1.legend()
        ax1.set_xticklabels(np.append(np.arange(40, 54, 4), np.arange(4, 40, 4)))
        ax1.set_xlabel('Week number')
        ax1.set_ylabel('Premises reported as infected')
        ax_top_1 = ax1.secondary_xaxis('top')
        ax_top_1.set_xticks(np.insert(np.cumsum(days_in_each_month), 0, 0)[::4]/7)
        ax_top_1.set_xticklabels(dates[::4])
        if birds:
            ax4.fill_between(np.arange(len(data_hist_bird)), np.percentile(model_hist_bird_ci, (100 - c_i) / 2, axis=0),
                                 np.percentile(model_hist_bird_ci, 100 - (100 - c_i) / 2, axis=0), alpha=0.2, color='#1f77b4', edgecolor='none')
            for row in model_hist_bird[:-1]:
                ax4.plot(np.arange(len(data_hist_bird)), row, color='#1f77b4', alpha=0.3)
            ax4.plot(np.arange(len(data_hist_bird)), model_hist_bird[-1], color='#1f77b4', alpha=0.3, label='Model')
            ax4.set_xlim([-3/7, len(data_hist_bird) - 4/7])

            ax4.plot(np.arange(len(data_hist_bird)), data_hist_bird, color='red', marker='x', label='Data',
                    linestyle='None', zorder=1000)
            ax4.set_xticks(np.arange(1,52,4))
            ax4.set_xticklabels(np.append(np.arange(40, 54, 4), np.arange(4, 40, 4)))
            ax4.set_xlabel('Week number')
            ax4.set_ylabel('Number of birds in premises reported as infected')
            ax_top_4 = ax4.secondary_xaxis('top')
            ax_top_4.set_xticks(np.insert(np.cumsum(days_in_each_month), 0, 0)[::4]/7)
            ax_top_4.set_xticklabels(dates[::4])

        self.region_sim = np.zeros((self.notified_day.shape[0], 11))
        self.region_data = np.zeros(11)
        total_cases_sim = np.sum(self.notified_day > -1e5, axis=1)
        total_cases_data = np.sum(self.report_day >= 0)
        for i in range(11):
            self.region_data[i] = np.sum(
                self.report_day[self.data.region[self.infected_farms] == i] >= 0) / total_cases_data
            self.region_sim[:, i] = np.sum(self.notified_day[:, self.data.region == i] > -1e5,
                                           axis=1) / total_cases_sim
        num_categories = 11
        if rain_cloud:
            cap_width = 0.25
            for j in range(11):
                ax2.hlines(y=j + 1, xmin=np.percentile(self.region_sim[:, 10 - j], 2.5),
                          xmax=np.percentile(self.region_sim[:, 10 - j], 97.5), color=0.3*np.array([1, 1, 1]), linewidth=1.5)
                ax2.vlines(x=np.percentile(self.region_sim[:, 10 - j], 2.5), ymin=j + 1 - cap_width, ymax=j + 1 + cap_width,
                          color=0.3*np.array([1, 1, 1]), linewidth=1.5)
                ax2.vlines(x=np.percentile(self.region_sim[:, 10 - j], 97.5), ymin=j + 1 - cap_width, ymax=j + 1 + cap_width,
                          color=0.3*np.array([1, 1, 1]), linewidth=1.5)
                ax2.plot(np.percentile(self.region_sim[:, 10 - j], 50), j + 1, 'o', color=0.3*np.array([1, 1, 1]), markersize=4)

                ax2.scatter(self.region_sim[:, 10 - j], 1 + j - 0.35 * np.random.rand(len(self.region_sim[:, 10 - j])),
                           marker='.', s=0.1, alpha=0.5, color='#1f77b4')
                violin = ax2.violinplot(self.region_sim[:, 10 - j], positions=[j + 1], widths=0.8, showextrema=False,
                                            vert=False)
                for body in violin['bodies']:
                    path = body.get_paths()[0]
                    verts = path.vertices
                    verts[:, 1] = np.maximum(verts[:, 1], j + 1)
                    body.set_facecolor('#1f77b4')
                    body.set_alpha(0.5)
            ax2.add_patch(Rectangle((-10, -10), 1, 1, fill='#1f77b4', edgecolor=None, label='Model', alpha=0.5))
            ax2.set_xlim([0, 1])
            ax2.set_ylim([0.5, 11.5])
        else:
            box = ax2.boxplot(self.region_sim[:, ::-1], vert=False, showfliers=False,
                              whis=[(100 - c_i) / 2, 100 - (100 - c_i) / 2], patch_artist=True, label='Model')
            plt.setp(box['boxes'], color='#1f77b4')
            plt.setp(box['medians'], color='1')
        for i, true_val in enumerate(self.region_data[::-1]):
            ax2.scatter(true_val, i + 1, color="red", label="Data" if i == 0 else "", zorder=5, marker='x')
        ax2.set_xlabel("Proportion of premises reported as infected")
        ax2.set_ylabel("Region of Great Britain")
        ax2.set_yticks(range(1, num_categories + 1), self.data.region_names_ab[::-1])
        ax2.legend()
        if birds:
            self.region_sim_birds = np.zeros((self.notified_day.shape[0], 11))
            self.region_data_birds = np.zeros(11)
            total_cases_sim_birds = np.sum((self.notified_day > -1e5) * np.sum(self.data.species_pop, axis=0), axis=1)
            total_cases_data_birds = np.sum((self.report_day >= 0) * np.sum(self.data.species_pop[:, self.infected_farms], axis=0))
            for i in range(11):
                self.region_data_birds[i] = np.sum((self.report_day[self.data.region[self.infected_farms] == i] >= 0) * np.sum(self.data.species_pop[:, self.infected_farms[self.data.region[self.infected_farms] == i]], axis=0)) / total_cases_data_birds
                for j in range(self.notified_day.shape[0]):
                    valid_indices = self.notified_day[j, self.data.region == i] > -1e5  # Filter valid indices for the current row
                    valid_weights = np.sum(self.data.species_pop[:, self.data.region == i][:, valid_indices], axis=0)  # Compute weights for valid indices
                    self.region_sim_birds[j, i] = np.sum(valid_weights) / total_cases_sim_birds[j]
            num_categories = 11
            box = ax5.boxplot(self.region_sim_birds[:, ::-1], vert=False, showfliers=False,
                              whis=[(100 - c_i) / 2, 100 - (100 - c_i) / 2], patch_artist=True)
            plt.setp(box['boxes'], color='#1f77b4')
            plt.setp(box['medians'], color='1')
            for i, true_val in enumerate(self.region_data_birds[::-1]):
                ax5.scatter(true_val, i + 1, color="red", label="Data" if i == 0 else "", zorder=5, marker='x')
            ax5.set_xlabel("Proportion of infected premises across regions")
            ax5.set_yticks(range(1, num_categories + 1), self.data.region_names[::-1])

        inf_farm_grid = np.zeros((self.notified_day.shape[0], self.data.n_grids))
        for i in range(self.data.n_grids):
            inf_farm_grid[:, i] = np.sum(self.notified_day[:, self.data.farm_grid == i] > -1e5, axis=1)
        cmap = cm.get_cmap("inferno")
        norm = colors.LogNorm(vmin=1e-3, vmax=10)
        uk = self.data.polygons.dissolve()
        uk.plot(ax=ax3, edgecolor=None, facecolor='#000004')
        uk = uk.geometry.iloc[0]
        paths = [Path(np.array(polygon.exterior.coords)) for polygon in uk.geoms]  # Handle multiple polygons
        uk_path = Path.make_compound_path(*paths)  # Combine into one Path
        uk_patch = PathPatch(uk_path, transform=ax3.transData, facecolor='none', edgecolor='none')
        for i in range(self.data.n_grids):
            if 1000*self.data.grid_location_y[i] < 1.08e6:
                color = cmap(norm(np.mean(inf_farm_grid, axis=0)[i]))
                rect = plt.Rectangle((1000*self.data.grid_location_x[i], 1000*self.data.grid_location_y[i]), 1000*self.data.grid_size[i],
                                           1000*self.data.grid_size[i], facecolor=color, edgecolor=None)
                rect.set_clip_path(uk_patch)
                ax3.add_patch(rect)
        ax3.scatter(1000*self.data.location_x[self.data.matched_farm], 1000*self.data.location_y[self.data.matched_farm], color='cyan', edgecolors='black', linewidths=0.15, s=3, zorder=3, label='Data')#, color='#00ffff', s=0.1, zorder=3)
        ax3.set_ylim([0, 1.08e6])
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array(np.mean(inf_farm_grid, axis=0))
        plt.colorbar(sm, ax=ax3, orientation="vertical", label="Mean number of premises reported as infected per grid cell")
        ax3.set_xticks([])
        ax3.set_yticks([])
        ax3_pos = ax3.get_position()
        inset_x = ax3_pos.width * 8.6e4 / (ax3.get_xlim()[1] - ax3.get_xlim()[0])
        inset_y = ax3_pos.height * 1.3e5 / (ax3.get_ylim()[1] - ax3.get_ylim()[0])
        inset_ax = fig.add_axes(
            [ax3_pos.x0 + ax3_pos.width +0.026, ax3_pos.y0 + ax3_pos.height + 0.0056, inset_x, inset_y])
        all_uk2 = self.data.polygons.dissolve()
        all_uk2.plot(ax=inset_ax, edgecolor=None, facecolor='#000004')
        uk = all_uk2.geometry.iloc[0]
        paths = [Path(np.array(polygon.exterior.coords)) for polygon in uk.geoms]
        uk_path = Path.make_compound_path(*paths)
        uk_patch = PathPatch(uk_path, transform=ax3.transData, facecolor='none', edgecolor='none')
        for i in range(self.data.n_grids):
            if 1000*self.data.grid_location_y[i] > 1.08e6:
                color = cmap(norm(np.mean(inf_farm_grid, axis=0)[i]))
                rect = plt.Rectangle((1000*self.data.grid_location_x[i], 1000*self.data.grid_location_y[i]), 1000*self.data.grid_size[i],
                                           1000*self.data.grid_size[i], facecolor=color, edgecolor=None)
                rect.set_clip_path(uk_patch)
                inset_ax.add_patch(rect)
        inset_ax.set_xlim([3.9e5, 4.76e5])
        inset_ax.set_ylim([1.1e6, 1.23e6])
        inset_ax.set_xticks([])
        inset_ax.set_yticks([])
        scat_cases = inset_ax.scatter(1000 * self.data.location_x[self.data.matched_farm],
                                      1000 * self.data.location_y[self.data.matched_farm], color='cyan', edgecolors='black', linewidths=0.15, s=3, zorder=3, label='Data')
        ax3.legend(loc="upper right")
        if birds:
            inf_farm_grid_bird = np.zeros((self.notified_day.shape[0], self.data.n_grids))
            for i in range(self.data.n_grids):
                inf_farm_grid_bird[:, i] = np.sum((self.notified_day[:, self.data.farm_grid == i] > -1e5) * np.sum(self.data.species_pop[:, self.data.farm_grid == i], axis=0), axis=1)
            cmap = cm.get_cmap("inferno")
            norm = colors.LogNorm(vmin=1e-2, vmax=1e6)
            uk = self.data.polygons.dissolve()
            uk.plot(ax=ax6, edgecolor=None, facecolor='#000004')
            uk = uk.geometry.iloc[0]
            paths = [Path(np.array(polygon.exterior.coords)) for polygon in uk.geoms]  # Handle multiple polygons
            uk_path = Path.make_compound_path(*paths)  # Combine into one Path
            uk_patch = PathPatch(uk_path, transform=ax6.transData, facecolor='none', edgecolor='none')
            for i in range(self.data.n_grids):
                if 1000 * self.data.grid_location_y[i] < 1.08e6:
                    color = cmap(norm(np.mean(inf_farm_grid_bird, axis=0)[i]))
                    rect = plt.Rectangle((1000 * self.data.grid_location_x[i], 1000 * self.data.grid_location_y[i]),
                                         1000 * self.data.grid_size[i],
                                         1000 * self.data.grid_size[i], facecolor=color, edgecolor=None)
                    rect.set_clip_path(uk_patch)
                    ax6.add_patch(rect)
            ax6.scatter(1000 * self.data.location_x[self.data.matched_farm],
                        1000 * self.data.location_y[self.data.matched_farm], color='cyan', edgecolors='black',
                        linewidths=0.1, s=1.5, zorder=3)
            ax6.set_ylim([0, 1.08e6])
            sm = cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array(np.mean(inf_farm_grid_bird, axis=0))
            plt.colorbar(sm, ax=ax6, orientation="vertical", label="Mean number of birds on infected premises per grid cell")
            ax6.set_xticks([])
            ax6.set_yticks([])
            ax6_pos = ax6.get_position()
            inset_x = ax6_pos.width * 8.6e4 / (ax6.get_xlim()[1] - ax6.get_xlim()[0])
            inset_y = ax6_pos.height * 1.3e5 / (ax6.get_ylim()[1] - ax6.get_ylim()[0])
            inset_ax = fig.add_axes(
                [ax6_pos.x0 + ax6_pos.width - inset_x - 0.005, ax6_pos.y0 + ax6_pos.height - inset_y - 0.005, inset_x,
                 inset_y])
            all_uk2 = self.data.polygons.dissolve()
            all_uk2.plot(ax=inset_ax, lw=1, edgecolor='k', facecolor='none')
            inset_ax.set_xlim([3.9e5, 4.76e5])
            inset_ax.set_ylim([1.1e6, 1.23e6])
            inset_ax.set_xticks([])
            inset_ax.set_yticks([])
            scat_cases = inset_ax.scatter(1000 * self.data.location_x[self.data.matched_farm],
                                          1000 * self.data.location_y[self.data.matched_farm], color='cyan',
                                          edgecolors='black', linewidths=0.1, s=1.5, zorder=3)
        fig.text(0.0095, 0.975, 'A', ha='center', va='center', fontweight='bold', fontsize=20)
        fig.text(0.0095, 0.54, 'B', ha='center', va='center', fontweight='bold', fontsize=20)
        fig.text(0.618, 0.975, 'C', ha='center', va='center', fontweight='bold', fontsize=20)

    def plot_figure_3(self, control_all=None, base_all=None):
        zone_labels = ['Baseline', '5 km', '10 km', '15 km', 'County', 'Region', '5 km', '10 km', '15 km', 'County',
                       'Region', '5 km', '10 km', '15 km', 'County', 'Region', '5 km', '10 km', '15 km', 'County',
                       'Region']
        level_labels = ['Susceptibility factor = 0.8', 'Susceptibility factor = 0.6', 'Susceptibility factor = 0.4',
                        'Susceptibility factor = 0.2']
        colors = ['#A62C00', '#003811', '#0F0A4D', '#810024', '#004000', '#804500',
                  '#006B44', '#423D80', '#B40057', '#337300', '#B37800',
                  '#1b9e77', '#7570b3', '#e7298a', '#66a61e', '#e6ab02',
                  '#4ED1AA', '#A8A3E6', '#FF5CBD', '#99D951', '#FFDE35',
                  '#81FFDD', '#DBD6FF', '#FF8FF0', '#CCFF84', '#FFFF68',
                  '#B4FFFF', '#F5F0FF', '#FFC2FF', '#FFFFB7' '#FFFF9B']

        fig, ax = plt.subplots(figsize=(13, 5))
        ax.vlines(x=1.5, ymin=np.percentile(base_all, 2.5),
                  ymax=np.percentile(base_all, 97.5), color=[0.3, 0.3, 0.3], linewidth=1.5)
        cap_width = 0.12
        ax.hlines(y=np.percentile(base_all, 2.5), xmin=1.5 - cap_width, xmax=1.5 + cap_width,
                  color=[0.3, 0.3, 0.3], linewidth=1.5)
        ax.hlines(y=np.percentile(base_all, 97.5), xmin=1.5 - cap_width, xmax=1.5 + cap_width,
                  color=[0.3, 0.3, 0.3], linewidth=1.5)
        ax.plot(1.5, np.percentile(base_all, 50), 'o', color=[0.3, 0.3, 0.3])
        ax.scatter(1.5 + 0.4 * np.random.rand(len(base_all)), base_all, color=colors[0],
                   marker='.', s=0.1, alpha=0.5)
        violin = ax.violinplot(base_all, positions=[1.5], widths=0.8, showextrema=False)
        for body in violin['bodies']:
            path = body.get_paths()[0]
            verts = path.vertices
            verts[:, 0] = np.minimum(verts[:, 0], 1.5)
            body.set_facecolor(colors[0])

        group_spacing = 0.5
        for i in range(4):
            y_positions = (np.arange(5) + (5 + group_spacing) * i + 3)
            for j in range(5):
                ax.vlines(x=y_positions[j], ymin=np.percentile(control_all[:, j + 1, 2, 3 - i], 2.5),
                          ymax=np.percentile(control_all[:, j + 1, 2, 3 - i], 97.5),
                          color=[0.3, 0.3, 0.3], linewidth=1.5)
                ax.hlines(y=np.percentile(control_all[:, j + 1, 2, 3 - i], 2.5),
                          xmin=y_positions[j] - cap_width, xmax=y_positions[j] + cap_width,
                          color=[0.3, 0.3, 0.3], linewidth=1.5)
                ax.hlines(y=np.percentile(control_all[:, j + 1, 2, 3 - i], 97.5),
                          xmin=y_positions[j] - cap_width, xmax=y_positions[j] + cap_width,
                          color=[0.3, 0.3, 0.3], linewidth=1.5)
                ax.plot(y_positions[j], np.percentile(control_all[:, j + 1, 2, 3 - i], 50), 'o',
                        color=[0.3, 0.3, 0.3])
                ax.scatter(
                    y_positions[j] + 0.4 * np.random.rand(len(control_all[:, j + 1, 2, 3 - i])),
                    control_all[:, j + 1, 2, 3 - i], color=colors[5 * i + j + 1], marker='.', s=0.1,
                    alpha=0.5)
            violin = ax.violinplot(control_all[:, 1:, 2, 3 - i], positions=y_positions, widths=0.8,
                                   showextrema=False)
            for j, body in enumerate(violin['bodies']):
                path = body.get_paths()[0]
                verts = path.vertices
                verts[:, 0] = np.minimum(verts[:, 0], y_positions[j])
                body.set_facecolor(colors[5 * i + j + 1])

            ax.text((2 + (5 + group_spacing) * i + 3), -150, level_labels[i], ha="center",
                    transform=ax.transData)
            ax.set_ylim([0, 1260])

        ax.set_ylabel("Number of premises reported as infected")
        ax.set_xticks(
            np.append(1.5, np.concatenate([(np.arange(5) + (5 + group_spacing) * i + 3) for i in range(4)])))
        ax.set_xticklabels(zone_labels)
        ax.set_xlim([1, 24])
        plt.tight_layout()

    def plot_figure_4(self, base_50=None, control_50=None, control_975=None, control_025=None):
        zone = ['5 km', '10 km', '15 km', 'County', 'Region']

        data = base_50 - control_50
        vmin, vmax = data.min(), data.max()
        if vmin < 30:
            vmin = 0
        vmax = 250
        fig, ax = plt.subplots(2, 5, figsize=(12, 5), constrained_layout=True)
        for i in range(1, 6):
            im = ax[0, i - 1].imshow(data[i], cmap='viridis', vmin=vmin, vmax=vmax)
            for x in range(data.shape[1]):
                for y in range(data.shape[2]):
                    ax[0, i - 1].text(y, x, f"{data[i, x, y]:.0f}",
                                      ha='center', va='center', color='w', fontsize=10)
            ax[0, i - 1].set_title('Enhanced area: ' + str(zone[i - 1]))
            if i == 0:
                ax[0, i - 1].set_yticks(np.arange(0, 4, 1))
                ax[0, i - 1].set_yticklabels(np.arange(7, 35, 7))
            else:
                ax[0, i - 1].set_yticks([])
            ax[0, i - 1].set_xticks([])
            ax[0, i - 1].invert_yaxis()
        fig.supxlabel('Susceptibility factor due to enhanced biosecurity')
        fig.supylabel('Duration of enhanced biosecurity (days)')
        cbar1 = fig.colorbar(im, ax=ax[0, :], location='right', shrink=1, fraction=0.08, pad=0.02)
        cbar1.set_label("Reduction in infected premises")
        data = control_975 - control_025
        vmin, vmax = data.min(), data.max()
        if vmin < 100:
            vmin = 0
        vmax = 500
        for i in range(1, 6):
            im = ax[1, i - 1].imshow(data[i], cmap='plasma', vmin=vmin, vmax=vmax)
            for x in range(data.shape[1]):
                for y in range(data.shape[2]):
                    ax[1, i - 1].text(y, x, f"{data[i, x, y]:.0f}",
                                      ha='center', va='center', color='w', fontsize=10)
            ax[1, i - 1].set_xticks(np.arange(0, 4, 1))
            ax[1, i - 1].set_xticklabels([0.2, 0.4, 0.6, 0.8])
            if i == 0:
                ax[1, i - 1].set_yticks(np.arange(0, 4, 1))
                ax[1, i - 1].set_yticklabels(np.arange(7, 35, 7))
            else:
                ax[1, i - 1].set_yticks([])
            ax[1, i - 1].invert_yaxis()
        fig.supxlabel('Susceptibility factor due to enhanced biosecurity')
        fig.supylabel('Duration of enhanced biosecurity (days)')
        cbar2 = fig.colorbar(im, ax=ax[1, :], location='right', shrink=1, fraction=0.08, pad=0.02)
        cbar2.set_label("Range of 95% interval")
