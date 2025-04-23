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
        # Initialise variables
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

        # Limit to data to a single region
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
                # Get minimum squared distance between grid cells by distance between centroids subtract distance to edge of grids
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
                # Get minimum squared distance between grid cells by distance between centroids subtract distance to edge of grids
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
