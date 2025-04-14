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


def plot_figure_1(model0, by_region=False, new=True):
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
    farms_in_region_cases = model0.data.matched_farm
    farms_in_region = np.arange(model0.data.n_farms)
    all_uk = model0.data.polygons.dissolve()

    times_id = np.array([np.where(model0.data.matched_farm == value)[0][0] for value in farms_in_region_cases])
    times_tmp = model0.data.report_day[times_id]
    times = times_tmp[times_tmp >= 0]
    farms_in_region_cases_times = farms_in_region_cases[times_tmp >= 0]
    times_id = times_id[times_tmp >= 0]
    scat_farms = ax_left.scatter(1000 * model0.data.location_x[farms_in_region],
                             1000 * model0.data.location_y[farms_in_region], s=0.1, c='#808080')
    if by_region:
        model0.data.polygons.plot(ax=ax_left, lw=1, edgecolor='k', facecolor='none')
    else:
        all_uk.plot(ax=ax_left, lw=1, edgecolor='k', facecolor='none')
    scat_cases = ax_left.scatter(1000 * model0.data.location_x[farms_in_region_cases_times],
                             1000 * model0.data.location_y[farms_in_region_cases_times], s=12, c=times,
                             edgecolor='black', linewidths=0.5, cmap=cmap_name)
    ax_left.set_xticks([])
    ax_left.set_yticks([])

    first_day = model0.data.date_start.replace(day=1)
    date_range = pd.date_range(start=first_day, end=model0.data.date_end, freq='MS')
    dates = date_range.strftime('%b %Y')
    days_in_each_month = [calendar.monthrange(date.year, date.month)[1] for date in date_range]
    if new:
        day_breaks = np.arange(-5,367,7)
        ylab = 'Weekly number of notified premises'
    else:
        day_breaks = np.insert(np.cumsum(days_in_each_month), 0, 0)
        ylab = 'Monthly notified premises'
    if by_region:
        data_hist = np.histogram(model0.data.report_day, bins=day_breaks)[0]
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
            data_hist = np.histogram(model0.data.report_day[model0.data.region[model0.data.matched_farm] == i], bins=day_breaks)[0]
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
            ax[(i + 1) // 4, (i + 1) % 4 + 2].set_title(model0.data.region_names[i])
            ax[(i + 1) // 4, (i + 1) % 4 + 2].yaxis.set_major_locator(MaxNLocator(integer=True))
    else:
        data_hist = np.histogram(model0.data.report_day, bins=day_breaks)[0]

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
    all_uk2 = model0.data.polygons.dissolve()
    all_uk2.plot(ax=inset_ax, lw=1, edgecolor='k', facecolor='none')
    inset_ax.set_xlim([3.9e5, 4.76e5])
    inset_ax.set_ylim([1.1e6, 1.23e6])
    inset_ax.set_xticks([])
    inset_ax.set_yticks([])

    scat_farms = inset_ax.scatter(1000 * model0.data.location_x[farms_in_region],
                                  1000 * model0.data.location_y[farms_in_region], s=0.1, c='k')
    scat_cases = inset_ax.scatter(1000 * model0.data.location_x[farms_in_region_cases_times],
                                  1000 * model0.data.location_y[farms_in_region_cases_times], s=20, c=times,
                                  edgecolor='black', linewidths=0.8, cmap=cmap_name)
    if by_region:
        fig.text(0.03, 0.965, 'A', ha='center', va='center', fontweight='bold', fontsize=20)
        fig.text(0.31, 0.5, 'Number of premises with infection', ha='center', va='center', fontsize=12, rotation=90)
    else:
        fig.text(0.025, 0.965, 'A', ha='center', va='center', fontweight='bold', fontsize=20)
    fig.text(0.302, 0.965, 'B', ha='center', va='center', fontweight='bold', fontsize=20)

def plot_figure_2(model0, c_i=95, birds=False, print_fig=None, best=10, rain_cloud=True):
    day_breaks = np.arange(-5, 367, 7)
    first_day = model0.data.date_start.replace(day=1)
    date_range = pd.date_range(start=first_day, end=model0.data.date_end + pd.DateOffset(days=1), freq='MS')
    dates = date_range.strftime('%b %Y')
    days_in_each_month = [calendar.monthrange(date.year, date.month)[1] for date in date_range]

    if birds:
        data_hist_bird = np.histogram(model0.report_day, bins=day_breaks, weights=np.sum(model0.data.species_pop[:, model0.infected_farms], axis=0))[0]
        model_hist_bird = np.zeros((model0.notified_day.shape[0], len(day_breaks) - 1))
        # Loop over rows (axis=0) and compute histograms efficiently
        for i in range(model0.notified_day.shape[0]):
            valid_indices = model0.notified_day[i, :] > -1e5  # Filter valid indices for the current row
            valid_days = model0.notified_day[i, valid_indices]  # Extract valid days
            valid_weights = np.sum(model0.data.species_pop[:, valid_indices], axis=0)  # Compute weights for valid indices
            # Compute histogram for the current row
            model_hist_bird[i, :] = np.histogram(valid_days, bins=day_breaks, weights=valid_weights)[0]
    data_hist = np.histogram(model0.report_day, bins=day_breaks)[0]
    model_hist = np.apply_along_axis(lambda a: np.histogram(a, bins=day_breaks)[0], 1, model0.notified_day)


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

    model0.region_sim = np.zeros((model0.notified_day.shape[0], 11))
    model0.region_data = np.zeros(11)
    total_cases_sim = np.sum(model0.notified_day > -1e5, axis=1)
    total_cases_data = np.sum(model0.report_day >= 0)
    for i in range(11):
        model0.region_data[i] = np.sum(
            model0.report_day[model0.data.region[model0.infected_farms] == i] >= 0) / total_cases_data
        model0.region_sim[:, i] = np.sum(model0.notified_day[:, model0.data.region == i] > -1e5,
                                       axis=1) / total_cases_sim
    num_categories = 11
    if rain_cloud:
        cap_width = 0.25
        for j in range(11):
            ax2.hlines(y=j + 1, xmin=np.percentile(model0.region_sim[:, 10 - j], 2.5),
                      xmax=np.percentile(model0.region_sim[:, 10 - j], 97.5), color=0.3*np.array([1, 1, 1]), linewidth=1.5)
            ax2.vlines(x=np.percentile(model0.region_sim[:, 10 - j], 2.5), ymin=j + 1 - cap_width, ymax=j + 1 + cap_width,
                      color=0.3*np.array([1, 1, 1]), linewidth=1.5)
            ax2.vlines(x=np.percentile(model0.region_sim[:, 10 - j], 97.5), ymin=j + 1 - cap_width, ymax=j + 1 + cap_width,
                      color=0.3*np.array([1, 1, 1]), linewidth=1.5)
            ax2.plot(np.percentile(model0.region_sim[:, 10 - j], 50), j + 1, 'o', color=0.3*np.array([1, 1, 1]), markersize=4)

            ax2.scatter(model0.region_sim[:, 10 - j], 1 + j - 0.35 * np.random.rand(len(model0.region_sim[:, 10 - j])),
                       marker='.', s=0.1, alpha=0.5, color='#1f77b4')
            violin = ax2.violinplot(model0.region_sim[:, 10 - j], positions=[j + 1], widths=0.8, showextrema=False,
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
        box = ax2.boxplot(model0.region_sim[:, ::-1], vert=False, showfliers=False,
                          whis=[(100 - c_i) / 2, 100 - (100 - c_i) / 2], patch_artist=True, label='Model')
        plt.setp(box['boxes'], color='#1f77b4')
        plt.setp(box['medians'], color='1')
    for i, true_val in enumerate(model0.region_data[::-1]):
        ax2.scatter(true_val, i + 1, color="red", label="Data" if i == 0 else "", zorder=5, marker='x')
    ax2.set_xlabel("Proportion of premises reported as infected")
    ax2.set_ylabel("Region of Great Britain")
    ax2.set_yticks(range(1, num_categories + 1), model0.data.region_names_ab[::-1])
    ax2.legend()
    if birds:
        model0.region_sim_birds = np.zeros((model0.notified_day.shape[0], 11))
        model0.region_data_birds = np.zeros(11)
        total_cases_sim_birds = np.sum((model0.notified_day > -1e5) * np.sum(model0.data.species_pop, axis=0), axis=1)
        total_cases_data_birds = np.sum((model0.report_day >= 0) * np.sum(model0.data.species_pop[:, model0.infected_farms], axis=0))
        for i in range(11):
            model0.region_data_birds[i] = np.sum((model0.report_day[model0.data.region[model0.infected_farms] == i] >= 0) * np.sum(model0.data.species_pop[:, model0.infected_farms[model0.data.region[model0.infected_farms] == i]], axis=0)) / total_cases_data_birds
            for j in range(model0.notified_day.shape[0]):
                valid_indices = model0.notified_day[j, model0.data.region == i] > -1e5  # Filter valid indices for the current row
                valid_weights = np.sum(model0.data.species_pop[:, model0.data.region == i][:, valid_indices], axis=0)  # Compute weights for valid indices
                model0.region_sim_birds[j, i] = np.sum(valid_weights) / total_cases_sim_birds[j]
        num_categories = 11
        box = ax5.boxplot(model0.region_sim_birds[:, ::-1], vert=False, showfliers=False,
                          whis=[(100 - c_i) / 2, 100 - (100 - c_i) / 2], patch_artist=True)
        plt.setp(box['boxes'], color='#1f77b4')
        plt.setp(box['medians'], color='1')
        for i, true_val in enumerate(model0.region_data_birds[::-1]):
            ax5.scatter(true_val, i + 1, color="red", label="Data" if i == 0 else "", zorder=5, marker='x')
        ax5.set_xlabel("Proportion of infected premises across regions")
        ax5.set_yticks(range(1, num_categories + 1), model0.data.region_names[::-1])

    inf_farm_grid = np.zeros((model0.notified_day.shape[0], model0.data.n_grids))
    for i in range(model0.data.n_grids):
        inf_farm_grid[:, i] = np.sum(model0.notified_day[:, model0.data.farm_grid == i] > -1e5, axis=1)
    cmap = cm.get_cmap("inferno")
    norm = colors.LogNorm(vmin=1e-3, vmax=10)
    uk = model0.data.polygons.dissolve()
    uk.plot(ax=ax3, edgecolor=None, facecolor='#000004')
    uk = uk.geometry.iloc[0]
    paths = [Path(np.array(polygon.exterior.coords)) for polygon in uk.geoms]  # Handle multiple polygons
    uk_path = Path.make_compound_path(*paths)  # Combine into one Path
    uk_patch = PathPatch(uk_path, transform=ax3.transData, facecolor='none', edgecolor='none')
    for i in range(model0.data.n_grids):
        if 1000*model0.data.grid_location_y[i] < 1.08e6:
            color = cmap(norm(np.mean(inf_farm_grid, axis=0)[i]))
            rect = plt.Rectangle((1000*model0.data.grid_location_x[i], 1000*model0.data.grid_location_y[i]), 1000*model0.data.grid_size[i],
                                       1000*model0.data.grid_size[i], facecolor=color, edgecolor=None)
            rect.set_clip_path(uk_patch)
            ax3.add_patch(rect)
    ax3.scatter(1000*model0.data.location_x[model0.data.matched_farm], 1000*model0.data.location_y[model0.data.matched_farm], color='cyan', edgecolors='black', linewidths=0.15, s=3, zorder=3, label='Data')#, color='#00ffff', s=0.1, zorder=3)
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
    all_uk2 = model0.data.polygons.dissolve()
    all_uk2.plot(ax=inset_ax, edgecolor=None, facecolor='#000004')
    uk = all_uk2.geometry.iloc[0]
    paths = [Path(np.array(polygon.exterior.coords)) for polygon in uk.geoms]
    uk_path = Path.make_compound_path(*paths)
    uk_patch = PathPatch(uk_path, transform=ax3.transData, facecolor='none', edgecolor='none')
    for i in range(model0.data.n_grids):
        if 1000*model0.data.grid_location_y[i] > 1.08e6:
            color = cmap(norm(np.mean(inf_farm_grid, axis=0)[i]))
            rect = plt.Rectangle((1000*model0.data.grid_location_x[i], 1000*model0.data.grid_location_y[i]), 1000*model0.data.grid_size[i],
                                       1000*model0.data.grid_size[i], facecolor=color, edgecolor=None)
            rect.set_clip_path(uk_patch)
            inset_ax.add_patch(rect)
    inset_ax.set_xlim([3.9e5, 4.76e5])
    inset_ax.set_ylim([1.1e6, 1.23e6])
    inset_ax.set_xticks([])
    inset_ax.set_yticks([])
    scat_cases = inset_ax.scatter(1000 * model0.data.location_x[model0.data.matched_farm],
                                  1000 * model0.data.location_y[model0.data.matched_farm], color='cyan', edgecolors='black', linewidths=0.15, s=3, zorder=3, label='Data')
    ax3.legend(loc="upper right")
    if birds:
        inf_farm_grid_bird = np.zeros((model0.notified_day.shape[0], model0.data.n_grids))
        for i in range(model0.data.n_grids):
            inf_farm_grid_bird[:, i] = np.sum((model0.notified_day[:, model0.data.farm_grid == i] > -1e5) * np.sum(model0.data.species_pop[:, model0.data.farm_grid == i], axis=0), axis=1)
        cmap = cm.get_cmap("inferno")
        norm = colors.LogNorm(vmin=1e-2, vmax=1e6)
        uk = model0.data.polygons.dissolve()
        uk.plot(ax=ax6, edgecolor=None, facecolor='#000004')
        uk = uk.geometry.iloc[0]
        paths = [Path(np.array(polygon.exterior.coords)) for polygon in uk.geoms]  # Handle multiple polygons
        uk_path = Path.make_compound_path(*paths)  # Combine into one Path
        uk_patch = PathPatch(uk_path, transform=ax6.transData, facecolor='none', edgecolor='none')
        for i in range(model0.data.n_grids):
            if 1000 * model0.data.grid_location_y[i] < 1.08e6:
                color = cmap(norm(np.mean(inf_farm_grid_bird, axis=0)[i]))
                rect = plt.Rectangle((1000 * model0.data.grid_location_x[i], 1000 * model0.data.grid_location_y[i]),
                                     1000 * model0.data.grid_size[i],
                                     1000 * model0.data.grid_size[i], facecolor=color, edgecolor=None)
                rect.set_clip_path(uk_patch)
                ax6.add_patch(rect)
        ax6.scatter(1000 * model0.data.location_x[model0.data.matched_farm],
                    1000 * model0.data.location_y[model0.data.matched_farm], color='cyan', edgecolors='black',
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
        all_uk2 = model0.data.polygons.dissolve()
        all_uk2.plot(ax=inset_ax, lw=1, edgecolor='k', facecolor='none')
        inset_ax.set_xlim([3.9e5, 4.76e5])
        inset_ax.set_ylim([1.1e6, 1.23e6])
        inset_ax.set_xticks([])
        inset_ax.set_yticks([])
        scat_cases = inset_ax.scatter(1000 * model0.data.location_x[model0.data.matched_farm],
                                      1000 * model0.data.location_y[model0.data.matched_farm], color='cyan',
                                      edgecolors='black', linewidths=0.1, s=1.5, zorder=3)
    fig.text(0.0095, 0.975, 'A', ha='center', va='center', fontweight='bold', fontsize=20)
    fig.text(0.0095, 0.54, 'B', ha='center', va='center', fontweight='bold', fontsize=20)
    fig.text(0.618, 0.975, 'C', ha='center', va='center', fontweight='bold', fontsize=20)

def plot_figure_3(model0, control_all=None, base_all=None):
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

def plot_figure_4(model0, base_50=None, control_50=None, control_975=None, control_025=None):
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
