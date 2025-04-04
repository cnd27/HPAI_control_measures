from pyproj import Transformer
import pandas as pd
import numpy as np

np.set_printoptions(suppress=True)
# Load case data
cases = pd.read_excel('../Data/case_data')

# Coordinate transformation
trans = Transformer.from_crs(
            "EPSG:4326",
            "EPSG:27700",
            always_xy=True,
        )
# Get coordinates in BNG
inf_x, inf_y = trans.transform(cases['Long'], cases['Lat'])
inf_x = inf_x / 1000
inf_y = inf_y / 1000

inf_birds = np.array(cases['TotalBirds']) # Total number of birds on infected premises
type_0_birds = np.nansum(np.array([cases['Chicken'], cases['Turkey']]), axis=0) # Number of chickens and turkeys
type_0_birds[type_0_birds > inf_birds] = inf_birds[type_0_birds > inf_birds] # Ensure fewer chickens and turkeys than total birds
type_1_birds = np.nansum(np.array([cases['Duck'], cases['Geese']]), axis=0) # Number of ducks and geese
type_1_birds[type_1_birds > inf_birds] = inf_birds[type_1_birds > inf_birds] # Ensure fewer ducks and geese than total birds
type_2_birds = inf_birds - (type_0_birds + type_1_birds) # Number of other birds
type_birds = np.array([type_0_birds, type_1_birds, type_2_birds])
most_inf_bird_type = np.argmax(type_birds, axis=0) # Most bird type

# Get if there are more other birds than chickens/turkeys or ducks/geese
most_inf_bird_type_any = np.zeros(len(inf_x))
for i in range(len(inf_x)):
    if most_inf_bird_type[i] == 0:
        if type_birds[1, i] + type_birds[2, i] >= type_birds[0, i]:
            most_inf_bird_type_any[i] = 1
    elif most_inf_bird_type[i] == 1:
        if type_birds[0, i] + type_birds[2, i] >= type_birds[1, i]:
            most_inf_bird_type_any[i] = 1
    else:
        most_inf_bird_type_any[i] = 1
both_inf_types = (type_birds[0, :] > 0) & (type_birds[1, :] > 0) # If both chicken/turkey and duck/geese

# Load farm data
farms = np.loadtxt('../Data/premises_data')
farms_old = np.copy(farms)
farms_x = farms[:, 2]
farms_y = farms[:, 3]
farms_birds = np.sum(farms[:, 4:], axis=1)
most_farm_bird_type = np.argmax(farms[:, 4:], axis=1)
both_farm_types = (farms[:, 4] > 0) & (farms[:, 5] > 0)

# Match farms
matched_type = np.zeros(len(inf_x))
matched_idx = 1e6 * np.ones(len(inf_x))
dist_farm = np.zeros(len(inf_x))
for i in range(len(inf_x)):
    # Get sorted farms by distance if < 2km
    dist = ((farms_x - inf_x[i]) ** 2 + (farms_y - inf_y[i]) ** 2) ** (1/2)
    dist_idx = np.argsort(dist)
    dist = dist[dist_idx]
    dists_idx = dist_idx[dist < 2]
    dists = dist[dist < 2]
    # Check if the most common bird type is the same and the farm size is similar
    correct_type = most_inf_bird_type[i] == most_farm_bird_type[dists_idx]
    correct_size = ((farms_birds[dists_idx] <= 100) & (farms_birds[dists_idx] > 0) & (inf_birds[i] <= 100) & (inf_birds[i] > 0)) | \
                     ((farms_birds[dists_idx] <= 1200) & (farms_birds[dists_idx] > 40) & (inf_birds[i] <= 1200) & (inf_birds[i] > 40)) | \
                   ((farms_birds[dists_idx] <= 12000) & (farms_birds[dists_idx] > 800) & (inf_birds[i] <= 12000) & (inf_birds[i] > 800)) | \
                   ((farms_birds[dists_idx] <= 120000) & (farms_birds[dists_idx] > 8000) & (inf_birds[i] <= 120000) & (inf_birds[i] > 8000)) | \
                   ((farms_birds[dists_idx] > 80000) & (inf_birds[i] > 80000))
    # Match closest farm with correct bird type and size
    matched_idx[i] = dists_idx[np.where(correct_type & correct_size)[0][0]] if len(np.where(correct_type & correct_size)[0]) > 0 else 1e6
    dist_farm[i] = dists[np.where(correct_type & correct_size)[0][0]] if len(np.where(correct_type & correct_size)[0]) > 0 else 0
    if matched_idx[i] != 1e6:
        # Note those matched in PHASE 1
        matched_type[i] = 1
    if matched_idx[i] == 1e6:
        # If no match, check if closest farm has the same bird type and is within 200m
        if len(dists) > 0:
            if dists[0] < 0.2:
                matched_idx[i] = dists_idx[0]
                dist_farm[i] = dists[0]
                # Note those matched in PHASE 2
                matched_type[i] = 2
    if matched_idx[i] == 1e6:
        # If no match, check if closest farm has the same bird type or any is largest type and is within 2km
        correct_type_ish = (most_inf_bird_type_any[i]*np.ones(len(correct_size))).astype(bool)
        matched_idx[i] = dists_idx[np.where(correct_type_ish & correct_size)[0][0]] if len(
            np.where(correct_type_ish & correct_size)[0]) > 0 else 1e6
        dist_farm[i] = dists[np.where(correct_type_ish & correct_size)[0][0]] if len(
            np.where(correct_type_ish & correct_size)[0]) > 0 else 0
        if matched_idx[i] != 1e6:
            # Note those matched in PHASE 3
            matched_type[i] = 3
    if matched_idx[i] == 1e6:
        # If both chicken/turkey and duck/geese in both data sets and farm is small and is within 2km
        correct_type_ish = both_inf_types[i] == both_farm_types[dists_idx]
        small_size = (farms_birds[dists_idx] <= 100) & (farms_birds[dists_idx] > 0) & (inf_birds[i] <= 100) & (inf_birds[i] > 0)
        matched_idx[i] = dists_idx[np.where(correct_type_ish & small_size)[0][0]] if len(
            np.where(correct_type_ish & small_size)[0]) > 0 else 1e6
        dist_farm[i] = dists[np.where(correct_type_ish & small_size)[0][0]] if len(
            np.where(correct_type_ish & small_size)[0]) > 0 else 0
        if matched_idx[i] != 1e6:
            # Note those matched in PHASE 4
            matched_type[i] = 4


# Get where there are multiple matches to same farm
matched_idx = matched_idx.astype(int)
multiple_matches = [np.where(np.where(np.bincount(matched_idx)>1)[0][i] == matched_idx)[0] for i in range(len(np.where(np.bincount(matched_idx)>1)[0]))][:-1]
# Manual intervention to exclude farms I believe have truly been infected multiple times
multiple_matches = [multiple_matches[i] for i in [0,2,3,4,5,6,8,9,11]] #[0,2,3,4,5,6,8,10]
keep_searching = []
for farm in multiple_matches:
    farm_size = np.sum(farms_old[matched_idx[farm[0]], 4:])
    inf_farm_size = np.sum(type_birds[:, farm].T, axis=1)
    new_farm = farm[np.argmax((farm_size-inf_farm_size) ** 2)]
    other_farms = farm[np.where(farm != new_farm)]
    matched_idx[new_farm] = 1e6
    matched_type[new_farm] = 0
    dist_farm[new_farm] = 0
    keep_searching.append(other_farms)

# Repeat process with best match removed for farms with multiple matches that I don't believe are the same
remove_list = matched_idx[matched_idx !=1e6]
farms_rem = np.copy(farms)
farms_rem[remove_list, :] = 1e10*np.ones_like(farms_rem[remove_list, :])
farms_x_rem = farms_rem[:, 2]
farms_y_rem = farms_rem[:, 3]
farms_birds_rem = np.sum(farms_rem[:, 4:], axis=1)
most_farm_bird_type_rem = np.argmax(farms_rem[:, 4:], axis=1)
both_farm_types_rem = (farms_rem[:, 4] > 0) & (farms_rem[:, 5] > 0)
for i in np.hstack(keep_searching):
    dist = ((farms_x_rem - inf_x[i]) ** 2 + (farms_y_rem - inf_y[i]) ** 2) ** (1/2)
    dist_idx = np.argsort(dist)
    dist = dist[dist_idx]
    dists_idx = dist_idx[dist < 2]
    dists = dist[dist < 2]
    correct_type = most_inf_bird_type[i] == most_farm_bird_type_rem[dists_idx]
    correct_size = ((farms_birds_rem[dists_idx] <= 100) & (farms_birds_rem[dists_idx] > 0) & (inf_birds[i] <= 100) & (inf_birds[i] > 0)) | \
                     ((farms_birds_rem[dists_idx] <= 1200) & (farms_birds_rem[dists_idx] > 40) & (inf_birds[i] <= 1200) & (inf_birds[i] > 40)) | \
                   ((farms_birds_rem[dists_idx] <= 12000) & (farms_birds_rem[dists_idx] > 800) & (inf_birds[i] <= 12000) & (inf_birds[i] > 800)) | \
                   ((farms_birds_rem[dists_idx] <= 120000) & (farms_birds_rem[dists_idx] > 8000) & (inf_birds[i] <= 120000) & (inf_birds[i] > 8000)) | \
                   ((farms_birds_rem[dists_idx] > 80000) & (inf_birds[i] > 80000))
    matched_idx[i] = dists_idx[np.where(correct_type & correct_size)[0][0]] if len(np.where(correct_type & correct_size)[0]) > 0 else 1e6
    dist_farm[i] = dists[np.where(correct_type & correct_size)[0][0]] if len(np.where(correct_type & correct_size)[0]) > 0 else 0
    if matched_idx[i] != 1e6:
        matched_type[i] = 1
    if matched_idx[i] == 1e6:
        if len(dists) > 0:
            if dists[0] < 0.2:
                matched_idx[i] = dists_idx[0]
                dist_farm[i] = dists[0]
                matched_type[i] = 2

    if matched_idx[i] == 1e6:
        correct_type_ish = (most_inf_bird_type_any[i]*np.ones(len(correct_size))).astype(bool)
        matched_idx[i] = dists_idx[np.where(correct_type_ish & correct_size)[0][0]] if len(
            np.where(correct_type_ish & correct_size)[0]) > 0 else 1e6
        dist_farm[i] = dists[np.where(correct_type_ish & correct_size)[0][0]] if len(
            np.where(correct_type_ish & correct_size)[0]) > 0 else 0
        if matched_idx[i] != 1e6:
            matched_type[i] = 3
    if matched_idx[i] == 1e6:
        correct_type_ish = both_inf_types[i] == both_farm_types_rem[dists_idx]
        small_size = (farms_birds[dists_idx] <= 100) & (farms_birds[dists_idx] > 0) & (inf_birds[i] <= 100) & (inf_birds[i] > 0)
        matched_idx[i] = dists_idx[np.where(correct_type_ish & small_size)[0][0]] if len(
            np.where(correct_type_ish & small_size)[0]) > 0 else 1e6
        dist_farm[i] = dists[np.where(correct_type_ish & small_size)[0][0]] if len(
            np.where(correct_type_ish & small_size)[0]) > 0 else 0
        if matched_idx[i] != 1e6:
            matched_type[i] = 4

multiple_matches = [np.where(np.where(np.bincount(matched_idx)>1)[0][i] == matched_idx)[0] for i in range(len(np.where(np.bincount(matched_idx)>1)[0]))][:-1]
# Check if there are still multiple matches
if len(multiple_matches) > 4:
    print(multiple_matches)
    raise ValueError("WARNING: Multiple matches still exist")

# Add farms to list that are still not matched
matched_farms = np.copy(matched_idx)
matched_farms[matched_idx == 1e6] = np.arange(farms_old.shape[0],farms_old.shape[0]+np.sum(matched_idx==1e6))
matched_farms = matched_farms.astype(int)
farms[matched_idx[matched_idx!=1e6].astype(int), 4:] = type_birds[:, matched_idx !=1e6].T
extra_farms = np.hstack((np.arange(46001,46001+np.sum(matched_idx==1e6))[:, np.newaxis], np.arange(9999901,9999901+np.sum(matched_idx==1e6))[:, np.newaxis], inf_x[np.where(matched_idx == 1e6)[0]][:, np.newaxis],inf_y[np.where(matched_idx == 1e6)[0]][:, np.newaxis],type_birds[:, np.where(matched_idx == 1e6)[0]].T))
farms = np.vstack((farms, extra_farms))

# Save data
farms_compare = np.zeros((len(inf_x), 4))
farms_compare[:, 0] = matched_farms
farms_compare[:, 1] = dist_farm
farms_compare[matched_farms < farms_old.shape[0], 2] = np.sum(farms_old[matched_farms[matched_farms < farms_old.shape[0]], 4:], axis=1)
farms_compare[:, 3] = np.sum(farms[matched_farms, 4:], axis=1)
np.savetxt('../Data/premises_data_model', farms, fmt='%d %d %f %f %d %d %d')
np.savetxt('../Data/matched_premises', matched_farms, fmt='%d')


