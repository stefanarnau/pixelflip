import numpy as np
import sys    
import scipy.stats as stats

# Test parameters
pval_voxel   = 0.025;
pval_cluster = 0.01;
n_perm      = 1000;


# Input data: List of n-observations (x by y data matrices)
data_1 = []
data_2 = []

# Data as 3d matrices
d1 = np.stack(data_1)    
d2 = np.stack(data_2)

# Check for equalness of shapes
if d1.shape != d2.shape:
    sys.exit('Input data dimension mismatch...')

# Get dimensions
n_obs, n_x, n_y = d1.shape


# Collectors
permuted_t = []
max_clust = []

# Permutation loop
for p in range(n_perm):

    # random boolean mask for which values will be swapped
    flip_mask = np.random.randint(0, 2, size=n_obs).astype(np.bool)

    # Swap
    d1_permuted = d1
    d1_permuted[flip_mask, :, :] = d2[flip_mask, :, :]
    d2_permuted = d2
    d2_permuted[flip_mask, :, :] = d1[flip_mask, :, :]
    
    # T-test shit
    t_vals, p_vals = stats.ttest_rel(d1_permuted, d2_permuted, axis=0, nan_policy='propagate', alternative='two-sided')

    # Binarize
    binmat = np.where(p_vals > pval_cluster, 1, 0)

    tnum = squeeze(mean(d1_perm - d2_perm, 1));
    tdenum = squeeze(std(d1_perm - d2_perm, 0, 1)) / sqrt(length(subject_list));
    fake_t = tnum ./ tdenum;
    permuted_t(p, :, :) = fake_t;
    
    
    
    fake_t(abs(fake_t) < tinv(1 - pval_voxel, length(subject_list) - 1)) = 0;
    clusts = bwconncomp(fake_t);
    sum_t = [];
    for clu = 1 : numel(clusts.PixelIdxList)
        cidx = clusts.PixelIdxList{clu};
        sum_t(end + 1) = sum(fake_t(cidx));
    end
    max_clust(p, 1) = min([0, sum_t]);
    max_clust(p, 2) = max([0, sum_t]);      
end


tnum = squeeze(mean(d1 - d2, 1));
tdenum = squeeze(std(d1 - d2, 0, 1)) / sqrt(length(subject_list));
tmat = tnum ./ tdenum;
tvals = tmat;
tmat(abs(tmat) < tinv(1 - pval_voxel, length(subject_list) - 1)) = 0;
threshtvals = tmat;
clusts = bwconncomp(tmat);
sum_t = [];
for clu = 1 : numel(clusts.PixelIdxList)
    cidx = clusts.PixelIdxList{clu};
    sum_t(end + 1) = sum(tmat(cidx));
end
clust_thresh_lower = prctile(max_clust(:, 1), pval_cluster * 100);
clust_thresh_upper = prctile(max_clust(:, 2), 100 - pval_cluster * 100);
clust2remove = find(sum_t > clust_thresh_lower & sum_t < clust_thresh_upper);
for clu = 1 : length(clust2remove)
    tmat(clusts.PixelIdxList{clust2remove(clu)}) = 0;
end

% Get clusters
cluster_outlines = logical(tmat);