import ast
import json
import os, argparse
import hashlib
import numpy as np
import pandas as pd
from collections import defaultdict, deque
from typing import List, Tuple, Dict
from PIL import Image
from numpy.lib.stride_tricks import sliding_window_view
from time import time
import matplotlib.pyplot as plt


class OverlappingWFC:
    """
    Overlapping Wave Function Collapse (WFC) model for procedural pattern generation.
    Supports user-defined pattern size, overlap, periodic input, and pattern augmentation.
    Use (x, y) indexing for wave array access: wave[y, x].
    0: Up, 1: Right, 2: Down, 3: Left
    (row, col) = (y, x)
    """
    DIRS: List[Tuple[int, int]] = [(0,-1),(1,0),(0,1),(-1,0)]  # Directions (dx, dy): Up, Right, Down, Left
    UP, RIGHT, DOWN, LEFT = 0, 1, 2, 3
    ID2DIRS = {UP:'UP', RIGHT:'RIGHT', DOWN:'DOWN', LEFT:'LEFT'}
    OPPOSITE = {UP:DOWN, RIGHT:LEFT, DOWN:UP, LEFT:RIGHT}

    def __init__(self, sample, N=3, overlap=None, periodic_input=False, augment_rot_reflect=True):
        """
        Initialize the WFC model with input sample and parameters.
        Args:
            sample: Input image array (H,W) or (H,W,C)
            N: Pattern size (NxN)
            overlap: Overlap size (default N-1)
            periodic_input: If True, wraps input for seamless tiling
            augment_rot_reflect: If True, adds rotated/reflected patterns
        """
        self.N = N
        self.overlap = overlap if overlap is not None else N-1
        self.periodic_input = periodic_input
        self.augment_rot_reflect = augment_rot_reflect
        self.sample = sample if sample.ndim == 3 else sample[..., None]
        # time for pattern extraction
        start_time = time()
        self.key2idx, self.idx2arr, self.weights = self.extract_patterns()
        print("Pattern extraction took {:.2f} seconds.".format(time() - start_time))
        self.K = len(self.idx2arr)
        # time for building compatibility
        start_time = time()
        self.allow = self.build_compatibility()
        print("Building compatibility took {:.2f} seconds.".format(time() - start_time))

    @staticmethod
    def pattern_id(patch):
        """Return a hashable ID for a pattern patch."""
        return tuple(patch.reshape(-1).tolist())

    @staticmethod
    def rotate90(p):
        """Rotate a pattern patch by 90 degrees."""
        return np.rot90(p, k=1)

    @staticmethod
    def reflectX(p):
        """Reflect a pattern patch horizontally."""
        return np.flip(p, axis=1)

    def extract_patterns(self):
        """
        Extract all NxN overlapping patterns from the sample.
        Optionally augment with rotations/reflections.
        Returns:
            keyidx: dict mapping pattern hash key -> index
            idx2arr: list of pattern arrays
            weights: array of pattern weights (normalized counts)
        """
        N = self.N
        sample = self.sample
        periodic_input = self.periodic_input
        augment_rot_reflect = self.augment_rot_reflect
        if periodic_input:
            sample = np.pad(sample, ((0, N-1),(0, N-1),(0,0)), mode='wrap')
        H, W, C = sample.shape
        patches = sliding_window_view(sample, (N, N, C)) # shape = (H-N+1, W-N+1, 1, N, N, C)
        counts = defaultdict(int)
        def add_patch(p):
            hash_key = self.pattern_id(p)
            counts[hash_key] += 1
        # Slide window over input and collect patterns
        for y in range(patches.shape[0]):
            for x in range(patches.shape[1]):
                patch = patches[y, x][0]
                add_patch(patch)
                if augment_rot_reflect:
                    rp = patch
                    for _ in range(3):
                        rp = self.rotate90(rp)
                        add_patch(rp)
                    add_patch(self.reflectX(patch))
                    rp = self.reflectX(patch)
                    for _ in range(3):
                        rp = self.rotate90(rp)
                        add_patch(rp)
        # Build unique patterns and weights
        unique_keys = list(counts.keys())
        print("patch size {}".format(self.N))
        print("rotate and reflect augment:{}".format(self.augment_rot_reflect))
        print("Extracted {} unique patterns.".format(len(unique_keys)))
        key2idx = {}  # map hash_key to index
        idx2arr = []  # list of pattern arrays (N, N, C)
        for i, key in enumerate(unique_keys):
            arr = np.array(key).reshape(N, N, C)
            arr = arr.astype(self.sample.dtype)
            idx2arr.append(arr)
            key2idx[key] = i
        weights = np.array([counts[key] for key in unique_keys], dtype=np.float64)
        weights = weights / np.sum(weights)
        # dataframe for pattern ids and hash_key and counts
        catalog = pd.DataFrame({
            'pattern_id': range(len(unique_keys)),
            'hash_key': unique_keys,
            'count': [counts[key] for key in unique_keys],
            'weight': weights
        })
        self.catalog = catalog
        sample_grid = np.full((H-N+1, W-N+1), -1, dtype=int)
        for r in range(H-N+1):
            for c in range(W-N+1):
                patch = patches[r, c][0]
                key = self.pattern_id(patch)
                sample_grid[r, c] = key2idx[key]
        self.sample_grid = sample_grid
        # cache the empirical distribution from the raw sample extraction (no augmentation)
        sample_counts = np.bincount(sample_grid.reshape(-1), minlength=len(unique_keys)).astype(np.float64)
        self.sample_counts = sample_counts
        self.sample_distribution = sample_counts / sample_counts.sum()
        return key2idx, idx2arr, weights
    
    def build_compatibility(self):
        """
        Build compatibility sets for each pattern in all four directions.
        Returns:
            allow: list of sets of allowed neighbor indices for each direction
        """
        N = self.N
        overlap = self.overlap # default N-1
        idx2arr = self.idx2arr
        K = len(idx2arr)
        allow: List[List[set[int]]] = [ [set() for _ in range(4)] for _ in range(K) ]
        for i, A in enumerate(idx2arr):
            for j, B in enumerate(idx2arr):
                # Check border compatibility for each direction
                # B is above A
                if np.array_equal(A[:overlap, :, :], B[N-overlap:, :, :]):
                    allow[i][self.UP].add(j)
                # B is below A
                if np.array_equal(A[N-overlap:, :, :], B[:overlap, :, :]):
                    allow[i][self.DOWN].add(j)
                # B is left of A
                if np.array_equal(A[:, :overlap, :], B[:, N-overlap:, :]):
                    allow[i][self.LEFT].add(j)
                # B is right of A
                if np.array_equal(A[:, N-overlap:, :], B[:, :overlap, :]):
                    allow[i][self.RIGHT].add(j)
        # check if any empty compatibility sets
        for i in range(K):
            for d in range(4):
                if len(allow[i][d]) == 0:
                    print("Warning: Pattern index {} has no compatible neighbors in direction {}.".format(i, self.ID2DIRS[d]))
        # update catalog dataframe with compatible neighbors pattern ids
        for i in range(K):
            i_to_UP = str(tuple(allow[i][self.UP])) # neighbors above to pattern i
            i_to_RIGHT = str(tuple(allow[i][self.RIGHT])) # neighbors right to pattern i
            i_to_DOWN = str(tuple(allow[i][self.DOWN])) # neighbors below to pattern i
            i_to_LEFT = str(tuple(allow[i][self.LEFT])) # neighbors left to pattern i
            self.catalog.at[i, 'UP'] = i_to_UP
            self.catalog.at[i, 'RIGHT'] = i_to_RIGHT
            self.catalog.at[i, 'DOWN'] = i_to_DOWN
            self.catalog.at[i, 'LEFT'] = i_to_LEFT
        #print(self.catalog)
        return allow

    def plot_pattern(self, ax, arr):
        ax.pcolormesh(arr.astype(np.uint8), cmap='gray', vmin=0, vmax=255, edgecolors='blue', linewidth=0.2)
        ax.set_aspect('equal')
        ax.invert_yaxis()
        ax.axis('off')

    def plot_neighbors(self, idx):
        """
        plot the compatible neighbors for a given pattern index.
        Args:
            idx: pattern index
        """
        north_neighbors = ast.literal_eval(self.catalog.at[idx, 'UP'])
        east_neighbors = ast.literal_eval(self.catalog.at[idx, 'RIGHT'])
        south_neighbors = ast.literal_eval(self.catalog.at[idx, 'DOWN'])
        west_neighbors = ast.literal_eval(self.catalog.at[idx, 'LEFT'])
        j0 = np.random.choice(north_neighbors) if len(north_neighbors)>0 else None
        j1 = np.random.choice(east_neighbors) if len(east_neighbors)>0 else None
        j2 = np.random.choice(south_neighbors) if len(south_neighbors)>0 else None
        j3 = np.random.choice(west_neighbors) if len(west_neighbors)>0 else None
        choices = [j0, j1, j2, j3]
        arr_center = self.idx2arr[idx]
        arr_up = self.idx2arr[j0] if j0 is not None else None
        arr_right = self.idx2arr[j1] if j1 is not None else None
        arr_down = self.idx2arr[j2] if j2 is not None else None
        arr_left = self.idx2arr[j3] if j3 is not None else None
        fig, axes = plt.subplots(3, 3, figsize=(5, 5))
        cmap = 'gray' if self.sample.shape[2]==1 else None
        self.plot_pattern(axes[1, 1], arr_center)
        if arr_up is not None:
            self.plot_pattern(axes[0, 1], arr_up)
        if arr_right is not None:
            self.plot_pattern(axes[1, 2], arr_right)
        if arr_down is not None:
            self.plot_pattern(axes[2, 1], arr_down)
        if arr_left is not None:
            self.plot_pattern(axes[1, 0], arr_left)
        fig.suptitle(f"Pattern index: {idx} and its neighbors\nUP: {choices[0]}, RIGHT: {choices[1]}, DOWN: {choices[2]}, LEFT: {choices[3]}")
        plt.tight_layout()

    def get_transitions_matrix(self):
        K = self.K
        transitions = np.zeros((K, K, 4), dtype=int)
        for i in range(K):
            for d in range(4):
                for j in self.allow[i][d]:
                    transitions[i, j, d] += 1
        P_east = transitions[:, :, self.RIGHT]   # P_east(j|i) = transitions[i,j,RIGHT], conditional probability of neighbor j given current i
        P_south = transitions[:, :, self.DOWN]   # P_south(j|i) = transitions[i,j,DOWN], conditional probability of neighbor j given current i
        P_joint = {}
        for i in range(K):
            for j in range(K):
                idx, = np.nonzero(P_east[i, :] * P_south[j, :])
                P_joint[(i, j)] = set(idx)
        return transitions

    def observe(self, wave, weights, rng):
        """
        Pick the cell with minimal nontrivial entropy and collapse it to one pattern, weighted by multiplicity.
        wave: shape = (H, W, K), boolean array
        weights: shape = (K,), array of pattern weights
        rng: numpy random generator
        Returns:
            y, x: coordinates of observed cell
        """
        H, W, K = wave.shape
        options = wave.sum(axis=2) # shape = (H, W)
        mask = (options > 1) # cells with more than one possible pattern
        if not np.any(mask):
            # all cells are collapsed
            return None, None
        p = wave * weights
        p_sum = p.sum(axis=2, keepdims=True)
        p_norm = np.divide(p, p_sum, out=np.zeros_like(p), where=(p_sum>0)) # normalized probabilities
        entropy = -(p_norm * np.where(p_norm>0, np.log(p_norm), 0)).sum(axis=2) # shape = (H, W)
        entropy[~mask] = np.inf # set entropy of non-trivial cells to inf, so those are not selected
        # add small random noise to break ties
        entropy = entropy + rng.random(entropy.shape)*1e-6
        y, x = np.unravel_index(np.argmin(entropy), entropy.shape) # (y, x) is (row, col)
        allowed = np.where(wave[y, x])[0] # the allowed pattern ids at (y, x)
        if len(allowed) == 0:
            # contradiction
            return y, x
        local_w = weights[allowed]
        local_w = local_w / local_w.sum()
        choice = rng.choice(allowed, p=local_w)
        # ban all other patterns at (y, x) except choice
        wave[y, x, :] = False
        wave[y, x, choice] = True
        return y, x

    def propagate(self, wave, allow, start_cells: List[Tuple[int, int]]):
        """
        Arc-consistency propagation using a queue.
        Returns:
            True if propagation succeeds, False if contradiction occurs.
        """
        H, W, K = wave.shape  # cell grid height, width, number of patterns
        q = deque(start_cells)
        while q:
            x, y = q.popleft()
            for d, (dx, dy) in enumerate(self.DIRS):
                # (0, 1, 2, 3) = (UP, RIGHT, DOWN, LEFT)
                nx, ny = x + dx, y + dy
                if nx < 0 or ny < 0 or nx >= W or ny >= H:
                    # outside grid
                    continue
                changed = False
                possible_here = np.where(wave[y, x])[0] # possible pattern ids at current cell
                if possible_here.size == 0:
                    # contradiction
                    return False
                neighbor_possible = np.where(wave[ny, nx])[0]
                supported = np.zeros_like(wave[ny, nx], dtype=bool)
                support_set = set()
                for i in possible_here:
                    support_set |= allow[i][d]
                for j in neighbor_possible:
                    if j in support_set:
                        supported[j] = True
                to_ban = np.where(wave[ny, nx] & ~supported)[0]
                if to_ban.size > 0:
                    wave[ny, nx, to_ban] = False
                    changed = True # the cell state (ny, nx) has changed
                if changed:
                    if not wave[ny, nx].any():
                        # contradiction: no possible patterns left
                        return False
                    q.append((nx, ny)) # add neighbor cell to queue for further propagation
        return True

    def run(self, cell_grid_h, cell_grid_w, seed=None):
        """
        Run the WFC algorithm to collapse the cell grid.
        Args:
            cell_grid_h, cell_grid_w: cell grid size (height, width)
            seed: random seed for reproducibility
        Returns:
            collapsed_grid: (cell_grid_h, cell_grid_w) array of pattern indices (token IDs)
        """
        # Initialize wave function with all patterns possible
        wave = np.ones((cell_grid_h, cell_grid_w, self.K), dtype=bool)
        rng = np.random.default_rng(seed)
        while True:
            oy, ox = self.observe(wave, self.weights, rng)
            if oy is None:
                # all cells are collapsed
                break
            if isinstance(oy, int) and isinstance(ox, int) and not wave[oy, ox].any():
                raise RuntimeError("Contradiction while observing (empty domain).")
            ok = self.propagate(wave, self.allow, start_cells=[(ox, oy)])
            if not ok:
                raise RuntimeError("Contradiction during propagation; try a new seed or enable backtracking.")
        collapsed_grid = np.argmax(wave, axis=2)
        return collapsed_grid

    def render(self, collapsed_grid, output_fn, blend_average=True):
        """
        Render the output pixel image from the collapsed grid by blending overlapping patterns.
        save the output pixel image as a PNG file.
        save the collapsed grid as a npy file.
        Args:
            collapsed_grid: shape = (cell_grid_h, cell_grid_w), 2D array of pattern indices
        Returns:
            outimg: output image array in pixel space
        """
        cell_grid_h, cell_grid_w = collapsed_grid.shape
        N, _, C = self.idx2arr[0].shape
        idx2arr = np.array(self.idx2arr)
        if blend_average:
            # blending average overlapping regions
            out_H, out_W = cell_grid_h + N - 1, cell_grid_w + N - 1
            outimg = np.zeros((out_H, out_W, C), dtype=np.float32)
            for r in range(cell_grid_h):
                for c in range(cell_grid_w):
                    outimg[r:r+N, c:c+N, :] += idx2arr[collapsed_grid[r, c]]
            count_img = np.zeros((out_H, out_W, C), dtype=np.float32)
            for r in range(cell_grid_h):
                for c in range(cell_grid_w):
                    count_img[r:r+N, c:c+N, :] += 1.0
            outimg = outimg / count_img
        else:
            # remove overlapping regions by taking every N-th pixel
            arr5D = idx2arr[collapsed_grid]  # (cell_grid_h, cell_grid_w, N, N, C)
            arr5D = arr5D.transpose(0,2,1,3,4)  # (cell_grid_h, N, cell_grid_w, N, C)
            raw_tiled_image = arr5D.reshape(cell_grid_h*N, cell_grid_w*N, C)  # no overlap, just tile side by side
            print(raw_tiled_image.shape, raw_tiled_image.dtype, raw_tiled_image.max(), raw_tiled_image.min())
            # remove overlapping regions by taking every N-th pixel
            outimg = raw_tiled_image[0::N, 0::N, :]
            # append last N-1 rows and columns to match expected output size
            outimg = np.concatenate([outimg, raw_tiled_image[-(N-1):, 0::N, :]], axis=0)
            last_n_1_cols = np.concatenate([raw_tiled_image[0::N, -(N-1):, :], raw_tiled_image[-(N-1):, -(N-1):, :]], axis=0)
            outimg = np.concatenate([outimg, last_n_1_cols], axis=1)
        outimg = outimg.astype(self.sample.dtype)
        if C==1:
            outimg = outimg[:, :, 0]
        print(outimg.shape, outimg.dtype, outimg.max(), outimg.min())
        np.savez_compressed(output_fn + "_collapsed_grid.npz", collapsed_grid=collapsed_grid)
        Image.fromarray(outimg).save(output_fn + ".png")
        return outimg

    def analyze_output_distribution(self, wfc_grid, use_augmented_counts=False, epsilon=1e-12):
        """
        Compare pattern distribution of a generated grid against the sample via KL-divergence.
        Args:
            wfc_grid: 2D array of pattern indices produced by run(...)
            use_augmented_counts: if True, compare against augmented pattern counts (rot/flip);
                otherwise use the raw NxN extraction from the input sample.
            epsilon: numerical floor to avoid log(0)
        Returns:
            dict with sample/output counts and distributions plus KL value (sample || output)
        """
        if wfc_grid.ndim != 2:
            raise ValueError("wfc_grid must be a 2D array of pattern indices.")
        if wfc_grid.size == 0:
            raise ValueError("wfc_grid is empty.")
        if wfc_grid.max() >= self.K or wfc_grid.min() < 0:
            raise ValueError("wfc_grid contains invalid pattern indices.")

        # output distribution from generated grid
        out_counts = np.bincount(wfc_grid.reshape(-1), minlength=self.K).astype(np.float64)
        out_dist = out_counts / out_counts.sum()

        # sample distribution baseline
        if use_augmented_counts:
            sample_counts = np.array([self.catalog.at[i, 'count'] for i in range(self.K)], dtype=np.float64)
        else:
            sample_counts = getattr(self, "sample_counts", None)
            if sample_counts is None:
                sample_counts = np.bincount(self.sample_grid.reshape(-1), minlength=self.K).astype(np.float64)
                self.sample_counts = sample_counts
                self.sample_distribution = sample_counts / sample_counts.sum()
        sample_dist = sample_counts / sample_counts.sum()

        safe_out = np.clip(out_dist, epsilon, 1.0)
        safe_sample = np.clip(sample_dist, epsilon, 1.0)
        kl = float(np.sum(safe_sample * np.log(safe_sample / safe_out)))
        return {
            "kl_divergence": kl,
            "sample_counts": sample_counts,
            "sample_distribution": sample_dist,
            "output_counts": out_counts,
            "output_distribution": out_dist,
            "missing_in_output": np.where((sample_counts > 0) & (out_counts == 0))[0],
        }

    def plot_distribution_bars(self, sample_dist, output_dist, title="Pattern distribution"):
        """
        Plot side-by-side bars comparing sample vs generated pattern distributions.
        """
        ids = np.arange(self.K)
        width = 0.4
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.bar(ids - width / 2, sample_dist, width=width, label="sample")
        ax.bar(ids + width / 2, output_dist, width=width, label="output")
        ax.set_xlabel("pattern id")
        ax.set_ylabel("probability")
        ax.set_title(title)
        ax.legend()
        ax.set_xticks(ids)
        ax.set_xticklabels(ids, rotation=90)
        plt.tight_layout()
        return fig, ax
    
    def markov_random_field(
        self,
        H,
        W,
        seed=None,
        start_token=None,
        max_sweeps=100,
        max_restarts=100,
        repair_radius=1,
    ):
        """
        Generate a field using the Markov Random Field model.
        Args:
            H, W: output field size (height, width)
            seed: random seed for reproducibility
            start_token: optional starting token index at (0,0)
            max_sweeps: number of Gibbs-like refinement sweeps before checking validity
            max_restarts: number of times to randomize the entire grid if conflicts persist
            repair_radius: radius of local patches randomized when conflicts appear
        Returns:
            grid: (H, W) array of pattern indices (token IDs)
        """
        rng = np.random.default_rng(seed)
        tokens = np.arange(self.K)
        weights = self.weights
        full_token_set = set(range(self.K))

        def init_grid():
            g = rng.choice(tokens, size=(H, W), p=weights)
            if start_token is not None:
                g[0, 0] = start_token
            return g

        def candidate_set(r, c, grid):
            """Return candidate tokens compatible with all current neighbors."""
            constraints = []
            if r > 0:
                constraints.append(self.allow[grid[r-1, c]][self.DOWN])
            if r + 1 < H:
                constraints.append(self.allow[grid[r+1, c]][self.UP])
            if c > 0:
                constraints.append(self.allow[grid[r, c-1]][self.RIGHT])
            if c + 1 < W:
                constraints.append(self.allow[grid[r, c+1]][self.LEFT])
            if not constraints:
                return set(full_token_set)
            candidates = set(constraints[0])
            for neighbor_set in constraints[1:]:
                candidates &= neighbor_set
                if not candidates:
                    break
            return candidates

        def grid_is_valid(grid):
            """Check local compatibility for all edges to guard against false positives."""
            for r in range(H):
                for c in range(W):
                    token = grid[r, c]
                    if r > 0:
                        north = grid[r - 1, c]
                        if token not in self.allow[north][self.DOWN]:
                            return False
                        if north not in self.allow[token][self.UP]:
                            return False
                    if c > 0:
                        west = grid[r, c - 1]
                        if token not in self.allow[west][self.RIGHT]:
                            return False
                        if west not in self.allow[token][self.LEFT]:
                            return False
            return True

        grid = init_grid()
        for restart in range(max_restarts):
            for _ in range(max_sweeps):
                changed = False
                conflict_cells = []
                order = rng.permutation(H * W)
                for idx in order:
                    r, c = divmod(idx, W)
                    if start_token is not None and r == 0 and c == 0:
                        continue
                    candidates = candidate_set(r, c, grid)
                    if not candidates:
                        conflict_cells.append((r, c))
                        continue
                    cand_list = np.fromiter(candidates, dtype=int)
                    probs = weights[cand_list]
                    if probs.sum() == 0:
                        probs = np.ones_like(probs) / probs.size
                    else:
                        probs = probs / probs.sum()
                    new_val = rng.choice(cand_list, p=probs)
                    if new_val != grid[r, c]:
                        grid[r, c] = new_val
                        changed = True
                if not conflict_cells and (not changed or grid_is_valid(grid)):
                    if grid_is_valid(grid):
                        return grid
                if conflict_cells:
                    for r, c in conflict_cells:
                        r0 = max(0, r - repair_radius)
                        r1 = min(H, r + repair_radius + 1)
                        c0 = max(0, c - repair_radius)
                        c1 = min(W, c + repair_radius + 1)
                        patch_shape = (r1 - r0, c1 - c0)
                        grid[r0:r1, c0:c1] = rng.choice(tokens, size=patch_shape, p=weights)
                    if start_token is not None:
                        grid[0, 0] = start_token
            if grid_is_valid(grid):
                return grid
            grid = init_grid()

        raise RuntimeError(
            "MRF synthesis failed to find a consistent assignment; try increasing max_sweeps or max_restarts."
        )

    def energy_minimization(
        self,
        H,
        W,
        seed=None,
        max_iters=200000,
        max_restarts=5,
        temp_start=1.0,
        temp_end=0.05,
        debug_energy=False,
    ):
        """
        Simulated annealing-style energy minimization on the pattern grid.
        Energy counts local compatibility violations; target energy 0 means a valid tiling.
        """
        rng = np.random.default_rng(seed)
        tokens = np.arange(self.K)
        weights = self.weights

        def init_grid():
            return rng.choice(tokens, size=(H, W), p=weights)

        def local_energy(grid, r, c, token=None):
            """Energy contribution around cell (r,c); counts incompatible neighbor edges."""
            if token is None:
                token = grid[r, c]
            e = 0
            if r > 0:
                north = grid[r - 1, c]
                if token not in self.allow[north][self.DOWN]:
                    e += 1
                if north not in self.allow[token][self.UP]:
                    e += 1
            if r + 1 < H:
                south = grid[r + 1, c]
                if token not in self.allow[south][self.UP]:
                    e += 1
                if south not in self.allow[token][self.DOWN]:
                    e += 1
            if c > 0:
                west = grid[r, c - 1]
                if token not in self.allow[west][self.RIGHT]:
                    e += 1
                if west not in self.allow[token][self.LEFT]:
                    e += 1
            if c + 1 < W:
                east = grid[r, c + 1]
                if token not in self.allow[east][self.LEFT]:
                    e += 1
                if east not in self.allow[token][self.RIGHT]:
                    e += 1
            return e

        def total_energy(grid):
            e = 0
            for r in range(H):
                for c in range(W):
                    e += local_energy(grid, r, c)
            return e * 0.5  # each edge counted twice

        for restart in range(max_restarts):
            grid = init_grid()
            best_grid = grid.copy()
            best_e = total_energy(grid)
            if debug_energy:
                print(f"[energy] restart {restart+1}/{max_restarts}, init energy={best_e}")
            if best_e == 0:
                return grid
            for it in range(max_iters):
                t = temp_start + (temp_end - temp_start) * (it / max_iters)
                r = rng.integers(0, H)
                c = rng.integers(0, W)
                current = grid[r, c]
                proposal = rng.choice(tokens, p=weights)
                if proposal == current:
                    continue
                e_before = local_energy(grid, r, c, current)
                e_after = local_energy(grid, r, c, proposal)
                delta = e_after - e_before
                if delta <= 0 or rng.random() < np.exp(-delta / max(t, 1e-6)):
                    grid[r, c] = proposal
                # periodic full energy check
                if it % 200 == 0 or delta < 0:
                    e_tot = total_energy(grid)
                    if debug_energy and it % 1000 == 0:
                        print(f"[energy] restart {restart+1} iter {it}: energy={e_tot}")
                    if e_tot < best_e:
                        best_e = e_tot
                        best_grid = grid.copy()
                    if e_tot == 0:
                        return grid
            # restart with best grid so far if valid, else randomize
            if best_e == 0:
                return best_grid
        raise RuntimeError("Energy minimization failed to find a consistent assignment; try increasing max_iters or max_restarts.")

# ----------------------------
# Example usage
# ----------------------------
if __name__ == "__main__":
    parse = argparse.ArgumentParser(description="Run Overlapping WFC on a sample image.")
    parse.add_argument("input_png_file", type=str, help="Path to input sample image.")
    parse.add_argument('--patch_size', type=int, help="Size of the patterns (NxN).", default=3)
    parse.add_argument('--out_size', type=int, nargs=2, help="Output size (height width).", default=[64,64])
    parse.add_argument('--seed', type=int, help="Random seed for reproducibility.", default=None)
    parse.add_argument('--num_trials', type=int, help="Number of synthesis trials (unique seeds each).", default=1)
    parse.add_argument('--periodic_input', action='store_true', help="Use periodic input for seamless tiling.")
    parse.add_argument('--augment_rot_reflect', action='store_true', help="Use data augmentation (rotation/reflection).")
    parse.add_argument('--convert_grayscale', action='store_true', help="Convert input image to grayscale.")
    parse.add_argument('--run_wfc', action='store_true', help="Run the WFC algorithm.")
    parse.add_argument('--method', choices=['wfc', 'energy', 'both'], default='wfc', help="Synthesis method to use.")
    parse.add_argument('--debug_energy', action='store_true', help="Print energy over iterations for energy minimization.")
    args = parse.parse_args()
        
    # Load input sample image and convert to grayscale numpy array
    sample = Image.open(args.input_png_file)
    if args.convert_grayscale:
        sample = sample.convert("L")
    sample = np.array(sample)
    print("sample input", sample.shape, sample.dtype, sample.min(), sample.max())
        
    # Set output size and pattern size
    out_H, out_W, N = args.out_size[0], args.out_size[1], args.patch_size
    cell_grid_h = out_H - N + 1
    cell_grid_w = out_W - N + 1
    # Initialize WFC model, pattern extraction, and build compatibility table
    wfc = OverlappingWFC(sample, N=N, overlap=N-1,
        periodic_input=args.periodic_input,
        augment_rot_reflect=args.augment_rot_reflect,
        )
    #wfc.weights = np.ones_like(wfc.weights) / len(wfc.weights)  # uniform weights
    print("sample grid of pattern indices:\n", wfc.sample_grid)
    output_dir = os.path.splitext(os.path.basename(args.input_png_file))[0] + "_WFC_outputs"
    os.makedirs(output_dir, exist_ok=True)
    output_fn = "overlapN{}_out{}x{}".format(N, out_H, out_W)
    output_fn = os.path.join(output_dir, output_fn)
        
    if args.run_wfc:
        methods = []
        if args.method in ("wfc", "both"):
            methods.append("wfc")
        if args.method in ("energy", "both"):
            methods.append("energy")

        rng_trials = np.random.default_rng(args.seed)
        total_start = time()
        trial_records = []
        method_data = {
            m: {
                "records": [],
                "grids": [],
                "imgs": [],
                "seeds": [],
                "failed": [],
                "last_analysis": None,
                "last_img": None,
            } for m in methods
        }

        for trial in range(args.num_trials):
            trial_seed = int(rng_trials.integers(0, 2**32 - 1, dtype=np.uint32))
            for method in methods:
                start_time = time()
                try:
                    if method == "wfc":
                        grid = wfc.run(cell_grid_h, cell_grid_w, seed=trial_seed)
                    else:
                        grid = wfc.energy_minimization(
                            cell_grid_h,
                            cell_grid_w,
                            seed=trial_seed,
                            debug_energy=args.debug_energy and args.num_trials == 1,
                        )
                    analysis = wfc.analyze_output_distribution(
                        grid,
                        use_augmented_counts=args.augment_rot_reflect,
                    )
                    method_data[method]["grids"].append(grid)
                    method_data[method]["seeds"].append(trial_seed)
                    method_data[method]["last_analysis"] = analysis
                    record = {
                        "method": method,
                        "trial": trial + 1,
                        "seed": trial_seed,
                        "status": "success",
                        "kl_divergence": analysis["kl_divergence"],
                        "missing_patterns": ";".join(map(str, analysis["missing_in_output"].tolist())) if analysis["missing_in_output"].size > 0 else "",
                        "elapsed_sec": time() - start_time,
                        "error": "",
                    }
                    trial_records.append(record)
                    method_data[method]["records"].append(record)
                    trial_base = f"{output_fn}_{method}_trial{trial+1}_seed{trial_seed}"
                    out_img = wfc.render(grid, output_fn=trial_base, blend_average=True)
                    method_data[method]["imgs"].append(out_img)
                    method_data[method]["last_img"] = out_img
                    if args.num_trials == 1 and method == "wfc":
                        print("Pattern KL-divergence (sample||output): {:.6f}".format(analysis["kl_divergence"]))
                        print("wfc_grid shape:", grid.shape, "patterns used:", np.unique(grid).size, "/", wfc.K)
                        if analysis["missing_in_output"].size > 0:
                            print("Pattern ids missing in output but present in sample:", analysis["missing_in_output"].tolist())
                        else:
                            print("All sample patterns appeared in the output.")
                        wfc.plot_distribution_bars(
                            sample_dist=analysis["sample_distribution"],
                            output_dist=analysis["output_distribution"],
                            title="Pattern distribution (sample vs output)",
                        )
                    print(f"[trial {trial+1}/{args.num_trials}] {method} succeeded in {time() - start_time:.2f}s (seed={trial_seed})")
                except Exception as exc:
                    method_data[method]["failed"].append((trial_seed, str(exc)))
                    record = {
                        "method": method,
                        "trial": trial + 1,
                        "seed": trial_seed,
                        "status": "fail",
                        "kl_divergence": "",
                        "missing_patterns": "",
                        "elapsed_sec": time() - start_time,
                        "error": str(exc),
                    }
                    trial_records.append(record)
                    method_data[method]["records"].append(record)
                    print(f"[trial {trial+1}/{args.num_trials}] {method} FAILED (seed={trial_seed}): {exc}")

        elapsed = time() - total_start
        print(f"Total synthesis wall time: {elapsed:.2f}s")

        for method in methods:
            records = method_data[method]["records"]
            successes = [r for r in records if r["status"] == "success"]
            success = len(successes)
            print(f"{method.upper()} success: {success}/{args.num_trials} ({success/args.num_trials:.2%})")
            if method_data[method]["failed"]:
                failed = method_data[method]["failed"]
                print(f"{method.upper()} failed seeds (seed, error):", failed[:10], ("... (+more)" if len(failed) > 10 else ""))
            last_analysis = method_data[method]["last_analysis"]
            if last_analysis is not None and args.num_trials > 1:
                print(f"{method.upper()} last successful trial KL-divergence (sample||output): {last_analysis['kl_divergence']:.6f}")
                if last_analysis["missing_in_output"].size > 0:
                    print(f"{method.upper()} last trial missing patterns:", last_analysis["missing_in_output"].tolist())
            grids = method_data[method]["grids"]
            imgs = method_data[method]["imgs"]
            if imgs:
                out_arr = np.stack(imgs, axis=0)
                grid_arr = np.stack(grids, axis=0)
                seeds_arr = np.array(method_data[method]["seeds"], dtype=np.uint32)
                all_npz = os.path.join(
                    output_dir,
                    f"all_outputs_{method}_N{N}_out{out_H}x{out_W}_{args.num_trials}_runs_seed{args.seed if args.seed is not None else 'rng'}.npz",
                )
                np.savez_compressed(all_npz, seeds=seeds_arr, outputs=out_arr, collapsed_grids=grid_arr)
                print(f"Saved aggregated outputs for {method} to", all_npz)
            if len(grids) > 1:
                flat = [g.reshape(-1) for g in grids]
                hashes = [hashlib.sha1(g.tobytes()).hexdigest() for g in grids]
                unique_hashes = len(set(hashes))
                pairwise_diff = []
                for i in range(len(flat)):
                    for j in range(i+1, len(flat)):
                        diff = np.mean(flat[i] != flat[j])
                        pairwise_diff.append(diff)
                mean_diff = float(np.mean(pairwise_diff)) if pairwise_diff else 0.0
                min_diff = float(np.min(pairwise_diff)) if pairwise_diff else 0.0
                max_diff = float(np.max(pairwise_diff)) if pairwise_diff else 0.0
                print(f"{method.upper()} similarity (token grids): unique={unique_hashes}/{len(grids)}, "
                      f"pairwise token mismatch mean={mean_diff:.4f}, min={min_diff:.4f}, max={max_diff:.4f}")

        # Cross-method comparison if both produced outputs
        if "wfc" in methods and "energy" in methods:
            w_map = {s: g for s, g in zip(method_data["wfc"]["seeds"], method_data["wfc"]["grids"])}
            e_map = {s: g for s, g in zip(method_data["energy"]["seeds"], method_data["energy"]["grids"])}
            common = sorted(set(w_map.keys()) & set(e_map.keys()))
            if common:
                diffs = []
                identical = 0
                for s in common:
                    wg = w_map[s]
                    eg = e_map[s]
                    mismatch = np.mean(wg.reshape(-1) != eg.reshape(-1))
                    diffs.append(mismatch)
                    if mismatch == 0:
                        identical += 1
                mean_diff = float(np.mean(diffs))
                print(f"Cross-method (wfc vs energy) on {len(common)} shared seeds: mean token mismatch={mean_diff:.4f}, identical outputs={identical}")

        if trial_records:
            trial_df = pd.DataFrame(trial_records)
            trials_csv = os.path.join(
                output_dir,
                f"trials_N{N}_out{out_H}x{out_W}_{args.num_trials}_runs_seed{args.seed if args.seed is not None else 'rng'}.csv",
            )
            trial_df.to_csv(trials_csv, index=False)
            print("Saved trial log to", trials_csv)

        # Plot input and output images for single-trial runs
        if args.run_wfc and args.num_trials == 1:
            plot_methods = [m for m in methods if method_data[m]["last_img"] is not None]
            if plot_methods:
                cmap = 'gray' if sample.ndim == 2 else None
                fig, axes = plt.subplots(1, len(plot_methods) + 1, figsize=(4*(len(plot_methods)+1), 4))
                axes = np.atleast_1d(axes)
                axes[0].set_title("Sample Input: {}".format(sample.shape))
                axes[0].pcolor(sample, cmap=cmap, edgecolors='blue', linewidth=0.1)
                axes[0].invert_yaxis()
                for i, method in enumerate(plot_methods, start=1):
                    img = method_data[method]["last_img"]
                    axes[i].set_title(f"{method.upper()} Output: {img.shape}, N={N}")
                    axes[i].pcolor(img, cmap=cmap, edgecolors='blue', linewidth=0.1)
                    axes[i].invert_yaxis()
                plt.tight_layout()
                plt.show()
