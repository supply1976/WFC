import ast
import json
import os, argparse
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
    
    def markov_random_field(self, H, W, seed=None, start_token=None):
        """
        Generate a field using the Markov Random Field model.
        Args:
            H, W: output field size (height, width)
            seed: random seed for reproducibility
            start_token: optional starting token index at (0,0)
        Returns:
            grid: (H, W) array of pattern indices (token IDs)
        """
        # row-major order implementation
        rng = np.random.default_rng(seed)
        if start_token is None:
            start_token = rng.choice(np.arange(self.K), p=self.weights)
        # initialize grid with -1 (wildcard)
        grid = np.full((H, W), -1, dtype=int)
        grid[0, 0] = start_token
        
        for r in range(H):
            for c in range(W):
                if r == 0 and c == 0:
                    continue
                # start with all possible tokens
                possible_tokens = set(range(self.K))
                if r > 0:
                    north_token = grid[r-1, c]
                    if north_token != -1:
                        possible_tokens &= self.allow[north_token][self.DOWN]
                if c > 0:
                    west_token = grid[r, c-1]
                    if west_token != -1:
                        possible_tokens &= self.allow[west_token][self.RIGHT]
                if possible_tokens:
                    selected = rng.choice(list(possible_tokens))
                    grid[r, c] = selected
                else:
                    # no candidates; leave as -1 wildcard
                    grid[r, c] = -1
        return grid

# ----------------------------
# Example usage
# ----------------------------
if __name__ == "__main__":
    parse = argparse.ArgumentParser(description="Run Overlapping WFC on a sample image.")
    parse.add_argument("input_png_file", type=str, help="Path to input sample image.")
    parse.add_argument('--patch_size', type=int, help="Size of the patterns (NxN).", default=3)
    parse.add_argument('--out_size', type=int, nargs=2, help="Output size (height width).", default=[64,64])
    parse.add_argument('--seed', type=int, help="Random seed for reproducibility.", default=None)
    parse.add_argument('--periodic_input', action='store_true', help="Use periodic input for seamless tiling.")
    parse.add_argument('--augment_rot_reflect', action='store_true', help="Use data augmentation (rotation/reflection).")
    parse.add_argument('--convert_grayscale', action='store_true', help="Convert input image to grayscale.")
    parse.add_argument('--run_wfc', action='store_true', help="Run the WFC algorithm.")
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
    # transitions = wfc.get_transitions_matrix()
    # check
    #wfc.plot_neighbors(idx=58)
    output_dir = os.path.splitext(os.path.basename(args.input_png_file))[0] + "_WFC_outputs"
    os.makedirs(output_dir, exist_ok=True)
    output_fn = "overlapN{}_out{}x{}".format(N, out_H, out_W)
    output_fn = os.path.join(output_dir, output_fn)
    
    if args.run_wfc:
        start_time = time()
        # Run WFC and render output
        wfc_grid = wfc.run(cell_grid_h, cell_grid_w, seed=args.seed)
        mrf_grid = wfc.markov_random_field(cell_grid_h, cell_grid_w, seed=args.seed, start_token=wfc_grid[0,0])
        wfc_out = wfc.render(wfc_grid, output_fn=output_fn, blend_average=True)
        mrf_out = wfc.render(mrf_grid, output_fn=output_fn + "_MRF", blend_average=True)
        print("WFC generation and rendering took {:.2f} seconds.".format(time() - start_time))

        # Plot input and output images
        cmap = 'gray' if sample.ndim == 2 else None
        fig, axes = plt.subplots(1,3, figsize=(12,4))
        axes[0].set_title("Sample Input: {}".format(sample.shape))
        axes[0].pcolor(sample, cmap=cmap, edgecolors='blue', linewidth=0.1)
        axes[0].invert_yaxis()
        axes[1].set_title("WFC Output: {}, patch size = {}".format(wfc_out.shape, N))
        axes[1].pcolor(wfc_out, cmap=cmap, edgecolors='blue', linewidth=0.1)
        axes[1].invert_yaxis()
        axes[2].set_title("MRF Output: {}, patch size = {}".format(mrf_out.shape, N))
        axes[2].pcolor(mrf_out, cmap=cmap, edgecolors='blue', linewidth=0.1)
        axes[2].invert_yaxis()

    plt.show()
