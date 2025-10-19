import os, sys
import numpy as np
from collections import defaultdict, deque
import random
from PIL import Image
from numpy.lib.stride_tricks import sliding_window_view
import matplotlib.pyplot as plt

class OverlappingWFC:
    """
    Overlapping Wave Function Collapse (WFC) model for procedural pattern generation.
    Supports user-defined pattern size, overlap, periodic input, and pattern augmentation.
    """
    DIRS = [(0,-1),(1,0),(0,1),(-1,0)]  # Directions: Up, Right, Down, Left
    UP, RIGHT, DOWN, LEFT = 0, 1, 2, 3
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
        self.pid2idx, self.idx2arr, self.weights = self.extract_patterns()
        self.K = len(self.idx2arr)
        self.allow = self.build_compatibility()

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
            pid2idx: dict mapping pattern_id -> index
            idx2arr: list of pattern arrays
            weights: array of pattern weights (normalized counts)
        """
        N = self.N
        sample = self.sample
        overlap = self.overlap
        periodic_input = self.periodic_input
        augment_rot_reflect = self.augment_rot_reflect
        if periodic_input:
            sample = np.pad(sample, ((0, overlap),(0, overlap),(0,0)), mode='wrap')
        H, W, C = sample.shape
        patches = sliding_window_view(sample, (N, N, C))
        counts = defaultdict(int)
        def add_patch(p):
            pid = self.pattern_id(p)
            counts[pid] += 1
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
        unique = list(counts.keys())
        print("Extracted {} unique patterns.".format(len(unique)))
        id_to_array = {}
        for pid in unique:
            arr = np.array(pid)
            arr = arr.reshape(N, N, C)
            id_to_array[pid] = arr
        weights = np.array([counts[pid] for pid in unique], dtype=np.float64)
        weights = weights / np.sum(weights)
        pid2idx = {pid:i for i, pid in enumerate(unique)}
        idx2arr = [id_to_array[pid] for pid in unique]
        return pid2idx, idx2arr, weights

    def build_compatibility(self):
        """
        Build compatibility sets for each pattern in all four directions.
        Returns:
            allow: list of sets of allowed neighbor indices for each direction
        """
        N = self.N
        overlap = self.overlap
        idx2arr = self.idx2arr
        K = len(idx2arr)
        allow = [ [set() for _ in range(4)] for _ in range(K) ]
        for i, A in enumerate(idx2arr):
            for j, B in enumerate(idx2arr):
                # Check border compatibility for each direction
                if np.array_equal(A[:overlap, :, :], B[N-overlap:, :, :]):
                    allow[i][self.UP].add(j)
                if np.array_equal(A[N-overlap:, :, :], B[:overlap, :, :]):
                    allow[i][self.DOWN].add(j)
                if np.array_equal(A[:, :overlap, :], B[:, N-overlap:, :]):
                    allow[i][self.LEFT].add(j)
                if np.array_equal(A[:, N-overlap:, :], B[:, :overlap, :]):
                    allow[i][self.RIGHT].add(j)
        return allow

    def observe(self, wave, weights, rng):
        """
        Pick the cell with minimal nontrivial entropy and collapse it to one pattern, weighted by multiplicity.
        Returns:
            y, x: coordinates of observed cell
        """
        H, W, K = wave.shape
        options = wave.sum(axis=2)
        mask = (options > 1)
        if not np.any(mask):
            return None, None
        p = wave * weights
        p_sum = p.sum(axis=2, keepdims=True)
        p_norm = np.divide(p, p_sum, out=np.zeros_like(p), where=(p_sum>0))
        entropy = -(p_norm * np.where(p_norm>0, np.log(p_norm), 0)).sum(axis=2)
        entropy[~mask] = np.inf
        entropy = entropy + rng.random(entropy.shape)*1e-6
        y, x = np.unravel_index(np.argmin(entropy), entropy.shape)
        allowed = np.where(wave[y, x])[0]
        if len(allowed) == 0:
            return y, x
        local_w = weights[allowed]
        local_w = local_w / local_w.sum()
        choice = rng.choice(allowed, p=local_w)
        wave[y, x, :] = False
        wave[y, x, choice] = True
        return y, x

    def propagate(self, wave, allow, start_cells):
        """
        Arc-consistency propagation using a queue.
        Returns:
            True if propagation succeeds, False if contradiction occurs.
        """
        H, W, K = wave.shape
        q = deque(start_cells)
        while q:
            x, y = q.popleft()
            for d, (dx, dy) in enumerate(self.DIRS):
                nx, ny = x + dx, y + dy
                if nx < 0 or ny < 0 or nx >= W or ny >= H:
                    continue
                changed = False
                possible_here = np.where(wave[y, x])[0]
                if possible_here.size == 0:
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
                    changed = True
                if changed:
                    if not wave[ny, nx].any():
                        return False
                    q.append((nx, ny))
        return True

    def run(self, Hp, Wp, seed=None):
        """
        Run the WFC algorithm to collapse the grid.
        Args:
            Hp, Wp: grid size (height, width)
            seed: random seed for reproducibility
        Returns:
            collapsed_grid: (Hp, Wp) array of pattern indices
        """
        wave = np.ones((Hp, Wp, self.K), dtype=bool)
        rng = np.random.default_rng(seed)
        while True:
            oy, ox = self.observe(wave, self.weights, rng)
            if oy is None:
                break
            if isinstance(oy, int) and isinstance(ox, int) and not wave[oy, ox].any():
                raise RuntimeError("Contradiction while observing (empty domain).")
            ok = self.propagate(wave, self.allow, start_cells=[(ox, oy)])
            if not ok:
                raise RuntimeError("Contradiction during propagation; try a new seed or enable backtracking.")
        collapsed_grid = np.argmax(wave, axis=2)
        return collapsed_grid

    def render(self, collapsed_grid):
        """
        Render the output image from the collapsed grid by blending overlapping patterns.
        Args:
            collapsed_grid: (Hp, Wp) array of pattern indices
        Returns:
            outimg: output image array
        """
        H, W = collapsed_grid.shape
        N, _, C = self.idx2arr[0].shape
        idx2arr = np.array(self.idx2arr)
        arr5D = idx2arr[collapsed_grid]
        outimg_raw = np.concatenate([np.concatenate(arr5D[_, :], axis=1) for _ in range(H)], axis=0)
        out_H, out_W = H + N - 1, W + N - 1
        outimg = np.zeros((out_H, out_W, C), dtype=outimg_raw.dtype)
        for r in range(H):
            for c in range(W):
                outimg[r:r+N, c:c+N, :] += idx2arr[collapsed_grid[r, c]]
        count_img = np.zeros((out_H, out_W, C), dtype=np.float32)
        for r in range(H):
            for c in range(W):
                count_img[r:r+N, c:c+N, :] += 1.0
        outimg = outimg / count_img
        if C==1:
            outimg = outimg[:, :, 0]
        return outimg

# ----------------------------
# Example usage
# ----------------------------
if __name__ == "__main__":
    # Load input sample image and convert to grayscale numpy array
    sample = Image.open(sys.argv[1])
    sample = sample.convert("L")
    sample = np.array(sample)
    print("sample input", sample.shape, sample.dtype, sample.min(), sample.max())
    # Set output size and pattern size
    out_H, out_W, N = (64, 64, 8)
    Hp = out_H - N + 1
    Wp = out_W - N + 1
    # Initialize WFC model
    wfc = OverlappingWFC(sample, N=N, overlap=N-1, periodic_input=True, augment_rot_reflect=True)
    # Run WFC and render output
    collapsed_grid = wfc.run(Hp, Wp, seed=42)
    out = wfc.render(collapsed_grid)
    # Plot input and output images
    fig, axes = plt.subplots(1,2, figsize=(10,5))
    axes[0].set_title("Sample Input")
    axes[0].pcolor(sample, cmap='gray', edgecolors='blue', linewidth=0.1)
    axes[0].invert_yaxis()
    axes[1].set_title("Blended Output")
    axes[1].pcolor(out, cmap='gray', edgecolors='blue', linewidth=0.1)
    axes[1].invert_yaxis()
    plt.show()