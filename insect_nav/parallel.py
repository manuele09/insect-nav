import multiprocessing as mp
import os, time, math, atexit, signal
import numpy as np

from insect_nav import NeuralNetwork
from insect_nav import NeuralModelBase
from insect_nav.infomax import Infomax
from insect_nav.memory import PerfectMemory


"""
Parallel Navigator Module
--------------------------

Implements a multiprocessing extension of `NeuralModelBase` that distributes
frame evaluation across multiple CPU cores.
"""

# --------------------------------------------------------------------------
# Global worker state (each process keeps its own network instance)
# --------------------------------------------------------------------------
_worker_nn = None
_worker_id = None
_worker_counter = mp.Value('i', 0)   # Shared atomic counter to assign worker IDs


def _cleanup_worker():
    """
    Automatically called when a worker process terminates.
    Ensures that neural networks are unloaded cleanly to free memory and GPU resources.
    """
    global _worker_nn, _worker_id
    if isinstance(_worker_nn, NeuralNetwork):
        print(f"[Worker {_worker_id}] Unloading NeuralNetwork resources...")
        _worker_nn.unload()
    _worker_nn = None


def _init_worker(params):
    """
    Initialize a worker process with its own copy of the neural network.

    Args:
        params (dict): Model configuration dictionary.
    """
    global _worker_nn, _worker_id

    # Let the main process handle Ctrl+C; workers should not process SIGINT directly.
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    # Assign unique worker ID atomically
    with _worker_counter.get_lock():
        wid = _worker_counter.value
        _worker_counter.value += 1
    _worker_id = wid

    # Each worker gets its own model instance
    params_copy = dict(params)
    params_copy["name"] = f"worker_{wid}"
    _worker_nn = None

    # Instantiate the appropriate network
    if params_copy["network_type"] == "spiking":
        _worker_nn = NeuralNetwork(params_copy, load_net={"pn_kc": True, "kc_mbon": True})
    elif params_copy["network_type"] == "infomax":
        _worker_nn = Infomax(params_copy, load_net=True, calculate_mean=False)
    elif params_copy["network_type"] == "perfect_memory":
        _worker_nn = PerfectMemory(params_copy, load_net=True)

    # Register automatic cleanup when process exits
    atexit.register(_cleanup_worker)
    print(f"[Worker {_worker_id}] Initialized network ({params_copy['network_type']}).")


def _run_shift(shift_degree, frame):
    """
    Perform a single test step for a specific angular shift.

    Args:
        shift_degree (float): Shift angle in degrees.
        frame (np.ndarray): Input image to evaluate.

    Returns:
        tuple: (shift_degree, novelty_score)
    """
    global _worker_nn, _worker_id
    novelty = _worker_nn.test(frame, shift_degree)
    return shift_degree, novelty


# --------------------------------------------------------------------------
# ParallelNavigator Class
# --------------------------------------------------------------------------

class ParallelNavigator(NeuralModelBase):
    """
    Multiprocessing-based navigator for biologically inspired visual neural networks.

    Extends `NeuralModelBase` with parallel testing capabilities:
        • Spawns multiple worker processes, each with its own neural model.
        • Distributes angular shift evaluations across all workers.
        • Aggregates results to determine optimal navigation direction.

    Attributes:
        n_processes (int): Number of worker processes to spawn.
        pool (mp.Pool): Multiprocessing pool managing worker lifetime.
    """

    def __init__(self, parameters, n_processes=None, num_shifts=None):
        """
        Initialize the parallel navigator and spawn worker pool.

        Args:
            parameters (dict): Configuration dictionary for the neural model.
            n_processes (int, optional): Number of worker processes (default: CPU count).
        """
        super().__init__(parameters, load_net=True, num_shifts=num_shifts)
        self.n_processes = n_processes or mp.cpu_count()

        # Initialize multiprocessing pool with persistent workers
        self.pool = mp.Pool(
            processes=self.n_processes,
            initializer=_init_worker,
            initargs=(self.params,)
        )
        print(f"[Main] ParallelNavigator initialized with {self.n_processes} workers.")

    # ----------------------------------------------------------------------
    # Overridden navigation testing
    # ----------------------------------------------------------------------

    def testNavigation(self, frame, frame_number=-1, log_path=None, debug_print=True, return_timing=False):
        """
        Evaluate navigation novelty in parallel across angular shifts.

        Args:
            frame (np.ndarray): Input frame for evaluation.
            frame_number (int): Frame index (for logging).
            log_path (str): Path to save test logs.

        Returns:
            float | tuple[float, float]:
                - Default: normalized turning command in radians (-pi to +pi).
                - If return_timing=True: (turning command, elapsed_time_seconds).
        """
        start_time = time.time()
        self.degree_array.clear()
        self.novelty_array.clear()

        # Define shifts and corresponding angles
        shifts = [(-self.num_shifts / 2 + k) * self.params["DEGREES_PER_SHIFT"]
                  for k in range(self.num_shifts + 1)]
        angles = [((deg + 180) % 360) - 180 for deg in shifts]

        # Distribute evaluation across workers
        results = self.pool.starmap(_run_shift, [(deg, frame) for deg in angles])
        results.sort(key=lambda x: x[0])  # Sort by degree

        # Aggregate results
        self.degree_array = [deg for deg, _ in results]
        self.novelty_array = [novelty for _, novelty in results]

        # Determine best navigation direction
        best_degree, uncertainty = self.find_optimal_degree()
        self.last_best_degree = best_degree
        elapsed_time = time.time() - start_time

        # Optional logging
        if log_path:
            self._log_navigation_results(frame_number, best_degree, uncertainty, log_path)

        if debug_print:
            print(f"[Main] Best Degree: {best_degree:.2f}°, Uncertainty: {uncertainty:.4f}, Time: {elapsed_time*1000:.3f} ms")

        angle_rad = math.radians(-best_degree)
        if return_timing:
            return angle_rad, elapsed_time
        return angle_rad

    # ----------------------------------------------------------------------
    # Resource cleanup
    # ----------------------------------------------------------------------

    def _get_pool_workers(self, pool):
        workers = getattr(pool, "_pool", None)
        if not workers:
            return []
        return [w for w in workers if w is not None]

    def _wait_workers(self, workers, timeout_s):
        timeout_s = max(0.0, float(timeout_s))
        deadline = time.time() + timeout_s

        for proc in workers:
            remaining = deadline - time.time()
            if remaining <= 0:
                break
            try:
                proc.join(timeout=remaining)
            except Exception:
                pass

        return all((not p.is_alive()) for p in workers)

    def _kill_workers(self, workers):
        for proc in workers:
            if not proc.is_alive():
                continue
            try:
                if hasattr(proc, "kill"):
                    proc.kill()
                else:
                    os.kill(proc.pid, signal.SIGKILL)
            except Exception:
                pass

    def close(self, force=False, timeout_s=3.0):
        """
        Gracefully terminate all worker processes and reset the shared counter.
        Should be called before shutting down the node or script.

        Args:
            force (bool): If True, terminate workers immediately.
            timeout_s (float): Max wait time in seconds before hard-kill.
        """
        print("[Main] Closing multiprocessing pool...")
        if not hasattr(self, "pool") or self.pool is None:
            return

        pool = self.pool
        self.pool = None
        workers = self._get_pool_workers(pool)

        try:
            if force:
                pool.terminate()
            else:
                pool.close()
        except KeyboardInterrupt:
            force = True
            try:
                pool.terminate()
            except Exception:
                pass
        except Exception as e:
            print(f"[Main] Error while closing pool: {e}. Forcing terminate.")
            force = True
            try:
                pool.terminate()
            except Exception:
                pass

        exited = self._wait_workers(workers, timeout_s=timeout_s)
        if not exited:
            print("[Main] Pool close timeout reached, forcing worker kill...")
            try:
                pool.terminate()
            except Exception:
                pass
            self._kill_workers(workers)
            exited = self._wait_workers(workers, timeout_s=1.0)
            if not exited:
                print("[Main] Warning: some workers still alive after hard-kill attempt.")

        _worker_counter.value = 0
        print("[Main] ✅ All worker processes closed cleanly.")
