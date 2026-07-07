#!/usr/bin/env python3
import os, math, pickle
from multiprocessing import Pool, cpu_count
from insect_nav.memory import PerfectMemory
from insect_nav.vision import loadFrame, countFrames
from insect_nav.parameters import load_parameters_from_file
from tqdm import tqdm
import numpy as np


class PmTeacher:
    def __init__(self, base_path, train_dataset_path, test_dataset_path, max_frames_to_test):
        """
        Args:
            base_path: path in which is expected to find the parameters.json
            train_dataset_path:
            test_dataset_path:
        """
        self.base_path = base_path
        self.parameters_path = os.path.join(self.base_path, "parameters.json")
        self.train_dataset_path = train_dataset_path
        self.test_dataset_path = test_dataset_path
        self.output_path = os.path.join(self.base_path, "pm_angles.pkl")
        
        self.train_frame_count = countFrames(self.train_dataset_path)
        self.test_frame_count = min(countFrames(self.test_dataset_path), max_frames_to_test)
        self.train_frames = self.load_frames(self.train_dataset_path)
        self.test_frames = self.load_frames(self.test_dataset_path)
        
        self.parameters = load_parameters_from_file(self.parameters_path)
        
        self.pool = None
        self.num_processes = cpu_count()
    
    def load_frames(self, dataset_path):
        frames = []
        for frame_number in range(0, countFrames(dataset_path)):
            frame = loadFrame(frame_number, frames_dir=dataset_path)
            frames.append(frame)
        return frames
    
    def train(self):
        pm = PerfectMemory(self.parameters)
        print(f"[PM] Training PerfectMemory.")
        for frame in tqdm(self.train_frames, desc="PM Training"):
            pm.train(frame)
        pm.save_weights()
        
    def test(self):
        print(f"[PM] Testing PerfectMemory in parallel ({self.num_processes} workers)...")

        frame_numbers = list(range(0, self.test_frame_count))
        chunk_size = len(frame_numbers) // self.num_processes + 1
        chunks = [frame_numbers[i:i + chunk_size] for i in range(0, len(frame_numbers), chunk_size)]

        with Pool(processes=self.num_processes) as p:
            try:
                results = p.starmap(self.worker, [(chunk, self.parameters) for chunk in chunks])
            except KeyboardInterrupt:
                p.terminate()
                raise

        # Flatten into dict
        all_results = [item for sublist in results for item in sublist]
        pm_angles = {frame_number: deg for frame_number, deg in all_results}

        # Save to pickle so workers can reload
        with open(self.output_path, "wb") as f:
            pickle.dump(pm_angles, f)

        print(f"[PM] Baseline saved: {self.output_path}")
        
    def worker(self, frame_range, params):
        """Worker: compute PM outputs for a block of frames"""
        pm = PerfectMemory(params, load_net=True, num_shifts=40)
        results = []
        for frame_number in frame_range:
            frame = loadFrame(frame_number, frames_dir=self.test_dataset_path)
            rad = pm.testNavigation(frame, frame_number=frame_number, debug_print=False)
            pm.plot_test_results(frame_number, params["plotsTestPath"])
            pm.saveFiguresToCsv(frame, frame_number, params["plotsTestPath"])
            deg = np.degrees(rad)
            results.append((frame_number, deg))
        return results

    def train_and_test(self):
        self.train()
        self.test()

    # Load PerfectMemory angles from disk (worker processes do not share memory).
    def load_pm_angles(self):
        """Load precomputed PM angles from pickle"""
        with open(self.output_path, "rb") as f:
            return pickle.load(f)

