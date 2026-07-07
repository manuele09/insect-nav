import os, re, datetime, math, csv
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from multiprocessing import Lock
from typing import List, Dict
from .pm_teacher import PmTeacher
from .utilities import *
from insect_nav.infomax import Infomax
from insect_nav import NeuralNetwork
from insect_nav.parameters import load_parameters_from_file, save_parameters_to_file
from insect_nav.vision import *
from scipy.optimize import differential_evolution, minimize
import pandas as pd
from time import time
lock = Lock()

class Tuner:
    def __init__(self, base_path_pm, output_path, 
                 train_dataset_path, test_dataset_path, max_frames_to_test,
                 network_type, base_net_param_path, net_bounds, 
                 popsize=20, mutation=0.8, recombination=0.7, cpuParallelization=True):
        self.network_type = network_type
        if self.network_type not in ["spiking", "infomax"]:
            print(f"Unrecognized network type: {self.network_type}")
            exit()

        self.pm_teacher = PmTeacher(base_path_pm, train_dataset_path, test_dataset_path, max_frames_to_test)
        # self.pm_teacher.train_and_test()
        self.output_path = output_path
        self.base_net_param_path = base_net_param_path
        self.net_bounds = net_bounds
            
        # Defining useful output paths
        self.plot_path = os.path.join(self.output_path, "live_error_plot.png")
        self.rng_path = os.path.join(self.output_path, "rng")
        self.populations_path = os.path.join(self.output_path, "populations.csv")
        self.best_individual_path = os.path.join(self.output_path, "best_individual.csv")
        
        self.popsize = popsize
        self.mutation = mutation
        self.recombination = recombination
        self.rng = np.random.default_rng(123)
        self.initial_population = None
        self.start_generation = 0
        
        self.cpuParallelization = cpuParallelization
        
        self.start_time = time()
    

    def resumeTuning(self, last_generation):
        self.load_rng_state(last_generation)
        self.load_generation(last_generation)
        self.start_generation = last_generation
    
    def startTuning(self):
        init_mod = 'latinhypercube' # default 
        if self.initial_population is not None:
            init_mod = self.initial_population
        
        if self.cpuParallelization:
            result = differential_evolution(
                self.objective,
                bounds=self.net_bounds,
                popsize=self.popsize,
                init=init_mod,
                mutation=self.mutation,
                recombination=self.recombination,
                workers=-1,
                updating="deferred",
                seed = self.rng,
                callback=self.callback
            )
        else:
            result = differential_evolution(
                self.objective,
                bounds=self.net_bounds,
                popsize=self.popsize,
                init=init_mod,
                mutation=0.8,
                recombination=0.7,
                seed = self.rng,
                callback=self.callback
            )


    def train_network(self, parameters):
        """Train network with given params on TRAIN dataset"""
        print("[Tuner] Training network ...")
        if self.network_type == "spiking":
            nn = NeuralNetwork(
                parameters,
                load_net={"pn_kc": False, "kc_mbon": False},
                tuneCurrent=True, use_gpu=True)
            nn.model.unload()
            nn = NeuralNetwork(
                parameters,
                load_net={"pn_kc": True, "kc_mbon": False},
                tuneCurrent=False, use_gpu=False)
            nn.logger.enable_novelty_tracking()
        else:
            nn = Infomax(parameters, load_net=False, calculate_mean=True)
        for i in range(0, self.pm_teacher.train_frame_count, parameters["train_step"]):
            nn.train(self.pm_teacher.train_frames[i])
        nn.save_weights()

        if self.network_type == "spiking":
            novelties = nn.logger.get_novelties()
            parameters["activated_kcs"] = novelties["total_unique_kcs"]
            print(parameters["activated_kcs"])
            nn.model.unload()
        save_parameters_to_file(parameters, parameters["parameters_path"])

    def test_network(self, parameters):
        """Test network network on TEST dataset and compute diff MAE wrt PM"""
        print("[Tuner] Testing network...")
        parameters = load_parameters_from_file(parameters["parameters_path"])
        if self.network_type == "spiking":
            nn = NeuralNetwork(
                parameters,
                load_net={"pn_kc": True, "kc_mbon": True},
                tuneCurrent=False, num_shifts=40, use_gpu=False)
        else:
            nn = Infomax(
                parameters, 
                load_net=True, 
                calculate_mean=False, num_shifts=40)
            
        # Load PM results locally
        pm_angles = self.pm_teacher.load_pm_angles()
        
        error = 0.0
        for frame_number in range(0, self.pm_teacher.test_frame_count):        
            rad = nn.testNavigation(self.pm_teacher.test_frames[frame_number], debug_print=False)
            deg = math.degrees(rad)
            error += abs(deg - pm_angles[frame_number])
        if self.network_type == "spiking":
            nn.model.unload()
            nn.delete_build_directory()
        error = error / self.pm_teacher.test_frame_count
        parameters["error"] = error
        print(f'Error: {parameters["error"]}')
        save_parameters_to_file(parameters, parameters["parameters_path"])
        return error

    def objective(self, vars):
        """
        Objective function for DE.
        Returns MAE diff wrt PM (to minimize).
        """
        params = vars_to_params_dict(vars, self.network_type, self.base_net_param_path, self.output_path)
        # ricorda: la chiamata a questa funzione va a sovrascrivere il parameters.json esistente
        self.train_network(params)
        error = self.test_network(params)
        return error

    def callback(self, intermediate_result):
        print(f"Elapsed time: {time() - self.start_time} seconds")
        generation = intermediate_result.nit + self.start_generation
        convergence = intermediate_result.convergence
        print(f"Generation: {generation}, Convergence: {convergence}, Error: {intermediate_result.fun}, Solution: {intermediate_result.x}")
        for pop in intermediate_result.population:
            self.write_vars_to_csv(self.populations_path, pop, generation, convergence)
        self.write_vars_to_csv(self.best_individual_path, intermediate_result.x, generation, convergence)
        self.save_rng_state(generation)
        self.update_progress_plot()

    # -----------------------------------------------------------------------
    # Utilities functions
    # -----------------------------------------------------------------------

    def get_header(self):
        """Generate header for csv log file.

        Returns:
            List[Str]: header, in the format of a list of strings
        """
        base_header = ["Timestamp", "Generation", "PopName", "Error", "Convergence"]
        if self.network_type == "spiking":
            extra_header = [
                        "Target_KCs", "Activated_KCs",
                        "Input_Scale", "Vertical_Weight", "Horizontal_Weight",
                        "PN_KC_FanIn", "PN_KC_Weight", "Vthresh", "Train_Step",
                        "mean_num_kc_fired", "std_num_kc_fired", 
                        "mean_num_spikes_kc", "std_num_spikes_kc",
                    ]
        else:
            extra_header = [
                "Learning_Rate", "Output_Units", "Train_Step"
            ]
        header = base_header + extra_header
        return header

    def vars_to_row(self, vars, generation, convergence):
        """Given a list of variables wich define an individual in a population,
        and its associated generation and convergence values, return a row (a 
        list of strings), to be saved for example into a csv file.

        Args:
            vars (List): 
            generation (int): 
            convergence (float): 

        Returns:
            List[Str]: _description_
        """
        # Extract the name of the individual, and use
        # it to locate its parameters.json
        name = vars_to_name(vars, self.network_type)
        base_folder = self.output_path / name
        params = load_parameters_from_file(base_folder / "parameters.json")
        
        # Data common to all network types
        row = [datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 
            generation, name, 
            params["error"], convergence]
        
        if self.network_type == "spiking":
            row.extend([
            params["target_kcs"], params["activated_kcs"],
            params["INPUT_SCALE"], params["VERTICAL_WEIGHT"], params["HORIZONTAL_WEIGHT"],
            params["PN_KC_FAN_IN"], params["PN_KC_WEIGHT"], params["IF_PARAMS"]["Vthresh"], params["train_step"],
            params["mean_num_kc_fired"], params["std_num_kc_fired"],
            params["mean_num_spikes_kc"], params["std_num_spikes_kc"]
            ])
        else:
            row.extend([
                params["learning_rate"], params["output_units"], params["train_step"],
            ])
        return row
    
    def write_vars_to_csv(self, csv_path, vars, generation, convergence):
        if not os.path.exists(csv_path):
            header = self.get_header()
            with lock:
                with open(csv_path, "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(header)
                    f.flush()
                    os.fsync(f.fileno())

        
        row = self.vars_to_row(vars, generation, convergence)
        with lock:
            with open(csv_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(row)
                f.flush()
                os.fsync(f.fileno())

    def save_rng_state(self, generation: int):
        """
        Salva lo stato del generatore RNG alla generazione indicata.

        Parameters
        ----------
        rng : np.random.Generator
            Generatore NumPy
        generation : int
            Indice di generazione
        path : str
            Directory dove salvare il file
        """
        os.makedirs(self.rng_path, exist_ok=True)
        filename = os.path.join(self.rng_path, f"rng_state_gen_{generation}.npy")
        np.save(filename, self.rng.bit_generator.state, allow_pickle=True)

    def load_rng_state(self, generation: int) -> np.random.Generator:
        """
        Carica lo stato del generatore RNG dalla generazione indicata
        e restituisce un Generator ripristinato.

        Parameters
        ----------
        generation : int
            Indice di generazione
        path : str
            Directory dove è salvato il file

        Returns
        -------
        rng : np.random.Generator
            Generatore con stato ripristinato
        """
        filename = os.path.join(self.rng_path, f"rng_state_gen_{generation}.npy")
        state = np.load(filename, allow_pickle=True).item()

        self.rng = np.random.default_rng()
        self.rng.bit_generator.state = state
        
    def load_generation(self, generation: int) -> List[List]:
        rows = []
        with open(self.populations_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if "Generation" not in reader.fieldnames or "PopName" not in reader.fieldnames:
                raise ValueError(f"CSV senza colonne richieste. Trovate: {reader.fieldnames}")

            for r in reader:
                try:
                    gen = int(r["Generation"])
                except (TypeError, ValueError):
                    raise ValueError(f"Valore Generation non valido: {r.get('Generation')}")

                if gen == generation:
                    rows.append(r)

        if len(rows) != self.popsize * len(self.net_bounds):
            raise ValueError(
                f"Popolazione non valida per generation={generation}: "
                f"trovate {len(rows)} righe, attese {self.popsize * len(self.net_bounds)}."
            )

        self.initial_population = []
        for r in rows:
            popname = r["PopName"]
            vars_list = name_to_vars(popname, self.network_type)
            self.initial_population.append(vars_list)
        
    def update_progress_plot(self):
        """Update and save MAE progress plot reading best individuals from CSV."""
        with lock:
            # Leggi errori dal CSV dei best (1 riga per generazione)
            errors = []
            try:
                with open(self.best_individual_path, "r", newline="", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    if reader.fieldnames is None or "Error" not in reader.fieldnames:
                        raise ValueError(
                            f"CSV best individuals senza colonna 'Error'. Colonne trovate: {reader.fieldnames}"
                        )

                    for row in reader:
                        val = row.get("Error", "")
                        if val is None or val == "":
                            continue
                        try:
                            errors.append(float(val))
                        except ValueError:
                            # salta righe corrotte/non numeriche
                            continue
            except FileNotFoundError:
                # Se il file non esiste ancora, non plottare
                return

            if not errors:
                return

            plt.figure(figsize=(8, 5))
            plt.plot(errors, marker="o", linestyle="-", alpha=0.8)

            plt.xlabel("Generations")
            plt.ylabel("Error")
            plt.title("Error best individual")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()

            plot_path = self.plot_path
            plt.savefig(plot_path)
            plt.close()

    def polish(self):
        df = pd.read_csv(self.best_individual_path)
        best = df.loc[df["Error"].idxmin()]
        vars = name_to_vars(best["PopName"], "spiking")
        
        result = minimize(self.objective, vars, args = (), method="L-BFGS-B")
        print(result)
        print(result.x)
        print(result.fun)
        print(result.message)