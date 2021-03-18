###########################################################################
# Imports
###########################################################################
# Standard library imports
from automan.api import Simulation, Problem, Automator
from automan.automation import filter_cases
import matplotlib.pyplot as plt
import numpy as np

###########################################################################
# Code
###########################################################################


def list_input_filter_cases(runs, predicate=None, **params):
    if predicate is not None:
        if callable(predicate):
            return list(filter(predicate, runs))
        else:
            params['predicate'] = predicate

    def _check_match(run):
        for param, expected in params.items():
            if param not in run.params:
                return False
            if isinstance(expected, list):
                for item in expected:
                    if run.params[param] == item:
                        return True
                return False
            else:
                if run.params[param] != expected:
                    return False

        return True

    return list(filter(_check_match, runs))


class ComplexSimulatedAnnealing(Problem):
    def get_name(self):
        return 'complex_simulated_annealing'

    def _plot_cost_history(self, ext='', **kwargs):
        fig, axs = plt.subplots(1, 1, figsize=(12, 6))

        cases = list_input_filter_cases(self.cases, **kwargs)
        for case in cases:
            data = case.data
            cost_hist = data['cost_hist']
            epochs = np.arange(1, len(cost_hist) + 1)
            axs.plot(epochs, cost_hist, label=case.name, linewidth=0.85)

        plt.legend()
        plt.xlabel(r'Epochs $\rightarrow$')
        plt.ylabel(r'Cost history $\rightarrow$')
        fig.savefig(
            self.output_path(ext + 'cost_history.png'),
            dpi=400, bbox_inches='tight'
        )
        plt.close()

    def _plot_runtimes(self, ext='', **kwargs):
        fig, axs = plt.subplots(1, 1, figsize=(12, 6))

        cases = list_input_filter_cases(self.cases, **kwargs)
        rts, names = [], []
        for case in cases:
            data = case.data
            rts.append(data['rt'])
            names.append(case.name)

        ypos = np.arange(len(rts))
        axs.barh(ypos, rts)
        axs.set_yticks(ypos)
        axs.set_yticklabels(names)
        axs.invert_yaxis()  # labels read top-to-bottom
        axs.set_xlabel(r'Runtime $(in s) \rightarrow$')

        fig.savefig(
            self.output_path(ext + 'runtime.png'),
            dpi=400, bbox_inches='tight'
        )
        plt.close()

    def setup(self):
        get_path = self.input_path

        # Base commands
        code_name = 'code/simulated_annealing/complex_simulated_annealing.py'
        base_cmd = f'python {code_name} --s --d $output_dir'

        self.epochs = 1000

        # Make cases
        self.cases = [
            Simulation(
                root=get_path(f'e_{self.epochs}_n_{i}'),
                job_info=dict(n_core=1, n_thread=1),
                base_command=base_cmd,
                n_epoch=i,
                epoch=self.epochs,
                n=80,
                T=200,
            )
            for i in [50, 100, 200, 400]
        ]

    def run(self):
        self.make_output_dir()
        self._plot_cost_history(ext='epoch_1000_', epoch=self.epochs)
        self._plot_runtimes(ext='epoch_1000_', epoch=self.epochs)


class SimpleSimulatedAnnealing(ComplexSimulatedAnnealing):
    def get_name(self):
        return 'simple_simulated_annealing'

    def setup(self):
        get_path = self.input_path

        # Base commands
        code_name = 'code/simulated_annealing/simple_simulated_annealing.py'
        base_cmd = f'python {code_name} --s --d $output_dir'

        self.epochs = 1000

        # Make cases
        self.cases = [
            Simulation(
                root=get_path(f'asym_e_{self.epochs}_n_{i}'),
                job_info=dict(n_core=1, n_thread=1),
                base_command=base_cmd,
                n_epoch=i,
                epoch=self.epochs,
                cfunc='exp',
                alpha=1e-4,
                T=1e7,
                tcn=3,
                sym='asym'
            )
            for i in [50, 100, 200, 400]
        ]

        self.cases.extend([
            Simulation(
                root=get_path(f'sym_e_{self.epochs}_n_{i}'),
                job_info=dict(n_core=1, n_thread=1),
                base_command=base_cmd,
                n_epoch=i,
                epoch=self.epochs,
                cfunc='exp',
                alpha=1e-4,
                T=1e7,
                tcn=3,
                sym='sym'
            )
            for i in [50, 100, 200, 400]
        ]
        )

    def run(self):
        self.make_output_dir()
        self._plot_cost_history(
            ext='asym_epoch_1000_', epoch=self.epochs, sym='asym'
        )
        self._plot_cost_history(
            ext='sym_epoch_1000_', epoch=self.epochs, sym='sym'
        )
        self._plot_runtimes(
            ext='asym_epoch_1000_', epoch=self.epochs, sym='asym'
        )
        self._plot_runtimes(
            ext='sym_epoch_1000_', epoch=self.epochs, sym='sym'
        )


###########################################################################
# Main Code
###########################################################################
if __name__ == '__main__':
    PROBLEMS = [
        SimpleSimulatedAnnealing, ComplexSimulatedAnnealing
    ]
    automator = Automator(
        simulation_dir='output/simulated_annealing/automate',
        output_dir='output/simulated_annealing/automate/figures',
        all_problems=PROBLEMS
    )
    automator.run()
