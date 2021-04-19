###########################################################################
# Imports
###########################################################################
# Standard library imports
from automan.api import Simulation, Problem, Automator
from automan.automation import filter_cases
import matplotlib.pyplot as plt
import numpy as np
from itertools import product

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


class ComplexSimulatedAnnealingParamTuning(Problem):
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

    def _plot_parameter_tuning(
        self, param_name, param_vals, cases_var, cases_vals, ext='',
        sim_cases=None
    ):
        if sim_cases is None:
            sim_cases = self.cases

        fig, axs = plt.subplots(1, 1, figsize=(12, 6))

        for case_item in cases_vals:
            final_cost, temp_x = [], []
            cases_attrs = {cases_var: case_item}
            for param_item in param_vals:
                attrs = {param_name: param_item}
                attrs.update(cases_attrs)
                cases = filter_cases(sim_cases, **attrs)
                for case in cases:
                    data = case.data
                    final_cost.append(data['final_sl'])
                    temp_x.append(param_item)
            axs.plot(
                temp_x, final_cost, label=f'{cases_var}:{case_item}',
                linewidth=0.85, marker='o'
            )

        plt.legend()
        plt.xlabel(f'{param_name}' + r'$\rightarrow$')
        plt.ylabel(r'Final Cost $\rightarrow$')
        plt.grid()
        fig.savefig(
            self.output_path(ext + f'{param_name}_final_cost.png'),
            dpi=400, bbox_inches='tight'
        )
        plt.close()

    def setup(self):
        get_path = self.input_path

        # Base commands
        code_name = 'code/simulated_annealing/complex_simulated_annealing.py'
        base_cmd = f'python {code_name} --s --mod-cost --d $output_dir'

        self.epochs, self.n_epochs = 1000, 200
        self.n = [40, 60, 80]
        N_param = 10

        # T
        self.T = np.linspace(30, 120, N_param).astype(int)
        self.cases_T = [
            Simulation(
                root=get_path(f'n_{num_nodes}_T_{param}'),
                job_info=dict(n_core=1, n_thread=1),
                base_command=base_cmd,
                n_epoch=self.n_epochs,
                epoch=self.epochs,
                n=num_nodes,
                t_max=num_nodes * 0.25,
                T=param,
            )
            for param, num_nodes in product(self.T, self.n)
        ]
        self.cases = self.cases_T.copy()

        # # Alpha
        self.alpha = np.append(
            np.round(np.linspace(0.5, 1.5, N_param), 3), 0.99
        )
        self.alpha = np.sort(self.alpha)
        self.cases_alpha = [
            Simulation(
                root=get_path(f'n_{num_nodes}_alpha_{param}'),
                job_info=dict(n_core=1, n_thread=1),
                base_command=base_cmd,
                n_epoch=self.n_epochs,
                epoch=self.epochs,
                n=num_nodes,
                t_max=num_nodes * 0.25,
                T=60,
                alpha=param,
            )
            for param, num_nodes in product(self.alpha, self.n)
        ]
        self.cases += self.cases_alpha

        # k
        self.k = np.append(
            np.round(np.linspace(0.01, 0.999, N_param), 3), 0.6
        )
        self.k = np.sort(self.k)
        self.cases_k = [
            Simulation(
                root=get_path(f'n_{num_nodes}_k_{param}'),
                job_info=dict(n_core=1, n_thread=1),
                base_command=base_cmd,
                n_epoch=self.n_epochs,
                epoch=self.epochs,
                n=num_nodes,
                t_max=num_nodes * 0.25,
                T=60,
                alpha=0.944,
                k=param,
            )
            for param, num_nodes in product(self.k, self.n)
        ]
        self.cases += self.cases_k

        # Delta
        self.delta = np.linspace(5, 30, N_param).astype(int)
        self.cases_delta = [
            Simulation(
                root=get_path(f'n_{num_nodes}_delta_{param}'),
                job_info=dict(n_core=1, n_thread=1),
                base_command=base_cmd,
                n_epoch=self.n_epochs,
                epoch=self.epochs,
                n=num_nodes,
                t_max=num_nodes * 0.25,
                T=60,
                alpha=0.944,
                k=0.889,
                delta=param,
            )
            for param, num_nodes in product(self.delta, self.n)
        ]
        self.cases += self.cases_delta

        # Lamda - With constraints
        self.lamda = np.append(
            np.round(np.linspace(0.01, 15, N_param), 3), 0.5
        )
        self.lamda = np.sort(self.lamda)
        self.cases_lamda = [
            Simulation(
                root=get_path(f'n_{num_nodes}_lamda_{param}'),
                job_info=dict(n_core=1, n_thread=1),
                base_command=base_cmd,
                n_epoch=self.n_epochs,
                epoch=self.epochs,
                n=num_nodes,
                t_max=num_nodes * 0.25,
                T=60,
                alpha=0.944,
                k=0.889,
                delta=21,
                lamda=param,
            )
            for param, num_nodes in product(self.lamda, self.n)
        ]
        self.cases += self.cases_lamda

        # Lamda - Without constraints
        self.lamda = np.append(
            np.round(np.linspace(0.01, 6, N_param), 3), 0.5
        )
        self.lamda = np.sort(self.lamda)
        self.cases_lamda_no_const = [
            Simulation(
                root=get_path(f'n_{num_nodes}_lamda_{param}_no_constr'),
                job_info=dict(n_core=1, n_thread=1),
                base_command=f'{base_cmd} --optim-dist',
                n_epoch=self.n_epochs,
                epoch=self.epochs,
                n=num_nodes,
                t_max=num_nodes * 0.8,
                T=60,
                alpha=0.944,
                k=0.889,
                delta=21,
                ignore_constr=None,
                lamda=param,
            )
            for param, num_nodes in product(self.lamda, self.n)
        ]
        self.cases += self.cases_lamda_no_const

    def run(self):
        self.make_output_dir()
        self._plot_parameter_tuning(
            param_name='T', param_vals=self.T, cases_var='n',
            cases_vals=self.n, sim_cases=self.cases_T
        )
        self._plot_parameter_tuning(
            param_name='alpha', param_vals=self.alpha, cases_var='n',
            cases_vals=self.n, sim_cases=self.cases_alpha
        )
        self._plot_parameter_tuning(
            param_name='k', param_vals=self.k, cases_var='n',
            cases_vals=self.n, sim_cases=self.cases_k
        )
        self._plot_parameter_tuning(
            param_name='delta', param_vals=self.delta, cases_var='n',
            cases_vals=self.n, sim_cases=self.cases_delta
        )
        self._plot_parameter_tuning(
            param_name='lamda', param_vals=self.lamda, cases_var='n',
            cases_vals=self.n, sim_cases=self.cases_lamda
        )
        self._plot_parameter_tuning(
            param_name='lamda', param_vals=self.lamda, cases_var='n',
            cases_vals=self.n, sim_cases=self.cases_lamda_no_const,
            ext='no_constr_'
        )
        # self._plot_cost_history(ext='epoch_1000_', epoch=self.epochs)
        # self._plot_runtimes(ext='epoch_1000_', epoch=self.epochs)


class SimpleSimulatedAnnealing(ComplexSimulatedAnnealingParamTuning):
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
        # SimpleSimulatedAnnealing,
        ComplexSimulatedAnnealingParamTuning
    ]
    automator = Automator(
        simulation_dir='output/simulated_annealing/automate',
        output_dir='output/simulated_annealing/automate/figures',
        all_problems=PROBLEMS
    )
    automator.run()
