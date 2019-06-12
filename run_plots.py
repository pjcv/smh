import collections
import copy
import h5py
import itertools
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas
import pickle
import scipy.stats
import seaborn as sns
import subprocess
import tempfile

import jobs

iterations = 20000
num_runs = 20


class JsonConfigCommand(jobs.Task):

    def Run(self, params):
        cmd, json_config = self.GetCommandAndJson(params)
        with tempfile.NamedTemporaryFile(mode='wt') as tf:
            json.dump(json_config, tf)
            tf.flush()
            p = subprocess.Popen(
                [
                    cmd,
                    '--config=%s' % tf.name,
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE)
            stdout, stderr = p.communicate()
            print(stdout)
            print(stderr)
            if p.returncode != 0:
                raise Exception('process returned %d' % p.returncode)

    def GetCommandAndJson(self, params):
        raise NotImplementedError()


@jobs.registertask('dataset')
class Dataset(JsonConfigCommand):

    def Outputs(self, params):
        task_id = jobs.params_to_id(params)
        return ['output/dataset_%s.h5' % (task_id)]

    def Dependencies(self, params):
        return []

    def GetCommandAndJson(self, params):
        json_config = copy.copy(params)
        outputs = self.Outputs(params)
        json_config['output_filename'] = outputs[0]
        return 'build/run_generate_data', json_config


@jobs.registertask('approximation')
class Approximation(JsonConfigCommand):

    def Outputs(self, params):
        task_id = jobs.params_to_id(params)
        return ['output/approximation_%s.h5' % (task_id)]

    def Dependencies(self, params):
        return [
            jobs.Job('dataset', params['dataset']),
        ]

    def GetCommandAndJson(self, params):
        deps = self.Dependencies(params)
        outputs = self.Outputs(params)

        dataset_outputs = Dataset().Outputs(params['dataset'])

        json_config = {
            'dataset_filename': dataset_outputs[0],
            'model': params['model'],
            'output_filename': outputs[0],
        }
        return 'build/run_approximation', json_config


@jobs.registertask('sample')
class Sample(JsonConfigCommand):

    def Outputs(self, params):
        task_id = jobs.params_to_id(params)
        return ['output/samples_%s.h5' % (task_id)]

    def Dependencies(self, params):
        return [
            jobs.Job('dataset', params['dataset']),
            jobs.Job('approximation', {
                'dataset': params['dataset'],
                'model': params['model'],
            }),
        ]

    def GetCommandAndJson(self, params):
        if params['alg']['type'] == 'zigzag':
            cmd = 'build/run_lr_zigzag'
        else:
            cmd = 'build/run_sampler'

        # Get filename for dataset.
        dataset_outputs = Dataset().Outputs(params['dataset'])
        assert len(dataset_outputs) == 1
        dataset_filename = dataset_outputs[0]

        approximation_outputs = Approximation().Outputs({
            'dataset': params['dataset'],
            'model': params['model'],
        })

        self_outputs = self.Outputs(params)[0]

        json_config = copy.copy(params)
        del json_config['dataset']
        json_config['dataset_filename'] = dataset_filename

        json_config['initial_theta'] = {
            'type': 'hdf5',
            'file': approximation_outputs[0],
            'dataset': 'mode',
        }

        json_config['mode'] = {
            'type': 'hdf5',
            'file': approximation_outputs[0],
            'dataset': 'mode',
        }

        outputs = self.Outputs(params)
        json_config['output_filename'] = outputs[0]

        return cmd, json_config


@jobs.registertask('iact')
class Iact(JsonConfigCommand):

    def Outputs(self, params):
        task_id = jobs.params_to_id(params)
        return ['output/iact_%s.h5' % (task_id)]

    def Dependencies(self, params):
        return [
            jobs.Job('sample', params),
        ]

    def GetCommandAndJson(self, params):
        # Get filename for samples.
        samples_outputs = Sample().Outputs(params)
        assert len(samples_outputs) == 1

        self_outputs = self.Outputs(params)

        json_config = {
            'samples_filename': samples_outputs[0],
            'output_filename': self_outputs[0],
        }
        return 'build/run_compute_iact', json_config


@jobs.registertask('iact_multiple')
class IactMultiple(jobs.Task):

    def Outputs(self, params):
        task_id = jobs.params_to_id(params)
        return ['output/iacts_%s.pkl' % (task_id)]

    def Dependencies(self, params):
        deps = []
        for i in range(params['num_runs']):
            dep_params = copy.copy(params)
            del dep_params['num_runs']
            dep_params['seed'] = i
            deps.append(jobs.Job('iact', dep_params))
        return deps

    def Run(self, params):
        # Get filename for samples.
        deps = self.Dependencies(params)
        dep_outputs = [jobs.Outputs(j)[0] for j in deps]

        datas = []
        for output in dep_outputs:
            with h5py.File(output) as hf:
                data = {
                    'iact_iterations': np.array(hf['/iact_iterations'])[0],
                    'iact_likelihood_evaluations': np.array(hf['/iact_likelihood_evaluations'])[0],
                    'iact_seconds': np.array(hf['/iact_seconds'])[0],
                }
            datas.append(data)

        pickle.dump(
            {
                'data': datas,
                'params': params,
            },
            open(self.Outputs(params)[0], 'wb')
        )


current_palette = sns.color_palette('deep', 5)
palette = {
    'MH': current_palette[0],
    'SMH-1': current_palette[1],
    'SMH-2': current_palette[2],
    'FlyMC': current_palette[3],
    'Zig-Zag': current_palette[4],
}

markers = {
    'MH': '.',
    'SMH-1': 'D',
    'SMH-2': 'o',
    'FlyMC': 'v',
    'Zig-Zag': 'x',
}

linestyles = {
    'MH': '-',
    'SMH-1': '-',
    'SMH-2': '-',
    'FlyMC': '-',
    'Zig-Zag': '-',
}


def nice_alg_name(params):
    if params['alg']['type'] == 'fmh':
        label = 'SMH-%d' % params['alg']['params']['taylor_order']
    elif params['alg']['type'] == 'mh':
        label = 'MH'
    elif params['alg']['type'] == 'flymc':
        label = 'FlyMC'
    elif params['alg']['type'] == 'zigzag':
        label = 'Zig-Zag'
    else:
        label = params['alg']['type']
    return label


@jobs.registertask('likelihoods_per_iteration_plot')
class LikelihoodsPerIterationPlot(jobs.Task):

    def Outputs(self, params):
        return ['figs/likelihoods_per_iteration.pdf']

    def Dependencies(self, params):
        deps = []
        for lN, (alg_type, alg_params) in itertools.product(
                range(6, 18),
                [
                    ('mh', {}),
                    ('fmh', {'taylor_order': 1}),
                    ('fmh', {'taylor_order': 2}),
                ]):
            N = 2 ** lN
            params = {
                'dataset': {
                    'N': N,
                    'covariate_distribution': 'gaussian',
                    'theta': [1] * 10,
                    'seed': 0,
                    'noise': 'bernoulli',
                },
                'model': {
                    'type': 'lr',
                },
                'alg': {
                    'type': alg_type,
                    'params': alg_params,
                },
                'seed': 0,
                'iterations': iterations,
                'proposal': {
                    'type': 'random_walk',
                },
            }
            deps.append(jobs.Job('sample', params))

        return deps

    def Run(self, params):
        print('running LikelihoodsPerIterationPlot: %s' % jobs.params_to_string(params))

        # Get filenames for inputs.
        deps = self.Dependencies(params)

        df = pandas.DataFrame(columns=[
            'alg',
            'N',
            'likelihood_evaluations',
            'iterations',
        ])

        for dep in deps:
            dep_outputs = jobs.Outputs(dep)
            assert len(dep_outputs) == 1
            dep_output = dep_outputs[0]

            label = nice_alg_name(dep.params)

            with h5py.File(dep_output) as hf:
                df = df.append({
                    'alg': label,
                    'N': dep.params['dataset']['N'],
                    'iterations': np.array(hf['/samples']).shape[0],
                    'likelihood_evaluations': np.array(hf['/likelihood_evaluations'])[0],
                }, ignore_index=True)

        df['likelihoods_per_iteration'] = df['likelihood_evaluations'] / df['iterations']

        print(df)

        hue_order = [
            'MH',
            'SMH-1',
            'SMH-2',
        ]

        poster = False
        if poster:
            context = {
                'text.usetex': True,
                'text.latex.preamble': '\\RequirePackage{cmbright}\n\\RequirePackage[default]{lato}',
                'axes.facecolor': 'ffffff',
                'axes.edgecolor': '000000',
                'axes.labelcolor': 'ffffff',
                'xtick.color': 'ffffff',
                'ytick.color': 'ffffff',
                'legend.edgecolor': '000000',
                'legend.facecolor': 'ffffff',
                'text.color': '000000',
                'savefig.facecolor': '002147',
            }
        else:
            context = {}

        with plt.rc_context(context):
            fig = plt.figure(figsize=[6.4, 3.2])
            ax = sns.pointplot(
                x='N',
                y='likelihoods_per_iteration',
                data=df,
                hue='alg',
                hue_order=hue_order,
                markers=[markers[k] for k in hue_order],
                linestyles=[linestyles[k] for k in hue_order],
                palette=palette,
            )
            ax.set(ylabel=r'Likelihood evaluations per iteration')
            legend = ax.legend()
            legend.set_title('')
            ax.set(xlabel=r'$n$')
            ax.set(yscale='log')

            min_n = df['N'].min()
            max_n = df['N'].max()

            new_ticklabels = ax.xaxis.get_ticklabels()
            new_ticklabels[-2] = ''
            new_ticklabels[-4] = ''
            ax.xaxis.set_ticklabels(new_ticklabels)

            plt.savefig(self.Outputs(params)[0],
                        bbox_inches='tight',
                        pad_inches=0)


@jobs.registertask('ess_random_walk')
class EssRandomWalk(jobs.Task):

    def Outputs(self, params):
        return ['figs/ess_random_walk_model=%s_d=%d.pdf' % (
            params['model_type'],
            params['d'],
        )]

    def Dependencies(self, params):
        d = params['d']

        deps = []

        algs = [
            ('mh', {}),
            ('fmh', {'taylor_order': 1}),
            ('fmh', {'taylor_order': 2}),
            ('flymc', {'qdb': 0.001}),
        ]
        if params['model_type'] == 'lr':
            algs.append(('zigzag', {}))

        for lN, (alg_type, alg_params) in itertools.product(
                range(6, 18),
                algs):
            N = 2 ** lN
            dep_params = {
                'alg': {
                    'type': alg_type,
                    'params': alg_params,
                },
                'num_runs': num_runs,
                'iterations': iterations,
                'proposal': {
                    'type': 'random_walk',
                },
            }

            if params['model_type'] == 'lr':
                dep_params['dataset'] = {
                    'N': N,
                    'covariate_distribution': 'gaussian',
                    'theta': [1] * d,
                    'seed': 0,
                    'noise': 'bernoulli',
                }
                dep_params['model'] = {'type': 'lr'}
            elif params['model_type'] == 'rlr':
                dep_params['dataset'] = {
                    'N': N,
                    'covariate_distribution': 'gaussian',
                    'theta': [1] * d,
                    'seed': 0,
                    'noise': 'gaussian',
                }
                dep_params['model'] = {'type': 'rlr', 'params': {'nu': 4.0}}

            deps.append(jobs.Job('iact_multiple', dep_params))

        return deps

    def Run(self, params):
        print('running EssRandomWalk: %s' % jobs.params_to_string(params))

        # Get filenames for inputs.
        deps = self.Dependencies(params)

        df = pandas.DataFrame(columns=[
            'alg',
            'N',
            'ess_seconds',
        ])

        for dep in deps:
            dep_outputs = jobs.Outputs(dep)
            assert len(dep_outputs) == 1
            dep_output = dep_outputs[0]

            label = nice_alg_name(dep.params)

            pickle_data = pickle.load(open(dep_output, 'rb'))

            # TODO rename this 'run' and restructure the pickled data
            for run in pickle_data['data']:
                df = df.append({
                    'alg': label,
                    'N': dep.params['dataset']['N'],
                    'ess_seconds': 1.0 / run['iact_seconds'],
                }, ignore_index=True)

        print(df)

        hue_order = [
            'MH',
            'SMH-1',
            'SMH-2',
            'FlyMC',
        ]
        if params['model_type'] == 'lr':
            hue_order.append('Zig-Zag')

        plt.figure(figsize=[6.4, 3.2])
        ax = sns.pointplot(
            x='N',
            y='ess_seconds',
            data=df,
            hue='alg',
            hue_order=hue_order,
            markers=[markers[k] for k in hue_order],
            linestyles=[linestyles[k] for k in hue_order],
            palette=palette,
        )
        ax.set(ylabel=r'Effective sample size per second')
        legend = ax.legend()
        legend.set_title('')
        ax.set(xlabel=r'$n$')
        ax.set(yscale='log')

        min_n = df['N'].min()
        max_n = df['N'].max()

        new_ticklabels = ax.xaxis.get_ticklabels()
        new_ticklabels[-2] = ''
        new_ticklabels[-4] = ''
        ax.xaxis.set_ticklabels(new_ticklabels)

        plt.savefig(self.Outputs(params)[0],
                    bbox_inches='tight',
                    pad_inches=0)


@jobs.registertask('ess_pcn')
class EssPCN(jobs.Task):

    rhos = [0.9, 0.5, 0.0]

    def Outputs(self, params):
        return ['figs/ess_pcn_model=%s_d=%d.pdf' % (
            params['model_type'],
            params['d'],
        )]

    def Dependencies(self, params):
        d = params['d']
        deps = []

        algs = [
            ('mh', {}),
            ('fmh', {'taylor_order': 1}),
            ('fmh', {'taylor_order': 2}),
            ('flymc', {'qdb': 0.001}),
        ]

        for lN, (alg_type, alg_params), rho in itertools.product(
                range(6, 18),
                algs,
                self.rhos,
        ):
            N = 2 ** lN
            dep_params = {
                'alg': {
                    'type': alg_type,
                    'params': alg_params,
                },
                'num_runs': num_runs,
                'iterations': iterations,
                'proposal': {
                    'type': 'pcn',
                    'params': {
                        'rho': rho,
                    },
                },
            }

            if params['model_type'] == 'lr':
                dep_params['dataset'] = {
                    'N': N,
                    'covariate_distribution': 'gaussian',
                    'theta': [1] * d,
                    'seed': 0,
                    'noise': 'bernoulli',
                }
                dep_params['model'] = {'type': 'lr'}
            elif params['model_type'] == 'rlr':
                dep_params['dataset'] = {
                    'N': N,
                    'covariate_distribution': 'gaussian',
                    'theta': [1] * d,
                    'seed': 0,
                    'noise': 'gaussian',
                }
                dep_params['model'] = {'type': 'rlr', 'params': {'nu': 4.0}}

            deps.append(jobs.Job('iact_multiple', dep_params))

        return deps

    def Run(self, params):
        print('running EssPcn: %s' % jobs.params_to_string(params))

        # Get filenames for inputs.
        deps = self.Dependencies(params)

        df = pandas.DataFrame(columns=[
            'alg',
            'N',
            'rho',
            'ess_seconds',
        ])

        for dep in deps:
            dep_outputs = jobs.Outputs(dep)
            assert len(dep_outputs) == 1
            dep_output = dep_outputs[0]

            label = nice_alg_name(dep.params)

            pickle_data = pickle.load(open(dep_output, 'rb'))

            print('pickle_data:')
            print(pickle_data)

            # TODO rename this 'run' and restructure the pickled data
            for run in pickle_data['data']:
                df = df.append({
                    'alg': label,
                    'N': dep.params['dataset']['N'],
                    'rho': dep.params['proposal']['params']['rho'],
                    'ess_seconds': 1.0 / run['iact_seconds'],
                }, ignore_index=True)

        print(df)

        hue_order = [
            'MH',
            'SMH-1',
            'SMH-2',
            'FlyMC',
        ]

        plt.figure(figsize=[6.4, 3.2])
        fig, subplots = plt.subplots(1, len(self.rhos))
        for i in range(len(self.rhos)):
            ax = subplots[i]
            rho = self.rhos[i]
            sns.pointplot(
                ax=ax,
                x='N',
                y='ess_seconds',
                data=df[df['rho'] == rho],
                hue='alg',
                hue_order=hue_order,
                markers=[markers[k] for k in hue_order],
                linestyles=[linestyles[k] for k in hue_order],
                palette=palette,
            )
            if i == 0:
                legend = ax.legend()
                legend.set_title('')
                ax.set(ylabel='First moment ESS per second')
            else:
                ax.get_legend().remove()
                ax.set(ylabel='')
                ax.yaxis.set_ticklabels([])
                ax.yaxis.set_ticklabels([], minor=True)

            ax.set_title(r'$\rho = %0.1f$' % rho)
            ax.set(xlabel=r'$n$')
            ax.set(yscale='log')

            ticklabels = ax.xaxis.get_ticklabels()
            num_ticks = len(ticklabels)
            ax.xaxis.set_ticks([0, num_ticks - 1])
            ax.xaxis.set_ticklabels([ticklabels[0], ticklabels[-1]])

            for j, l in enumerate(ax.xaxis.get_ticklabels()):
                if j != 0 and j != len(ax.xaxis.get_ticklabels()) - 1:
                    l.set_visible(False)

        # Square up axes.
        min_y = np.inf
        max_y = -np.inf
        for ax in subplots:
            axis = ax.axis()
            min_y = min(min_y, axis[2])
            max_y = max(max_y, axis[3])
        for ax in subplots:
            axis = ax.axis()
            ax.axis([axis[0], axis[1], min_y, max_y])

        for i, ax in enumerate(subplots):
            if i != 0:
                ax.yaxis.set_ticklabels([])
                ax.yaxis.set_ticklabels([], minor=True)

        plt.savefig(self.Outputs(params)[0],
                    bbox_inches='tight',
                    pad_inches=0)


@jobs.registertask('acceptance_pcn')
class AcceptancePCN(jobs.Task):

    rhos = [0.9, 0.5, 0.0]

    def Outputs(self, params):
        return ['figs/acceptance_pcn_model=%s_d=%d.pdf' % (
            params['model_type'],
            params['d'],
        )]

    def Dependencies(self, params):
        d = params['d']

        deps = []

        algs = [
            ('mh', {}),
            ('fmh', {'taylor_order': 1}),
            ('fmh', {'taylor_order': 2}),
            ('flymc', {'qdb': 0.001}),
        ]

        for lN, (alg_type, alg_params), rho in itertools.product(
                range(6, 18),
                algs,
                self.rhos,
        ):
            N = 2 ** lN
            dep_params = {
                'alg': {
                    'type': alg_type,
                    'params': alg_params,
                },
                'seed': 0,
                'iterations': iterations,
                'proposal': {
                    'type': 'pcn',
                    'params': {
                        'rho': rho,
                    },
                },
            }

            if params['model_type'] == 'lr':
                dep_params['dataset'] = {
                    'N': N,
                    'covariate_distribution': 'gaussian',
                    'theta': [1] * d,
                    'seed': 0,
                    'noise': 'bernoulli',
                }
                dep_params['model'] = {'type': 'lr'}
            elif params['model_type'] == 'rlr':
                dep_params['dataset'] = {
                    'N': N,
                    'covariate_distribution': 'gaussian',
                    'theta': [1] * d,
                    'seed': 0,
                    'noise': 'gaussian',
                }
                dep_params['model'] = {'type': 'rlr', 'params': {'nu': 4.0}}

            deps.append(jobs.Job('sample', dep_params))

        return deps

    def Run(self, params):
        print('running AcceptancePcn: %s' % jobs.params_to_string(params))

        # Get filenames for inputs.
        deps = self.Dependencies(params)

        df = pandas.DataFrame(columns=[
            'alg',
            'N',
            'rho',
            'iact_seconds',
        ])

        for dep in deps:
            dep_outputs = jobs.Outputs(dep)
            assert len(dep_outputs) == 1
            dep_output = dep_outputs[0]

            with h5py.File(dep_output) as hf:
                samples = np.array(hf['/samples'])
                acceptances = sum(samples[1:] != samples[:-1])
                rate = acceptances / (len(samples) - 1)

                label = nice_alg_name(dep.params)

                df = df.append({
                    'alg': label,
                    'N': dep.params['dataset']['N'],
                    'rho': dep.params['proposal']['params']['rho'],
                    'acceptance_rate': rate,
                }, ignore_index=True)

        print(df)

        hue_order = [
            'MH',
            'SMH-1',
            'SMH-2',
            'FlyMC',
        ]

        plt.figure(figsize=[6.4, 3.2])
        fig, subplots = plt.subplots(1, len(self.rhos))
        for i in range(len(self.rhos)):
            ax = subplots[i]
            rho = self.rhos[i]
            sns.pointplot(
                ax=ax,
                x='N',
                y='acceptance_rate',
                data=df[df['rho'] == rho],
                hue='alg',
                hue_order=hue_order,
                markers=[markers[k] for k in hue_order],
                linestyles=[linestyles[k] for k in hue_order],
                palette=palette,
            )
            current_axes = ax.axis()
            ax.axis([current_axes[0], current_axes[1], 0.0, 1.0])
            if i == 0:
                legend = ax.legend()
                legend.set_title('')
            else:
                ax.get_legend().remove()
                ax.set(ylabel='')
                ax.yaxis.set_ticks([])
                ax.yaxis.set_ticklabels([])

            ticklabels = ax.xaxis.get_ticklabels()
            num_ticks = len(ticklabels)
            ax.xaxis.set_ticks([0, num_ticks-1])
            ax.xaxis.set_ticklabels([ticklabels[0], ticklabels[-1]])

            for i, l in enumerate(ax.xaxis.get_ticklabels()):
                if i != 0 and i != len(ax.xaxis.get_ticklabels()) - 1:
                    l.set_visible(False)

            ax.set(xlabel=r'$n$')
            ax.set(ylabel=i == 0 and 'Acceptance Rate' or '')
            ax.set_title(r'$\sqrt{\rho}=%0.1f$' % rho)

        plt.savefig(self.Outputs(params)[0],
                    bbox_inches='tight',
                    pad_inches=0)


@jobs.registertask('density_estimates')
class DensityEstimates(jobs.Task):

    def Outputs(self, params):
        return ['figs/density_estimates.pdf']

    def Dependencies(self, params):
        d = 2
        N = 16

        deps = []

        base_params = {
            'seed': 0,
            'iterations': 10000000,
            'dataset': {
                'N': N,
                'covariate_distribution': 'gaussian',
                'theta': [1] * d,
                'seed': 0,
                'noise': 'bernoulli',
            },
            'model': {
                'type': 'lr'
            },
        }

        # Zig-zag has no proposal so is separated here.
        zigzag_params = copy.copy(base_params)
        zigzag_params['iterations'] *= 50
        zigzag_params.update({'alg': {'type': 'zigzag', 'params': {}}})
        deps.append(jobs.Job('sample', zigzag_params))

        # All discrete-time algorithms.
        algs = [
            ('mh', {}),
            ('fmh', {'taylor_order': 1}),
            ('fmh', {'taylor_order': 2}),
            ('flymc', {'qdb': 0}),
            ('flymc', {'qdb': 0.001}),
            ('flymc', {'qdb': 0.1}),
            ('flymc', {'qdb': 1.0}),
        ]

        props = [
            ('random_walk', {}),
            ('pcn', {'rho': 0.5}),
        ]

        for (alg_type, alg_params), (proposal_type, proposal_params) in itertools.product(algs, props):
            dep_params = copy.copy(base_params)
            dep_params.update({
                'alg': {
                    'type': alg_type,
                    'params': alg_params,
                },
                'proposal': {
                    'type': proposal_type,
                    'params': proposal_params,
                },
            })
            deps.append(jobs.Job('sample', dep_params))

        deps.append(
            jobs.Job(
                'approximation',
                {
                    'dataset': {
                        'N': N,
                        'covariate_distribution': 'gaussian',
                        'theta': [1] * d,
                        'seed': 0,
                        'noise': 'bernoulli',
                    },
                    'model': {
                        'type': 'lr',
                    },
                }
            )
        )

        return deps

    def Run(self, params):
        # Get filenames for inputs.
        deps = self.Dependencies(params)

        fig = plt.figure(figsize=[20, 10])

        min_x_all = np.inf
        max_x_all = -np.inf

        for dep in (dep for dep in deps if dep.task == 'sample'):
            dep_outputs = jobs.Outputs(dep)
            assert len(dep_outputs) == 1
            dep_output = dep_outputs[0]

            with h5py.File(dep_output) as hf:
                samples = np.array(hf['/samples'])
                print(dep.params)
                print(len(samples))
                print(samples)

                label = nice_alg_name(dep.params)

                label += str(dep.params['alg']['params'])
                if 'proposal' in dep.params:
                    label += str(dep.params['proposal'])

                kde = scipy.stats.gaussian_kde(samples[len(samples) // 2:])

                min_x = np.min(samples)
                max_x = np.max(samples)
                xs = np.linspace(min_x, max_x, num=100, endpoint=True)

                plt.plot(xs, kde(xs), label=label)

                min_x_all = min(min_x_all, min_x)
                max_x_all = max(max_x_all, max_x)

        for dep in (dep for dep in deps if dep.task == 'approximation'):
            dep_outputs = jobs.Outputs(dep)
            assert len(dep_outputs) == 1
            dep_output = dep_outputs[0]
            with h5py.File(dep_output) as hf:
                mode = np.array(hf['/mode'])
                covariance = np.array(hf['/covariance'])

                xs = np.linspace(min_x_all, max_x_all, num=100, endpoint=True)
                plt.plot(xs,
                         scipy.stats.norm.pdf(xs, loc=mode[0], scale=np.sqrt(covariance[0, 0])),
                         '--',
                         linewidth=5,
                         alpha=0.5,
                         label='approximation')

        plt.legend()
        plt.savefig(self.Outputs(params)[0],
                    bbox_inches='tight',
                    pad_inches=0)


@jobs.registertask('histogram')
class Histogram(jobs.Task):
    '''Plot sample histogram vs approximating density.'''

    def Outputs(self, params):
        return ['figs/histogram_d=%d_N=%d.pdf' % (params['d'], params['N'])]

    def Dependencies(self, params):
        approx_params = {
            'dataset': {
                'N': params['N'],
                'covariate_distribution': 'gaussian',
                'theta': [1] * params['d'],
                'seed': 0,
                'noise': 'bernoulli',
            },
            'model': {
                'type': 'lr'
            },
        }
        sample_params = copy.copy(approx_params)
        sample_params.update({
            'seed': 0,
            'iterations': 1000000,
            'alg': {
                'type': 'fmh',
                'params': {'taylor_order': 2},
            },
            'proposal': {
                'type': 'random_walk',
            },
        })
        return [
            jobs.Job('approximation', approx_params),
            jobs.Job('sample', sample_params),
        ]

    def Run(self, params):
        # Get filenames for inputs.
        deps = self.Dependencies(params)

        fig = plt.figure(figsize=[20, 10])

        for dep in (dep for dep in deps if dep.task == 'sample'):
            dep_outputs = jobs.Outputs(dep)
            assert len(dep_outputs) == 1
            dep_output = dep_outputs[0]

            with h5py.File(dep_output) as hf:
                samples = np.array(hf['/samples'])
                min_x = np.min(samples)
                max_x = np.max(samples)
                plt.hist(samples, bins=100, density=True)

        for dep in (dep for dep in deps if dep.task == 'approximation'):
            dep_outputs = jobs.Outputs(dep)
            assert len(dep_outputs) == 1
            dep_output = dep_outputs[0]
            with h5py.File(dep_output) as hf:
                mode = np.array(hf['/mode'])
                covariance = np.array(hf['/covariance'])

                xs = np.linspace(min_x, max_x, num=100, endpoint=True)
                plt.plot(xs,
                         scipy.stats.norm.pdf(xs, loc=mode[0], scale=np.sqrt(covariance[0, 0])),
                         label='approximation')

        plt.legend()
        plt.savefig(self.Outputs(params)[0],
                    bbox_inches='tight',
                    pad_inches=0)


if __name__ == '__main__':
    for dirname in ['figs', 'output']:
        if not os.path.exists(dirname):
            os.mkdir(dirname)

    root_jobs = []
    root_jobs.append(
        jobs.Job('likelihoods_per_iteration_plot', {})
    )
    root_jobs.append(
        jobs.Job('ess_random_walk', {
            'model_type': 'lr',
            'd': 10,
        })
    )
    root_jobs.append(
        jobs.Job('ess_pcn', {
            'model_type': 'lr',
            'd': 10,
        })
    )
    root_jobs.append(
        jobs.Job('acceptance_pcn', {
            'model_type': 'lr',
            'd': 10,
        })
    )
    root_jobs.append(
        jobs.Job('density_estimates', {})
    )
    root_jobs += [
        jobs.Job('histogram', {'d': 10, 'N': 2048}),
        jobs.Job('histogram', {'d': 10, 'N': 8192}),
    ]
    jobs.RunJobs(root_jobs)
