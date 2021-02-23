# Distributed Momentum for Byzantine-resilient Stochastic Gradient Descent

Authors<sup>1</sup>: El-Mahdi El-Mhamdi, Rachid Guerraoui, Sébastien Rouault

This is the code used in the experiments of [Distributed Momentum for Byzantine-resilient Stochastic Gradient Descent](https://openreview.net/forum?id=H8UHdhWG6A3),
co-authored by El-Mahdi El-Mhamdi, Rachid Guerraoui and Sébastien Rouault.

<sup>1</sup><sub>alphabetical order, as for all the papers from the _Distributed Computing Laboratory_ (DCL) of EPFL.</sub>

## Reproducing the results (see [the paper](https://openreview.net/pdf?id=H8UHdhWG6A3))

### Software dependencies

Python 3.7.3 has been used to run our scripts.

Besides the standard libraries associated with Python 3.7.3, our scripts have been tested with<sup>2</sup>:

| Library     | Version    |
| ----------- | ---------- |
| numpy       | 1.19.1     |
| torch       | 1.6.0      |
| torchvision | 0.7.0      |
| pandas      | 1.1.0      |
| matplotlib  | 3.0.2      |
| PIL         | 7.2.0      |
| requests    | 2.21.0     |
| urllib3     | 1.24.1     |
| chardet     | 3.0.4      |
| certifi     | 2018.08.24 |
| idna        | 2.6        |
| six         | 1.15.0     |
| pytz        | 2020.1     |
| dateutil    | 2.6.1      |
| pyparsing   | 2.2.0      |
| cycler      | 0.10.0     |
| kiwisolver  | 1.0.1      |
| cffi        | 1.13.2     |

<sup>2</sup><sub>this list was automatically generated (see `get_loaded_dependencies()` in `tools/misc.py`).
Some dependencies depend on others, and some dependencies are optional (e.g., only used to process the results and produce the plots).</sub>

We list below the OS on which our scripts have been tested:
* Debian 10 (GNU/Linux 4.19.171-2 x86_64)
* Ubuntu 18.04.5 LTS (GNU/Linux 4.15.0-128-generic x86_64)

### Hardware dependencies

Although our experiments are time-agnostic, we list below the hardware components used:
* 1 Intel(R) Core(TM) i7-8700K CPU @ 3.70GHz
* 2 Nvidia GeForce GTX 1080 Ti
* 64 GB of RAM

### Command

Our results are reproducible in one command.
In the root directory of the ZIP file:
```
$ python3 reproduce.py
```
On our hardware, reproducing the results takes a bit less than a week.

Please be aware this script will require non-negligible disk space: 2.1 GB of run data, and 132 MB of graphs.

Depending on the hardware, instructing the script to launch several runs per available GPU may reduce the total runtime.
For instance, to push up to 4 concurrent runs per GPU:
```
$ python3 reproduce.py --supercharge 4
```

### Citation

If you are using algorithms, results or code from this paper, you can include in your bibliography:
```
@inproceedings{distributed-momentum,
  author    = {El-Mahdi El-Mhamdi and
               Rachid Guerraoui and
               S\'{e}bastien Rouault},
  title     = {Distributed Momentum for Byzantine-resilient Stochastic Gradient Descent},
  booktitle = {9th International Conference on Learning Representations, {ICLR} 2021,
               Vienna, Austria, May 4–8, 2021},
  publisher = {OpenReview.net},
  year      = {2021},
  url       = {https://openreview.net/forum?id=H8UHdhWG6A3},
}
```

## Capabilities

`attack.py` can be used to simulate reproducible<sup>3</sup> distributed stochastic gradient descent sessions,
with several customizations (besides number of workers, batch size, learning rate evolution, choice of loss/regularization, etc):
* which gradient aggregation rule (GAR) is used instead of the standard average
  * GAR-specific additional parameters can be specified
* which attack is followed by the adversary
  * attack-specific additional parameters can be specified
* momentum applied at 3 different positions:
  * _at update_ (called _at server_ in the paper), the standard
  * _at the workers_ (see the paper)
  * _at the server_, a variant of _at the workers_ where the server keeps the momentum
* additional features:
  * gradient clipping (upper bound on the norm of the _sampled_, correct gradients)
  * multiple local steps (instead of one) between communication
  * checkpointing for training continuation<sup>4</sup>

<sup>3</sup><sub>reasonable efforts to remove sources of non-determinism have been made, please see [PyTorch's documentation](https://pytorch.org/docs/stable/notes/randomness.html) for further details.</sub>\
<sup>4</sup><sub>reproducibility is currently not guaranteed when continuing experiments (PRNG/dataset loader states are not saved in checkpoints).</sub>

When measurement is enabled (i.e., `--result-directory` has been specified), 17 metrics are captured for each step:
* variances between gradients
  * _sampled_ for the variance between the sampled, correct gradients
  * _honest_ for the variance between the correct gradients as received by the server,
    which can be different than _sampled_ when momentum is computed at the workers
  * _attack_ for the variance between the adversarial gradients (usually 0 since the Byzantine workers usually send the same gradient)
* average norms of gradients
  * _sampled_ for the average norm of the sampled, correct gradients (and after optional clipping)
  * _honest_ for the average norm of the correct gradients as received by the server,
    which can be different than _sampled_ when momentum is computed at the workers
  * _attack_ for the average norm of the adversarial gradients
  * _defense_ for the norm of the gradient computed by the GAR
* maximum coordinates of average gradients
  * ditto as for average norms (i.e.: _sampled_, _honest_, _attack_, _defense_)
* cosine angles between average gradients (for each of the 6 possible pairs)
  * _sampled-honest_, _sampled-attack_, _sampled-defense_
  * _honest-attack_, _honest-defense_
  * _attack-defense_

and 5 additional metrics:
* cumulative (until the current step) number of training points collectively consumed by the correct workers
* average loss observed by the correct worker
* _l<sub>2</sub>_-norm of the parameters from the initial parameters
* cosine angle between the average of the sampled, clipped, correct gradients and the one of the previous step
* the _composite curvature_ (i.e. the quantity _s<sub>t</sub>_ in the paper) using `--nb-for-study-past` past gradients for computation

The result directory will contain:
* the output metrics for each step formatted in mere CSV, with a header starting with character "#"
* human-readable and machine-readable (JSON) information on the run made
* standard and error outputs of `attack.py`

`study.py` can help process these result directories.
This tool can compute additional metrics (e.g., for which steps the theoretical requirements on the variance-norm ratio was satisfied),
and plot graphs.

### Usage

See `reproduce.py` for relevant uses of `attack.py`, and `study.py` (which is to be used as a library).

Otherwise the command-line help (`python3 attack.py --help`) can get you a glimpse of how to use the tool.

For setup testing purpose, just run `python3 attack.py`.
The standard output will contain information on the running defaults.

### Extensibility

It should be easy to add new GARs.
In directory `aggregators`:
* copy-paste `template.py` with a new name
* fill in the `aggregate`, `check` and (optionally) the `upper_bound` functions;\
  the interface of these functions is specified in the docstring of `__init__.py`
* use your new GAR with the `--gar` command-line argument of `attack.py`;\
  arbitrary, additional keyword-arguments can be passed with `--gar-args`

It should be easy to add new attacks.
In directory `attacks`:
* copy-paste `template.py` with a new name
* fill in the `attack` and `check` functions;\
  the interface of these functions is specified in the docstring of `__init__.py`
* use your new attack with the `--attack` command-line argument of `attack.py`;\
  arbitrary, additional keyword-arguments can be passed with `--attack-args`

Following the same principle, new datasets can be added in `experiments/datasets`.
