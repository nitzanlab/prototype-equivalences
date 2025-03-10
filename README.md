# Characterizing Nonlinear Dynamics via Smooth Prototype Equivalences
[Roy Friedman](https://friedmanroy.github.io/), [Noa Moriel](https://nomoriel.github.io/), [Matthew Ricci](https://www.matthew-ricci.net/), Guy Pelc, [Yair Weiss](https://www.cs.huji.ac.il/~yweiss/), [Mor Nitzan](https://www.nitzanlab.com/)

[![arXiv](https://img.shields.io/badge/arXiv-0000.00000-red.svg)](https://arxiv.org/abs/0000.00000)


**Abstract:** Characterizing dynamical systems from limited measurements could deepen our understanding of temporal processes in nature. However, the diversity of possible behaviors and the transient variability in systems with equivalent long-term dynamics make this task challenging.
  We address this by introducing _**smooth prototype equivalences (SPE)**_, a framework that fits a diffeomorphism using normalizing flows to distinct prototypes — simplified dynamical systems that define equivalence classes of behavior. 
  Given sparse, high-dimensional measurements (not necessarily a time-series), SPE can classify the long-term behavior  
 by comparing the deformation loss of multiple prototype dynamics. Furthermore, our approach enables estimation of the invariant sets of the observed dynamics through the learned mapping from prototype space to data space. Our method outperforms existing techniques in the classification of oscillatory systems and can efficiently identify invariant structures like limit cycles and fixed points in an equation-free manner, even when only a small, noisy subset of the phase space is observed.  Finally, we show how our method can be used for the detection of biological processes such as the cell cycle trajectory from high-dimensional single-cell gene expression data.

![](https://github.com/nitzanlab/prototype-equivalences/blob/main/assets/schematic.png)

---

## Requirements

This code is based on `python 3`. The main package requirements in this repository are:
```
torch>=2.0.1
tqdm
matplotlib
```
Full requirements can be found in the `requirements.txt` file. For easy installation, use the following `pip` command:
```
pip install -r requirements.txt
```

## Basic Usage

SPE allows for the characterization of observations from a vector field, the set $D=\{(x\_i,\dot{x}\_i)\}\_{i=1}^N$  comprised of positions ($x_i$) and velocities ($\dot{x}_i$). To do so, the data is matched to a prototype, which is a simple vector field governed by a known equation $\dot{y}=g(y)$. This matching is carried out using a diffeomorphism (an invertible and differentiable function) parameterized with a neural network, specifically a normalizing flow (NF).

This codebase has two main components: 
1. An implementation of NFs that are expressive but lightweight - the `Diffeo` class from [`NFDiffeo.py`](https://github.com/nitzanlab/prototype-equivalences/blob/main/NFDiffeo.py)
2. The fitting procedure to the prototypes - the `fit_prototype` function from [`fit_SPE.py`](https://github.com/nitzanlab/prototype-equivalences/blob/main/fit_SPE.py)

Using these, we showed that SPE can be used to estimate limit cycles directly from the vector field data, and also that it is possible to classify the observed data to pre-specified categories of behavior dictated by a set of prototypes.

An example usage of SPE for estimating limit cycles in 2D and higher-dimensional systems can be found in [`demo.ipynb`](https://github.com/nitzanlab/prototype-equivalences/blob/main/demo.ipynb). For classification, multiple prototypes need to be fitted. The wrapper function `fit_all_prototypes` from [`fit_SPE.py`](https://github.com/nitzanlab/prototype-equivalences/blob/main/fit_SPE.py) is designed to handle this natively. 

### Prototype Definition

To use SPE, a prototype has to be defined. In our experiments, we used limit-cycle simple oscillators (SO) as prototypes. These are governed by the following equations (in polar coordinates):
- $\dot{r}=r(a-r)^2$
- $\dot{\theta}=\omega$
  
where $a$ and $\omega$ are scalar parameters. When $a<0$, this system has a (single) node attractor at $x=y=0$, while $a>0$ has a limit cycle which is a circle with radius $\sqrt{a}$  about the origin. Positive $\omega$ implies counter-clockwise movement. Using this simple system as a prototype is optimal, as it allows us to learn a mapping from the observed data to some simple behavior. In all of our implementations, prototypes are considered as `Callable` objects. The SO prototype can be instantiated using the `get_oscillator` function from [`Hutils.py`](https://github.com/nitzanlab/prototype-equivalences/blob/main/Hutils.py).

## Simulated Systems

The notebook [`create_data.ipynb`](https://github.com/nitzanlab/prototype-equivalences/blob/main/create_data.ipynb) allows for the generation of the simulated systems in our experiments. The file [`systems.py`](https://github.com/nitzanlab/prototype-equivalences/blob/main/create_data.ipynb) includes implemented classes for all of the considered simulated systems with more functionalities useful for plotting. Much of these are adapted from [time-warp-attend](https://github.com/nitzanlab/time-warp-attend).

## Contact

Please be in touch if you have any questions! For contact, you can email Roy Friedman at roy.friedman@mail.huji.ac.il .