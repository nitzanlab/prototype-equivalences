# Characterizing Nonlinear Dynamics via Smooth Prototype Equivalences
[Roy Friedman](https://friedmanroy.github.io/), Noa Moriel, [Matthew Ricci](https://www.matthew-ricci.net/), Guy Pelc, [Yair Weiss](https://www.cs.huji.ac.il/~yweiss/), [Mor Nitzan](https://www.nitzanlab.com/)

[![arXiv](https://img.shields.io/badge/arXiv-2409.07940-red.svg)](https://arxiv.org/abs/2409.07940)


**Abstract:** Characterizing dynamical systems given limited measurements could further our understanding of many processes in nature which are inherently temporal. However, this task is challenging, especially due to transient variability in systems with equivalent long-term dynamics and the breadth of possible behaviors in dynamics systems.  We address this by introducing _**smooth prototype equivalences (SPE)**_, a framework that fits a diffeomorphism using normalizing flows to distinct prototypes — simplified dynamical systems that define equivalence classes of behavior. SPE enables classification by comparing the deformation loss of the observed sparse, high-dimensional measurements to the prototype dynamics. Furthermore, our approach enables estimation of the invariant sets of the observed dynamics through the learned mapping from prototype space to data space. Our method outperforms existing techniques in the classification of oscillatory systems and can efficiently identify invariant structures like limit cycles and fixed points in an equation-free manner, even when only a small, noisy subset of the phase space is observed.  Finally, we show how our method can be used for the detection of biological processes like the cell cycle trajectory from high-dimensional single-cell gene expression data.

![](https://github.com/nitzanlab/prototype-equivalences/blob/main/assets/schematic.png)

---

