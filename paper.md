---
title: 'The Empirical Sample Complexity Estimator: a data analysis workflow for assessing the effect of training sample size on machine learning performance'
tags:
  - machine-learning
  - python
authors:
  - name: Marc-Andre Schulz
    orcid: 
    affiliation: "1, 2"
  - name: Alexander Koch
    orcid: 
    affiliation: 1
  - name: Kerstin Ritter
    orcid: 
    affiliation: "1, 2"
affiliations:
 - name: Charité – Universitätsmedizin Berlin (corporate member of Freie Universität Berlin, Humboldt-Universität zu Berlin, and Berlin Institute of Health), Department of Psychiatry and Psychotherapy, Berlin, Germany
   index: 1
 - name: Bernstein Center for Computational Neuroscience, Berlin, Germany
   index: 2
date: 18 October 2023
bibliography: paper.bib
---

# Summary

We provide a Snakemake data analysis workflow designed to examine the performance of machine learning models as sample size increases. The workflow enables comparison of scaling behaviour across different models, feature sets, and target variables.

# Statement of Need

The Empirical Sample Complexity Estimator (ESCE) is designed to meet the need for tools that can analyze the scaling behaviour of machine learning models with increasing training sample size, especially in fields like biomedicine where data aggregation and labeling pose significant challenges.

ESCE offers a comprehensive data analysis workflow capable of handling multiple feature sets, target variables, and covariates-of-no-interest. It integrates data cleaning functionalities, automates the creation of train/validation/test splits across different sample sizes, and benchmarks various machine learning models with nested hyperparameter optimization. 

A distinguishing feature of ESCE is its capacity to use curve fitting techniques to extrapolate machine learning models' scaling behaviour to larger unseen sample sizes. This enables researchers to predict model performance beyond existing datasets and compare scaling behaviour across different models, feature sets, and target variables.

# Research Applications

The use cases of ESCE span a range of research areas that utilize machine learning models and deal with large, complex datasets. An illustrative example of ESCE's application is found in the study "Performance reserves in brain-imaging-based phenotype prediction" [@schulz2022]. This research leveraged ESCE to investigate performace ceilings of machine learning models operating on brain imaging data.

# Acknowledgements

The project was funded by the DFG (414984028/CRC-1404 FONDA).

# References