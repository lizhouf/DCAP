# Data-driven and Circulating Archival Processing (DACP)
This repository corresponds to the article [Using Machine Learning to Enhance Archival Processing of Social Media Archives](https://dl.acm.org/doi/abs/10.1145/3547146) and provides the code structure for using the Generative Adversarial Network (GAN)-inspired DCAP Method, which ingestes and aggregates within the COVID-19 Hate Speech Twitter Archive ([CHSTA](https://github.com/lizhouf/CHSTA)) .

To cite our work, please use either of the the following ways:

* BibTex:

```bibtex
@article{10.1145/3547146,
author = {Fan, Lizhou and Yin, Zhanyuan and Yu, Huizi and Gilliland, Anne J.},
title = {Using Machine Learning to Enhance Archival Processing of Social Media Archives},
year = {2021},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
issn = {1556-4673},
url = {https://doi.org/10.1145/3547146},
doi = {10.1145/3547146},
abstract = {This paper reports on a study using machine learning to identify incidences and shifting dynamics of hate speech in social media archives. To better cope with the archival processing need for such large scale and fast evolving archives, we propose the Data-driven and Circulating Archival Processing (DCAP) method. As a proof-of-concept, our study focuses on an English language Twitter archive relating to COVID-19: tweets were repeatedly scraped between February and June 2020, ingested and aggregated within the COVID-19 Hate Speech Twitter Archive (CHSTA) and analyzed for hate speech using the Generative Adversarial Network (GAN)-inspired DCAP Method. Outcomes suggest that it is possible to use machine learning and data analytics to surface and substantiate trends from CHSTA and similar social media archives that could provide immediately useful knowledge for crisis response, in controversial situations, or for public policy development, as well as for subsequent historical analysis. The approach shows potential for integrating multiple aspects of the archival workflow, and supporting automatic iterative redescription and reappraisal activities in ways that make them more accountable and more rapidly responsive to changing societal interests and unfolding developments.},
note = {Just Accepted},
journal = {J. Comput. Cult. Herit.},
month = {oct},
keywords = {Hate Speech, Machine Learning, COVID-19, Archival Processing, Generative Adversarial Network}
}
```

* ACM Ref:

> Lizhou Fan, Zhanyuan Yin, Huizi Yu, and Anne J. Gilliland. 2021. Using Machine Learning to Enhance Archival Processing of Social Media Archives. J. Comput. Cult. Herit. Just Accepted (October 2021). https://doi.org/10.1145/3547146 

## Implementation

The main.py shows the archival processing of the [CHSTA](https://github.com/lizhouf/CHSTA). All other files include utilities.
