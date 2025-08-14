# USBuildingArchetypes

## Publications

|Publication|Title|Paper|Code|Summary/presentation|
|:-|:-|:-:|:-:|:-:|
|Proceedings of the 19th IBPSA Conference (Brisbane, Australia, Aug 24-27, 2025)|["Deriving high-fidelity residential building archetypes and typical usage patterns from national energy use surveys to enhance “initial guesses” for Urban Building Energy Model (UBEM) inputs"](https://doi.org/10.26868/25222708.2025.11336)|[Full paper PDF](papers/ibpsa2025/paper/11336_Li_BS2025.pdf)|[Notebooks](papers/ibpsa2025/workflow/)|[Conference presentation slides](papers/ibpsa2025/paper/Presentation.pdf)|

## Planned for feature releases

- [ ] Bayesian networks?

- [ ] In what form should the archetypes be?

- [ ] How should we explain the feature selection process for clustering?

- [X] Make separate and reproduceable workflows for clustering and for prediction.
- [X] Move utiliy functions to external `.py` file
- [X] Use k-fold CV for classifier performance evaluation and model selection.
- [X] Consider hyper-parameter search in classifier performance evaluation and model selection.
- [ ] Deploy the RECS-based archetype workflow to all Coppen climate zones in the United States
- [ ] A release of all readily useable building archetypes for the U.S., including guides to generalize/tweak parameters for usage in other countries and territories
- [ ] Deploy the archetype workflow for CBECS
- [ ] Normalize thermal loads EUI agaist HDD65 and CDD55 (?)
- [ ] Examine how catboost predicts based on incomplete features
- [ ] Mechanism to merge archetypes?
- [ ] Racing conditions: excol has equipment info, archetype has contradictionary equipment info




## Possible journal discussion points

First implement archetypes for all climate zones.
- [ ] How does building age say about the impact on archetype assignment, in general? Are there any regional differences?
- [ ] Which regions (climate zones) have more archetypes than others? What are the reasons?



## Search for the best fit in the table
When searching for the best fit, there are cases where a perfect match simply does not exist in the RECS response.

In this case we should search for the closest match or the closest matches where only a subset of all important features are matched. The selection for the "important features" could be the result of a feature importance ranking produced from the cluster exercise, (or not).

How is this connected to the "Bayesian Network" approach?

