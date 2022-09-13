## Random strategies

::: energizer.query_strategies.RandomStrategy
    options:
        show_root_heading: true


## Uncertainty-based query strategies

Uncertainty-based query strategies select instance with high aleatoric uncertainty
or epistemic uncertainty. Aleatoric uncertainty refers to the uncertainty
in data due the data generation processes (sometimes called irreducible uncertainty).
Epistemic uncertainty comes from the modeling/learning process and is caused by a
lack of knowledge.

::: energizer.query_strategies.LeastConfidenceStrategy
    options:
        show_root_heading: true

::: energizer.query_strategies.MarginStrategy
    options:
        show_root_heading: true

::: energizer.query_strategies.EntropyStrategy
    options:
        show_root_heading: true



---

## Base classes
::: energizer.query_strategies.base
