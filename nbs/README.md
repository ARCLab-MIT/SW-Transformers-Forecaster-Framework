# Functions Implementation Contents 🧩

This folder contains all the secondary implementations of the repository, including logic for plotting, preprocessing pipelines, custom losses, metrics, and more. In these notebooks, we provide explanations of the implementation details and also test the functions. Below is a description of the main notebooks:

- [`losses.ipynb`](/nbs/losses.ipynb): This notebook explains and implements the logic behind the custom loss functions used in the repository. Each loss function is explained, and a `LossFactory()` class is included at the end to manage the creation of loss functions. Some custom loss functions include weighted losses or losses that are not yet implemented in `fastai` but are relevant to this project.
- [`metrics.ipynb`](/nbs/metrics.ipynb): Here, metrics are explained, classified, and implemented. These metrics are used to validate models and guide code optimization. Similar to the losses notebook, it contains several factory classes for easy retrieval of desired metrics. It also implements metrics used to track the training process when applying weighted loss functions.
- [`preprocessing.ipynb`](/nbs/preprocessing.ipynb): This notebook implements the two preprocessing pipeline classes used during the code implementation.
- [`plots.ipynb`](/nbs/plots.ipynb): This notebook consolidates all the plotting logic, which is reused in various notebooks across the repository.
- [`models.ipynb`](/nbs/models.ipynb): This notebook implements additional models, such as `PersistenceModel()`, which serves as a persistence forecasting baseline.
- [`utils.ipynb`](/nbs/utils.ipynb): This notebook gathers miscellaneous functions that don’t fit into a specific category but are frequently reused across different notebooks, enhancing reusability.

> [!IMPORTANT]
> We use `nbdev` to export these notebooks into the `swdf` folder, where they can be imported and executed as `.py` files from other parts of the repository.
