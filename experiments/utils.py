from typing import Dict

from src.datasets import Communities, Adult, Polynomial, NonLinear, Dataset
from src.hgr import HGR, KernelBasedHGR, DensityHGR, ChiSquare, AdversarialHGR

DATASETS: Dict[str, Dataset] = dict(
    communities=Communities(continuous=True),
    communities_continuous=Communities(continuous=True),
    communities_categorical=Communities(continuous=False),
    adult=Adult(continuous=True),
    adult_continuous=Adult(continuous=True),
    adult_categorical=Adult(continuous=False),
    linear=Polynomial(degree_x=1, degree_y=1),
    x_squared=Polynomial(degree_x=2, degree_y=1),
    x_cubic=Polynomial(degree_x=3, degree_y=1),
    y_squared=Polynomial(degree_x=1, degree_y=2),
    y_cubic=Polynomial(degree_x=1, degree_y=3),
    circle=Polynomial(degree_x=2, degree_y=2),
    sign=NonLinear(name='sign'),
    relu=NonLinear(name='relu'),
    sin=NonLinear(name='sin'),
    tanh=NonLinear(name='tanh')
)
"""Dictionary of dataset aliases."""

METRICS: Dict[str, HGR] = dict(
    pears=KernelBasedHGR(degree_a=1, degree_b=1),
    kb=KernelBasedHGR(degree_a=5, degree_b=5),
    nn=AdversarialHGR(),
    kde=DensityHGR(),
    chi2=ChiSquare()
)
"""Dictionary of HGR metrics aliases."""
