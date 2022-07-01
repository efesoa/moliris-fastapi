from pydantic import BaseModel


class IrisBase(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


class Flower(IrisBase):
    sec_sepal_length: float
    sec_sepal_width: float
    sec_petal_length: float
    sec_petal_width: float


class KIris(IrisBase):
    k: int
