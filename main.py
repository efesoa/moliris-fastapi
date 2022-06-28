from io import BytesIO

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from starlette.responses import HTMLResponse

import k_nearest_neighbor
import similarity_metric
import numpy as np

app = FastAPI(
    title="Moliris",
    description="""
    Moliris APIs
    """
)

origins = [
    "http://localhost:3000",
    "localhost:3000"
]


app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


def dataset():
    data = pd.read_csv('iris.data', header=None,
                       names=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'specie'])
    return data


@app.get("/", response_class=HTMLResponse)
def index():
    return """
        <!Doctype html>
        <html>
            <body>
                <h1>Welcome to Moliris API</h1>
                <div class="btn-group">
                    <a href="/docs"><button>SwaggerUI</button></a>
                    <a href="/redoc"><button>Redoc</button></a>
                </div>
            </body>
        </html>
    """


@app.get("/dataset")
async def default_dataset():
    data = pd.read_csv('iris.data', header=None,
                       names=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'specie'])
    return data.to_dict(orient='records')


@app.post("/upload")
async def upload_dataset(file: UploadFile = File(...)):
    contents = await file.read()
    buffer = BytesIO(contents)
    data = pd.read_csv(buffer, header=None,
                       names=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'specie'])
    buffer.close()
    return data.to_dict(orient='records')
    # return {"filename": file.filename}


@app.post("/measurements")
async def input_data(sepal_length: float, sepal_width: float, petal_length: float,
                     petal_width: float, sec_sepal_length: float, sec_sepal_width: float,
                     sec_petal_length: float, sec_petal_width: float):
    sl1: float = sepal_length
    sw1: float = sepal_width
    pl1: float = petal_length
    pw1: float = petal_width
    sl2: float = sec_sepal_length
    sw2: float = sec_sepal_width
    pl2: float = sec_petal_length
    pw2: float = sec_petal_width
    obj = ([sl1, sw1, pl1, pw1], [sl2, sw2, pl2, pw2])
    obj1 = obj[0]
    obj1_d = pd.DataFrame(obj1)
    obj2 = obj[1]
    obj2_d = pd.DataFrame(obj2)
    predicting_object1, n = k_nearest_neighbor.knn(dataset(), obj1_d, 1)
    predicting_object2, n = k_nearest_neighbor.knn(dataset(), obj2_d, 1)
    cal_manhattan = similarity_metric.manhattan(obj[0], obj[1])
    cal_euclidean = similarity_metric.euclidean_distance(obj[0], obj[1], 4)
    cal_cosine = similarity_metric.cosine_similarity(obj[0], obj[1])
    return obj, predicting_object1, predicting_object2, cal_cosine, cal_euclidean, cal_manhattan


@app.post("/similar_objects")
async def input_data(sepal_length: float, sepal_width: float, petal_length: float,
                     petal_width: float, k: int):
    sl1: float = sepal_length
    sw1: float = sepal_width
    pl1: float = petal_length
    pw1: float = petal_width
    number_of_neighbors: int = k
    obj = [sl1, sw1, pl1, pw1]
    obj1_d = pd.DataFrame(obj)
    predicting_object_knn, nn = k_nearest_neighbor.knn(dataset(), obj1_d, number_of_neighbors)
    data = dataset()
    x = data.iloc[:, 0:4]
    y = data.iloc[:, 4]
    values = np.array(x)
    specieNames = np.array(y)
    a = []
    b = []
    for n in nn:
        a.append(specieNames[n])
        for l in values[n - 1]:
            b.append(l)
    return a, b, predicting_object_knn
