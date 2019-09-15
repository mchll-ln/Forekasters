from tensorflow.python.keras.models import load_model

from data import get_data
from model import f1_m

dim_x, dim_y = (300, 500), (300, 500)
classes = 4
batch_size = 16
epochs = 40

model = load_model("model.h5", custom_objects={'f1_m': f1_m})
x, y = get_data(dim_x, dim_y, year=2004)
model.evaluate(x, y, batch_size=batch_size)
