from data import get_data
from model import get_model, f1_m

dim_x, dim_y = (300, 500), (300, 500)
classes = 4
batch_size = 16
epochs = 40

x, y = get_data(dim_x, dim_y)

model = get_model(classes=classes, input_shape=(200, 200, 16))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', f1_m])

model.fit(x, y, epochs=epochs, batch_size=batch_size)
model.save('model.h5')
