#Grab a digit image from any website and make a prediction
from PIL import Image
import requests
from io import BytesIO
response = requests.get("https://vignette.wikia.nocookie.net/phobia/images/f/fe/7.jpg/revision/latest?cb=20170121103340")
img = Image.open(BytesIO(response.content)).convert("L")
plt.imshow(img)
plt.show()

img = img.resize((28,28))
im2arr = np.array(img)
im2arr = im2arr.reshape(1,784)
model = load_model("model.h5")
predictions = model.predict(im2arr)
print(predictions)
print(np.argmax(predictions, axis=1))
