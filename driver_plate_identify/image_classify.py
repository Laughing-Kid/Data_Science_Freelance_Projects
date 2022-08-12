import joblib
from PIL import Image
from driver_plate_identify import ExifOrientationNormalize, MODEL_PATH


image_path = '../Driver_dataset/Abdullah_Anas/Abdullah Anas.jpg'

preprocess = ExifOrientationNormalize()
img = Image.open(image_path)
filename = img.filename
img = preprocess(img)
img = img.convert('RGB')

faces = joblib.load(MODEL_PATH)(img)
if faces:
  print('The face is of: ', faces[0].top_prediction.label.upper())
  print('The prediction precentage is: ', faces[0].top_prediction.confidence * 100)

if not faces:
    print('No faces found in this image.')