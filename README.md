# CropGuard 🌱

Final assignment for the Building AI course

## Summary

CropGuard is a CNN-based image classifier that lets smallholder farmers photograph a diseased leaf with a phone and instantly receive a diagnosis and treatment suggestion, closing the gap between symptom onset and expert consultation. Building AI course project.


## Background

Crop diseases destroy a significant share of global harvests every year, and smallholder farmers — who produce a large portion of the world's food — are hit hardest, because they rarely have quick access to a plant pathologist or agricultural extension worker. By the time a farmer notices something is wrong and travels to ask for advice, the disease has often already spread across the field.

* Diseases are frequently misdiagnosed by farmers relying on guesswork or word-of-mouth, leading to the wrong (or no) treatment
* Professional agronomists are scarce in rural areas and expensive to reach
* Delayed diagnosis compounds yield loss, since many fungal and bacterial infections spread quickly once established
* Farmers often can't tell whether a spot on a leaf is a minor cosmetic issue or an early sign of a disease that will wipe out the crop

My personal motivation comes from documentation work I did on rural revitalization for a state-owned energy company, where our project supported a small agricultural village. Talking with local farmers there, I kept hearing the same story: by the time someone with the right expertise looked at a sick plant, it was often too late to save that season's yield. A tool that gives an instant first opinion — even an imperfect one — could buy farmers the time they need to act.


## How is it used?

A farmer opens a simple mobile app in the field, points their phone camera at a leaf that looks unhealthy, and takes a photo. The photo is fed into a CNN model (running either on-device or on a lightweight server), which returns:

1. The most likely disease (or "healthy") with a confidence score
2. A short, plain-language description of the disease
3. A basic first-response recommendation (e.g., remove affected leaves, apply a specific class of fungicide, or "consult an expert" if confidence is low)

**Context of use:** outdoors, in the field, often with poor or no internet connectivity — so the model needs to be small enough to run offline on a low-end smartphone, syncing results and collecting feedback whenever a connection becomes available.

**Users:** smallholder farmers and local agricultural extension workers, most of whom are not technical and may have limited literacy — so the interface should lean on icons, photos, and short audio explanations rather than dense text.

**Important consideration:** the tool is explicitly a *first opinion*, not a replacement for an agronomist. Low-confidence predictions should always be flagged as "uncertain — please seek expert advice" rather than forcing a guess.


## Data sources and AI methods

**Data:** The [PlantVillage dataset](https://www.kaggle.com/datasets/emmarex/plantdisease) is a widely-used open dataset with over 50,000 labeled leaf images spanning dozens of diseases across more than a dozen crops (tomato, potato, corn, apple, grape, and more). It's a good starting point for training and for building a working demo, though a real deployment would need to be supplemented with locally-collected images, since PlantVillage photos are taken under fairly controlled lab conditions and a model trained only on them tends to struggle with messy real-field lighting, backgrounds, and camera angles.

**AI methods:**
* **Convolutional Neural Network (CNN)** for image classification — this is the natural fit for the task, since CNNs are built to pick up on spatial patterns like leaf spots, discoloration, and texture changes.
* **Transfer learning**, starting from a small pretrained backbone (e.g., MobileNetV2) rather than training from scratch — this dramatically cuts down the amount of labeled data and compute needed, and MobileNet-style architectures are specifically designed to run efficiently on phones.
* **Data augmentation** (random rotation, brightness/contrast jitter, flips) to make the model more robust to the variety of lighting and angles a farmer's photo will actually have, compared to the clean lab photos in the training set.

Here's a simplified sketch of what the core model and training loop look like:

```python
import tensorflow as tf
from tensorflow.keras import layers, models

IMG_SIZE = (160, 160)
NUM_CLASSES = 38  # e.g. number of crop/disease combinations in PlantVillage

# Load a pretrained backbone and freeze its weights
base_model = tf.keras.applications.MobileNetV2(
    input_shape=IMG_SIZE + (3,),
    include_top=False,
    weights="imagenet"
)
base_model.trainable = False

# Add a small classification head on top
model = models.Sequential([
    layers.Rescaling(1. / 127.5, offset=-1, input_shape=IMG_SIZE + (3,)),
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.3),
    layers.Dense(NUM_CLASSES, activation="softmax"),
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=10,
)

# Convert to TensorFlow Lite for on-device (offline) inference
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open("cropguard.tflite", "wb") as f:
    f.write(tflite_model)
```

The final `.tflite` conversion step matters: it's what makes offline, on-phone inference realistic in low-connectivity rural settings.


## Challenges

CropGuard does **not**:

* Diagnose diseases outside the set it was trained on — an unfamiliar disease will either be misclassified or (ideally) flagged as low-confidence
* Replace an agronomist for severe, ambiguous, or high-stakes cases
* Generalize perfectly out of the box — a model trained mostly on one region's dataset can perform noticeably worse on crops, soils, or lighting conditions from another region, and needs local recalibration
* Account for combined stresses (e.g., a plant that's both diseased *and* nutrient-deficient), which is common in the field but rare in curated datasets

There are also ethical considerations to keep in mind: over-trusting a confident-looking wrong prediction could cause a farmer to apply the wrong treatment (wasting money, or in the case of fungicides/pesticides, causing environmental or health harm), so the app should be conservative about when it claims high confidence, and should always make it easy to fall back to a human expert.


## What next?

* Build a feedback loop where farmers can confirm or correct a diagnosis, and use that data to keep improving the model with region-specific images
* Partner with local agricultural extension services so uncertain cases route to a real person rather than dead-ending
* Add a simple voice interface in local languages, since text-heavy apps can exclude farmers with limited literacy
* Expand beyond leaf-based diseases to pest detection and early signs of nutrient deficiency
* Explore lightweight quantization further to run smoothly on very low-end devices


## Acknowledgments

* [PlantVillage dataset](https://www.kaggle.com/datasets/emmarex/plantdisease) — Hughes, D.P. and Salathé, M. (2015), *An open access repository of images on plant health to enable the development of mobile disease diagnostics*
* [MobileNetV2](https://arxiv.org/abs/1801.04381) architecture — Sandler et al., Google
* Building AI — Reaktor Innovations and University of Helsinki
