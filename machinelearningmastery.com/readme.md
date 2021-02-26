
Result of stepping through the tutorial from https://machinelearningmastery.com/how-to-develop-a-convolutional-neural-network-to-classify-photos-of-dogs-and-cats/

I hope I included everything you need to see the end result, along with a few real-world test images and a misclassified image (e.g., `wolf.png`).

1. Use 7zip to decompress `final_model_vgg16_transfer_zip.001` - `final_model_vgg16_transfer_zip.008` into `final_model_vgg16_transfer.h5`.

1. Run python script dogs_vs_cats.py

My favorites are the dogs wearing ears and the stuffed animal.

Output:
```
model_type: vgg16_transfer
image filename: unseen_sample_image_1.jpg
model type filename: final_model_vgg16_transfer.h5
exact result (close to 0.0=cat, close to 1.0=dog): 1.0
Should be: a dog
Predicted answer: a dog

image filename: kitten-440379.jpg
model type filename: final_model_vgg16_transfer.h5
exact result (close to 0.0=cat, close to 1.0=dog): 0.002306572161614895
Should be: a kitten
Predicted answer: a cat

image filename: tiger.jpg
model type filename: final_model_vgg16_transfer.h5
exact result (close to 0.0=cat, close to 1.0=dog): 6.196310961071108e-14
Should be: a tiger
Predicted answer: a cat

image filename: wolf.png
model type filename: final_model_vgg16_transfer.h5
exact result (close to 0.0=cat, close to 1.0=dog): 0.46810227632522583
Should be: wolf in art
Predicted answer: a cat

image filename: wolf_2.jpg
model type filename: final_model_vgg16_transfer.h5
exact result (close to 0.0=cat, close to 1.0=dog): 0.999248206615448
Should be: wolf
Predicted answer: a dog

image filename: harry.JPG
model type filename: final_model_vgg16_transfer.h5
exact result (close to 0.0=cat, close to 1.0=dog): 0.00040015517151914537
Should be: harry, our cat
Predicted answer: a cat

image filename: card_with_puppy.JPG
model type filename: final_model_vgg16_transfer.h5
exact result (close to 0.0=cat, close to 1.0=dog): 0.9961304664611816
Should be: puppy on a card
Predicted answer: a dog

image filename: dog with cat ears.jpg
model type filename: final_model_vgg16_transfer.h5
exact result (close to 0.0=cat, close to 1.0=dog): 0.9999898672103882
Should be: dog with cat ears
Predicted answer: a dog

image filename: dog with bunny ears.jpg
model type filename: final_model_vgg16_transfer.h5
exact result (close to 0.0=cat, close to 1.0=dog): 0.8947312831878662
Should be: dog with bunny ears
Predicted answer: a dog

image filename: stuffed animal dog.jpg
model type filename: final_model_vgg16_transfer.h5
exact result (close to 0.0=cat, close to 1.0=dog): 0.8378120064735413
Should be: stuffed animal dog
Predicted answer: a dog


Process finished with exit code 0
```
