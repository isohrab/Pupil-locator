# Real time pupil detection in noisy images

Finding pupil location inside the eye image is the prerequisite for eye tracking. Recent state of the art method works based on algorithmic approach and tries to find pupil features and fit an ellipse. However, If the pupil be distorted by strong reflection or eyelash or eyelid (which we call it noisy image) algorithmic approaches will be failed. In this project, which was a part of my master thesis, I designed an hybrid model by inspiring YOLO, Network in Network and using Inception as the core CNN to predict the pupil location inside the image of the eye.

The images for training are noise free and the pupil is evident. The images are automatically labeled by [PuRe](https://arxiv.org/pdf/1712.08900.pdf) and the ground truth is the parameter of an ellipse. I used data augmentation to increase the data quantity and more importantly simulate the noisy images such as occlusion and strong reflection.

To evaluate the model, I used publicly available datasets [ExCuse, ElSe, PupilNet](http://www.ti.uni-tuebingen.de/Pupil-detection.1827.0.html). The results surpass previous state of the art result (PuRe) by 9.2% and achieved 84.4%. The model's speed in an Intel CPU Core i7 7700 is 34 fps and in GPU gtx1070 is 124 fps.

### Results
| Dataset | I | II | III | IV | V | VI | VII | VIII | IX | X | XI | XII | XIII | XIV | XV | XVI |
|:------- |:-:|:--:|:---:|:--:|:-:|:--:|:---:|:----:|:--:|:-:|:--:|:---:|:----:|:---:|:--:|:---:|
| Accuracy(%)| 90 | 88 | 88 | 93 | 97 | 94 | 81 | 89 | 91 | 94 | 97 | 88 | 83 | 97 | 77 | 87 |


| Dataset | XVII | XVIII | XIX | XX | XXI | XXII | XXIII | XXIV | PI | PII | PIII | PIV | PV |
|:--------|:----:|:-----:|:---:|:--:|:---:|:----:|:-----:|:----:|:--:|:---:|:---:|:----:|:---:|
| Accuracy(%)| 96 | 75 | 46 | 89 | 89 | 76 | 100 | 73 | 83 | 60 | 69 | 88 | 82 |

To watch the predicted results please visit this [youtube playlist](https://www.youtube.com/playlist?list=PLDfKspcVguXSZdOxKxqZM25xzhTwUpjpe)

### Run model
Use model to predict the pupil location in a video use this command:
```
python inferno.py PATH_TO_VIDEO_FILE

```
or you can pass 0 for camera.

### Acknowledgement 

This work has been accomplished as a project for my master thesis. I would like to thank Dr. Shahram Eivazi for his generous helps and Thiago Santini for providing the training data. 
