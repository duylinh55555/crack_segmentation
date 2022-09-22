#!/bin/sh
gdown https://drive.google.com/uc?id=1xrOqv0-3uMHjZyEUrerOYiYXW_E8SUMP
mkdir ./models
gdown https://drive.google.com/uc?id=1wA2eAsyFZArG3Zc9OaKvnBuxSAPyDl08 -O "./models/model_unet_vgg_16_best.pt"