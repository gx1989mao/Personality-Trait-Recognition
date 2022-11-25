import audiovisual_stream
import chainer.serializers
import librosa
import numpy
import skvideo.io
import cv2

def load_audio(data):
    return librosa.load(data, 16000)[0][None, None, None, :]

def load_model():
    model = audiovisual_stream.ResNet18()
    
    
    chainer.serializers.load_npz('./model', model)
    
    return model

def load_video(data):
    videoCapture = cv2.VideoCapture(data)
    # videoCapture = skvideo.io.VideoCapture(data, (456, 256))
    
    # videoCapture.open()
    
    x = []
    
    while True:
        retval, image = videoCapture.read()
        
        if retval:
            image = cv2.resize(image,(456, 256))
            x.append(numpy.rollaxis(image, 2))
        else:
            break
    
    return numpy.array(x, 'float32')

# def load_video(data):
#     videogen = skvideo.io.vreader(data)
#     for frame in videogen:
#         print(frame.shape)

def predict_trait(data, model):
    x = [load_audio(data), load_video(data)]
    
    return model(x)
