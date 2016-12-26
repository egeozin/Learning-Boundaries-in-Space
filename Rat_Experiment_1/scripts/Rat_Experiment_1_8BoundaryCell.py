#! /usr/bin/env python3
"""
Test client for the <Rat_Experiment_1> simulation environment.

This simple program shows how to control a robot from Python.

For real applications, you may want to rely on a full middleware,
like ROS (www.ros.org).
"""

import sys
import base64
import random
import os
import keras
from keras.models import model_from_json
import math
import h5py
from functools import partial


try:
    from pymorse import Morse
except ImportError:
    print("you need first to install pymorse, the Python bindings for MORSE!")
    sys.exit(1)
from PIL import Image as PILImage

import kivy
from kivy.app import App
from kivy.lang import Builder
from kivy.factory import Factory
from kivy.animation import Animation
from kivy.clock import Clock, mainthread
from kivy.uix.gridlayout import GridLayout
from kivy.graphics import *
from kivy.graphics.texture import Texture
from kivy.uix.image import Image
from kivy.uix.widget import Widget
import threading
import time
import numpy as np
from numpy import linalg as LA

from GridCell import GridCell
import utils
from TrainModel import CNNTraining
#import matplotlib.pyplot as plt

Builder.load_string("""
<AnimWidget@Widget>:
    canvas:
        Color:
            rgba: 0.7, 0.3, 0.9, 1
        Rectangle:
            pos: self.pos
            size: self.size
    size_hint: None, None
    size: 400, 30


<RootWidget>:
    
    rows: 3
    cols: 3
    canvas:
        Color:
            rgba: 0.9, 0.9, 0.9, 1
        Rectangle:
            pos: self.pos
            size: self.size

    but_1: but_1
    but_2: but_2
    but_3: but_3
    but_4: but_4
    but_5: but_5
    but_7: but_7
    img_1: img_1
    training_label:training_label
    check_1:check_1
    check_lbl:check_lbl

    Button:
        id: but_1
        font_size: 20
        text: 'Start Simulation'
        on_press: root.start_second_thread()

    Button:
        id: but_2
        font_size: 20
        text: 'Save Model'
        on_press:root.save_model()
    Button:
        id: but_3
        font_size:20
        text:"Predict"
        on_press:root.predict()
    Button:
        id: but_4
        font_size:20
        text:"Save Data"
        on_press:root.f_save_data()
    Button:
        id: but_5
        font_size:20
        text:"Record Data"
        on_press:root.record_data()

    Button:
        id: but_7
        font_size:20
        text:"Toggle Training Mode"
        on_press:root.toggle_training()

    BoxLayout:
        CheckBox:
            id: check_1
        Label:
            id: check_lbl
            text: "Load Model?"
            color: 0,0,0,1

    Image:
        id: img_1
        size: self.texture_size
        size_hint: None, None
    
    Label:
        id: training_label
        font_size:20
        color: 0,0,0,1
        text: "Training Mode is Disabled"
    Label:
        id: sens_1
        font_size:20

""")

# class GridCell:
#   def __init__(self,px,py,f):
#     self.px=px
#     self.py=py
#     self.f=f
#     self.activation=self.update(0,0)
#   def update(self,x,y):
#     self.activation = 0.5 * math.sin(self.px+2*math.pi*x/self.f)*math.sin(self.py+2*math.pi*y/self.f);  

class RootWidget(GridLayout):
    
    stop = threading.Event()

    counter=0

    def toggle_training(self):
      self.training=not self.training
      if(self.training):
        self.training_label.text= "Training Mode Active"
      else:
        self.training_label.text= "Prediction Mode Active"

    def f_save_data(self):
        self.save_data=True;

    def record_data(self):
        self.record_mode=True

    def load_model(self,file):
        self.cnn.load_weights(file)

    def save_model(self):
        self.cnn.model.save_weights('rat_weights_8_boundary.h5',overwrite=True)

    def predict(self):
        self.predict_mode=not self.predict_mode

    def start_second_thread(self):
        threading.Thread(target=self.second_thread).start()

    def second_thread(self):
        # Remove a widget, update a widget property, create a new widget,
        # add it and animate it in the main thread by scheduling a function
        # call with Clock.
        self.imageSize=256
        self.remove_widget(self.but_1)

        Clock.schedule_once(self.start_test, 0)

        while (True):
          if self.stop.isSet()==True:
            self.simu.quit()
            return
          time.sleep(1)

        # Remove some widgets and update some properties in the main thread
        # by decorating the called function with @mainthread.
        #self.stop_test()

        # Start a new thread with an infinite loop and stop the current one.
        #threading.Thread(target=self.infinite_loop).start()

    def updateData(self,dt):
          #Get Image Data

          image=self.cam.get()["image"]
          #Write Image to Interface

          imageFormat=base64.b64decode( image )
          #np_trial_plot = plt.imshow(np_trial)
          texture = Texture.create(size=(self.imageSize, self.imageSize), colorfmt='rgba' )
          texture.blit_buffer(imageFormat, bufferfmt="ubyte", colorfmt="rgba")

          self.img_1.texture=texture

          #Convert Image to NP Matrix
          image = PILImage.fromstring('RGBA',(self.imageSize,self.imageSize),imageFormat)
          np_image_RGBA=np.array(image)
          np_image_RGB = np.delete(np_image_RGBA,(3), axis=2)
          np_image_RGB = np_image_RGB / 255.0
          #Update Data and Get Features
          raw_boundary=self.dist.get()["range_list"]
          raw_direction= self.pose.get()["yaw"]
          pos_x = self.pose.get()["x"]
          pos_y =self.pose.get()["y"]

          utils.updateGrids(self.Grids,pos_x,pos_y)
          feature_vector=utils.getFeatures(self.Grids.copy(), raw_boundary,raw_direction)
          print ("Actual :",feature_vector)


          if(self.training):
            loss= self.cnn.update_model_online(np_image_RGB, feature_vector)
            # self.move_training(raw_boundary)
            if(self.record_mode):
              self.data_file.write("%s %s %s %s %s  \n" % (pos_x ,  pos_y , raw_direction ,  feature_vector, loss.history))
          else:
            pred  = self.cnn.predict_model(np_image_RGB)
            print("Prediction : ",pred)
            loss=np.mean(np.square(np.log(np.maximum(0.000000001,pred)+1.)-np.log(np.maximum(0.000000001,feature_vector)+1.)))
           
            # loss= self.cnn.update_model_online(np_image_RGB, feature_vector)
            self.move_predict(pred)
            if(self.record_mode):
              self.data_file.write("%s %s %s %s %s \n" % (pos_x , pos_y , raw_direction , pred, loss))

          if(self.save_data):
            self.data_file.close()

          self.simu.sleep(.1)
          self.flush_camera(self.cam,self.simu,.1)
          


    def move_training(self, raw_boundary):
      if(not self.isMoving):
        self.motion.set_speed(0,self.motion_direction*0.4)
        if(raw_boundary[4]>=2):
          self.isMoving=True
      else:
        self.motion.set_speed(1,0)
        print("Distance Ahead :",raw_boundary[4])
        if(raw_boundary[4]<=2):
          self.isMoving=False
          self.motion_direction=np.sign(random.random()-0.5)*self.motion_direction

    def move_predict(self,pred):
      if(not self.isMoving):
        self.motion.set_speed(0,self.motion_direction*0.4)
        if((pred[0,3]+pred[0,4]+pred[0,5])/3.0<=np.mean(pred[0])):
          self.isMoving=True
      else:
        self.motion.set_speed(1,0)
        if((pred[0,3]+pred[0,4]+pred[0,5])/3.0>np.mean(pred[0])):
          self.isMoving=False
          self.motion_direction=np.sign(random.random()-0.5)*self.motion_direction

    def start_test(self, *args):
      self.simu=Morse()
      self.cam=self.simu.robot.camera
      self.motion=self.simu.robot.motion
      self.pose = self.simu.robot.pose
      self.dist = self.simu.robot.laserscanner
      
      numGrids= 24
      self.Grids=utils.generateGrids(numGrids)
      
      # Get one image and feed it to CNN Model
      image=self.cam.get()["image"]
          #Write Image to Interface

      imageFormat=base64.b64decode( image )
      image = PILImage.fromstring('RGBA',(self.imageSize,self.imageSize),imageFormat)
      np_image_RGBA=np.array(image)
      np_image_RGB = np.delete(np_image_RGBA,(3), axis=2)
      np_image_RGB= np_image_RGB/255.0
      self.cnn=CNNTraining(np_image_RGB,8)
      self.currentNode=0

      self.predict_mode=False
      self.isMoving = False
      self.motion_direction=1
      self.is_load_model=False
      self.training = False
      self.record_mode=False
      self.save_data=False
      self.data_file=open('3Min_Training_8B_Room7_Prediction','w')
      if(self.check_1.active):
        self.load_model('rat_weights_8_boundary.h5')
      # Remove the button.
    
      Clock.schedule_interval(self.updateData, .5)
      



    @mainthread
    def stop_test(self):
        self.simu.quit()


    def flush_camera(self,camera_stream, morse, timeout):
      timeout_t = morse.time() + timeout
      print("flush_camera %f" % timeout_t)
      while morse.time() < timeout_t:
          print("%f %f" % (morse.time(), timeout_t))
          # get a new image from the camera stream
          camera_stream.get()
          morse.sleep(.1)


class ThreadedApp(App):

    def on_stop(self):
        # The Kivy event loop is about to stop, set a stop signal;
        # otherwise the app window will close, but the Python process will
        # keep running until all secondary threads exit.
        self.root.stop.set()

    def build(self):
        return RootWidget()

if __name__ == '__main__':
    ThreadedApp().run()



