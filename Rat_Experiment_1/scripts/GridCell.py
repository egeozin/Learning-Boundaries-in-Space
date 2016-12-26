import math

class GridCell:
  def __init__(self,px,py,f):
    self.px=px
    self.py=py
    self.f=f
    self.activation=self.update(0,0)
  
  def update(self,x,y):
    self.activation = 0.5 * math.sin(self.px+2*math.pi*x/self.f)*math.sin(self.py+2*math.pi*y/self.f); 
  