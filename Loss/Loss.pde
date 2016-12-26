Table table;
float posX;
float posY;
PVector pos;
PVector rot ;
int counter=0;
void setup(){
  background(255);
  table = loadTable("Grid_Activations_1G_Room_2_Actual.csv");
  colorMode(HSB,255);
  size(1000 ,1000);
   frameRate(5);
   rectMode(CENTER);
  noStroke();
  noLoop();
}


void draw(){
  //background(0);
int x=1;
float prevLoss=0;
PVector dir=new PVector(0,0); 
PVector north= new PVector(1,-0.000001);
print(north.heading());
for(TableRow row:table.rows()){
  
  //dir=PVector.fromAngle(row.getFloat(2));
  //float angle=PVector.angleBetween(dir,north);
  
  //if(row.getFloat(2)-north.heading()<0 ||row.getFloat(2)-north.heading()>PI ){
  //angle=-1*angle;
//}
////  int globalCell=int(round(9-(angle/(PI/4))));
  //if(globalCell==15)
  //globalCell=3;
  //float front=row.getFloat(globalCell);
  //int front2=0;
  //int front3=0;
  //if(globalCell==3){
  // front3=14;
  // front2=4;
//}
////  else if(globalCell==14){
  //  front2=3;
  //  front3=13;
//}
////  else{
  //  front3=globalCell-1;
  //  front2=globalCell+1;
  //}
  
  //println(globalCell + " , " + front2 + " , " + front3);
  //front=(front+row.getFloat(front2)+row.getFloat(front3))/3;
  //// Front facing boundary detector
  //front=map(front,0,0.7,0,1);
  
  
  
  

  
  //float loss=row.getFloat(11);
  float pos_x=row.getFloat(0);
  float pos_y=row.getFloat(1);
  
  float grid24=row.getFloat(3);
  println(grid24);
  grid24=map(grid24,0.4,0.6,0,1);
  fill(25*grid24,255,255);
  float xx=pos_x*40+500;
  float yy = pos_y*40+500;
  
  
  rect(xx,yy,5,5);
  line(xx,yy,xx+dir.x*15,yy+dir.y*15);
  
  //text(globalCell,xx+50,yy);
  //print(row.getFloat(2));
  //println("");
  //println(angle);
  
  //prevLoss=loss;
  x++;
}
saveFrame("Room2_Grid23.tiff");
}