#include <Servo.h>

Servo pan;
Servo tilt;


void setup() {

  // Setup Servomotors
  // initialize serial communication:
  Serial.begin(115200);
  // initialize the LED pin as an output:
  pan.attach(8);
  pan.write(0);

  tilt.attach(9);
  tilt.write(0);

}

void loop() {
  
  control_servo();

}

void control_servo(){

  String incomingByte;
  int ind_sep;
  int pos_pan_, pos_tilt_;
  
  
  if (Serial.available()){
    incomingByte = Serial.readStringUntil('\n');
    ind_sep = incomingByte.indexOf('/');

    pos_pan_ = incomingByte.substring(0,ind_sep).toInt();
    pos_tilt_ = incomingByte.substring(ind_sep+1,incomingByte.length()).toInt();
    
    pan.write(pos_pan_);
    tilt.write(pos_tilt_);

    
  }

}
