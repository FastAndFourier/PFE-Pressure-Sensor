#include <Servo.h>

Servo pan;
Servo tilt;
int pos_pan = 0;
int pos_tilt = 0;

int ind_sep;

String incomingByte;


void setup() {
  // initialize serial communication:
  Serial.begin(9600);
  // initialize the LED pin as an output:
  pan.attach(8);
  pan.write(0);

  tilt.attach(9);
  tilt.write(0);

}

void loop() {

  
  if (Serial.available()){
    incomingByte = Serial.readStringUntil('\n');
    ind_sep = incomingByte.indexOf('/');

    
    pos_pan = incomingByte.substring(0,ind_sep).toInt();
    pos_tilt = incomingByte.substring(ind_sep+1,incomingByte.length()).toInt();
    
    pan.write(pos_pan);
    tilt.write(pos_tilt);
    
    Serial.print(incomingByte);
  }


}
