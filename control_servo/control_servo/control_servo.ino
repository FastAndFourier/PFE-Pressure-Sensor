#include <Servo.h>

Servo pan;
Servo tilt;

int i_pan = 0;
int i_tilt = 0;

String measures;

String flag_data;


//****************************************//

#define DECLINATION -5.55 

#define PRINT_CALCULATED  //print calculated values

// Call of libraries
#include <Wire.h>
#include <SparkFunLSM9DS1.h>

// defining module addresses
#define LSM9DS1_AG 0x6B //accelerometer and gyroscope

LSM9DS1 imu; // Creation of the object

void setup() {

  // Setup Servomotors
  // initialize serial communication:
  Serial.begin(115200);
  // initialize the LED pin as an output:
  pan.attach(32);
  pan.write(0);

  tilt.attach(33);
  tilt.write(0);


  // Setup accelerometer
  Wire.begin();     //initialization of the I2C communication
  imu.settings.device.commInterface = IMU_MODE_I2C; // initialization of the module
  imu.settings.device.agAddress = LSM9DS1_AG;
  if (!imu.begin()) //display error message if that's the case
  {
    Serial.println("Communication problem.");
    while (1);
  }


}

void loop() {
  
  control_servo();

  /*
  if(Serial.available()>0){
    flag_data = Serial.readStringUntil('\n');
    if (flag_data.equals("send_data")){
      print_servo();
    }
  }*/
  
  


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
    /*
    if( imu.accelAvailable() ){
      imu.readAccel(); //measure with the accelerometer
    }*/

    Serial.print("r/");
    /*
    if(imu.calcAccel(imu.ax) > 0){
      Serial.print('+');
    }
    Serial.print(imu.calcAccel(imu.ax),2);
    Serial.print("/");
    if(imu.calcAccel(imu.ay) > 0){
      Serial.print('+');
    }
    Serial.print(imu.calcAccel(imu.ay),2);
    Serial.print("/");
    if(imu.calcAccel(imu.az) > 0){
      Serial.print('+');
    }
    Serial.print(imu.calcAccel(imu.az),2);
    Serial.print("//");*/

    
  }

}

void print_accel(){
  
  Serial.print("Accel/");
  Serial.print(imu.calcAccel(imu.ax),2);
  Serial.print("/");
  Serial.print(imu.calcAccel(imu.ay),2);
  Serial.print("/");
  Serial.print(imu.calcAccel(imu.az),2);
  Serial.print("//");
}

void print_servo(){
  Serial.print("Servo/");
  String pos_pan = String(pan.read());
  String pos_tilt = String(tilt.read());
  
  for(int k=0; k<= 2-pos_pan.length(); k++){
    Serial.print("0");
  }
  Serial.print(pos_pan+'/');
  
  for(int k=0; k<= 2-pos_tilt.length(); k++){
    Serial.print("0");
  }
  Serial.println(pos_tilt);
}

void display_accel(){
  Serial.print(-2);   //calculate acceleration
  Serial.print(", ");
  Serial.print(2);   //calculate acceleration
  Serial.print(", ");
  Serial.print(imu.calcAccel(imu.ax), 2);   //calculate acceleration
  Serial.print(", ");
  Serial.print(imu.calcAccel(imu.ay), 2);
  Serial.print(", ");
  Serial.print(imu.calcAccel(imu.az), 2);
  Serial.print("\n");
}

/*
void printAccel()
{
#ifdef PRINT_CALCULATED
  Serial.print(-2);   //calculate acceleration
  Serial.print(", ");
  Serial.print(2);   //calculate acceleration
  Serial.print(", ");
  Serial.print(imu.calcAccel(imu.ax), 2);   //calculate acceleration
  Serial.print(", ");
  Serial.print(imu.calcAccel(imu.ay), 2);
  Serial.print(", ");
  Serial.print(imu.calcAccel(imu.az), 2);
  Serial.print("\n");                    //measured in g
#elif defined PRINT_RAW
  Serial.print(imu.ax);                   //or display raw data
  Serial.print(", ");
  Serial.print(imu.ay);
  Serial.print(", ");
  Serial.println(imu.az);
#endif
}
*/
