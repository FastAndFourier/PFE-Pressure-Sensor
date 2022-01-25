#define DECLINATION -5.55 

#define PRINT_CALCULATED  //print calculated values

// Call of libraries
#include <Wire.h>
#include <SparkFunLSM9DS1.h>

// defining module addresses
#define LSM9DS1_AG 0x6B //accelerometer and gyroscope

LSM9DS1 imu; // Creation of the object

const int wpin[] = {36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53};
const int rpin[] = {A7,A8,A9,A10,A11,A12,A13,A14,A15};
int nb_numpin = 18;
int nb_analogpin = 9;



void setup() {
  // initialize serial communication:
  Serial.begin(115200);

  // Setup accelerometer
  Wire.begin();     //initialization of the I2C communication
  imu.settings.device.commInterface = IMU_MODE_I2C; // initialization of the module
  imu.settings.device.agAddress = LSM9DS1_AG;
  if (!imu.begin()) //display error message if that's the case
  {
    Serial.println("Communication problem.");
    while (1);
  }

  for (int w = 0; w<nb_numpin; w++)
  {
    pinMode(wpin[w], INPUT);
  }
  for (int r = 0; r<nb_analogpin; r++)
  {
    pinMode(rpin[r], INPUT_PULLUP);
  }

}

void loop() {

  if( imu.accelAvailable() ){
    imu.readAccel(); //measure with the accelerometer
  }

    
  Serial.println("a");
  Serial.println(imu.calcAccel(imu.ax),2);
  Serial.println(imu.calcAccel(imu.ay),2);
  Serial.println(imu.calcAccel(imu.az),2);

  //Serial.println("//");
  Serial.println("p");
  for (int w = 0; w<nb_numpin; w++)
  {
    pinMode(wpin[w], OUTPUT);
    digitalWrite(wpin[w], LOW);
    for (int r = 0; r<nb_analogpin; r++)
    {
      Serial.println(analogRead(rpin[r]));
    }
    pinMode(wpin[w],INPUT);
  }
  

}
