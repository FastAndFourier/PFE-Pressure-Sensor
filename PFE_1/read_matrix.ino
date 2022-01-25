//--------------------------------------------------//
//this soft read data from the pressure matrix and send it through serial port. It can be read by a python file named read_data.py
//In this file you have to have matching table size (change taillex and tailley values to match nb_numpin and nb_analogpin)
//--------------------------------------------------//

//pin map for the foot
const int wpin[] = {36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53};
const int rpin[] = {A7,A8,A9,A10,A11,A12,A13,A14,A15};
int nb_numpin = 18;
int nb_analogpin = 9;

//pin map for the soft gripper :
/*
const int wpin[] = {};
const int rpin[] = {};
int nb_numpin = 18;
int nb_analogpin = 9;
*/

void setup() {
  // put your setup code here, to run once:
  Serial.begin(9600);
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
  // put your main code here, to run repeatedly:
  Serial.println(1028);//caractere de fin les valeurs ne sepassent jamais 1023
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
