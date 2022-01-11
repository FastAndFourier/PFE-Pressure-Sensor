const byte wpin[] = {0,1,2,3,4,5,6,7,8,9,10,11,12,13}; //,40,42,44,46,48,50,52};
const int rpin[] = {A0,A1,A2,A3,A4,A5};
void setup() {
  // put your setup code here, to run once:
  Serial.begin(9600);
  for (int w = 0; w<9; w++)
  {
    pinMode(wpin[w], INPUT);
  }
  for (int r = 0; r<9; r++)
  {
    pinMode(rpin[r], INPUT_PULLUP);
  }
}

void loop() {
  // put your main code here, to run repeatedly:
  Serial.println(1028);//caractere de fin les valeurs ne sepassent jamais 1023
  for (int w = 0; w<14; w++)
  {
    pinMode(wpin[w], OUTPUT);
    digitalWrite(wpin[w], LOW);
    for (int r = 0; r<6; r++)
    {
      Serial.println(analogRead(rpin[r]));
    }
    pinMode(wpin[w],INPUT);
  }
}
