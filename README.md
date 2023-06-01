# api
Die Sensor Based Activity Recognition (SBAR) API ist eine Machine Learning-basierte API, die auf unkalibrierten Sensordaten beruht. Die API verwendet verschiedene Machine- und Deep-Learning Modelle, um Bewegungsprofile vorherzusagen. Diese API hat zwei Hauptendpunkte, die CNN und HGBC heißen und jeweils unterschiedliche Modelle verwenden.

## Endpunkte:
Die beiden Hauptendpunkte sind:

- CNN: Ein Endpunkt, der ein Convolutional Neural Network (CNN) verwendet. Dieses Modell nutzt STFT Spektren zur Vorhersage von Bewegungsprofilen. Zu erreichen unter: https://sbar.fuet.ch/CNN

- HGBC: Ein Endpunkt, der den Histogram Gradient Boosting Classifier (HGBC) verwendet. Dieses Modell verwendet eine Zusammenführung von Fast Fourier Transformationen. Zu erreichen unter: https://sbar.fuet.ch/HGBC

Die Endpunkte erwarten Daten im CSV-Format, die gzip-komprimiert sind.

## Anforderungen:
- Python 3.10 oder höher
- Die Python-Pakete, die in der Datei requirements.txt aufgeführt sind.

## Installation mittels Docker (Linux):
Um die API mittels Docker zu installieren, führen Sie die folgenden Schritte aus:

1. Klonen Sie das Repository mit ```git clone https://github.com/Sensor-Based-Activity-Recognition/api.git```.
2. Wechseln Sie in das Verzeichnis.
3. Passen Sie die Netzwerkeinstellungen in der Datei **create_docker_container.sh** an.
4. Führen Sie das Skript mit ```sudo sh **create_docker_container**.sh``` aus.
5. Richten Sie einen Reverse Proxy (z.B. NGINX) ein, um die API zu routen.

## Nutzung:
Ein Beispiel für einen API-Request finden Sie im Jupyter Notebook **example_request.ipynb**. 

## Response:
Die Response der API ist ein JSON-Objekt, das die Vorhersage der Aktivitäten für verschiedene Zeitsegmente beinhaltet. Beispielsweise sieht eine Response wie folgt aus:

```json
{
  "0": {
    "Sitzen": 1.0,
    "Laufen": 0.0,
    "Velofahren": 0.0,
    "Rennen": 0.0,
    "Stehen": 0.0,
    "Treppenlaufen": 0.0
  },
  "1": {
    "Sitzen": 1.0,
    "Laufen": 0.0,
    "Velofahren": 0.0,
    "Rennen": 0.0,
    "Stehen": 0.0,
    "Treppenlaufen": 0.0
  },
  ...
}
```
Die Daten werden beim Einlesen in 5-Sekunden-Segmente unterteilt. Für jedes dieser Segmente wird eine Vorhersage gemacht. Die Segmente sind durch einen Index gekennzeichnet, der als Schlüssel ("Key") im JSON-Objekt verwendet wird.

Die Vorhersage für ein Segment ist ein weiteres JSON-Objekt, das die Wahrscheinlichkeit für jede Aktivität enthält. Die Aktivitäten sind als Schlüssel und die Wahrscheinlichkeiten als Werte ("Value") gespeichert
