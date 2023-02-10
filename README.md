# Application-Breast-Cancer-Predicting
### Main Window                                                                                    
![image](https://user-images.githubusercontent.com/36747104/218097307-83197823-e108-437c-9e32-ede9d99b1e5f.png)

### Checking accuracy of current model                                                                           
![image](https://user-images.githubusercontent.com/36747104/218105549-58059f63-d4b7-4a14-b9cf-7f1998713aa4.png)

### Predicting status based on input values                                                                        
![image](https://user-images.githubusercontent.com/36747104/218105804-9152a37f-be39-4943-b2f9-b93a66ec0798.png)

### Training net several times. Sometimes it might improve your accuracy.
![image](https://user-images.githubusercontent.com/36747104/218107179-bbe29e81-fab0-4ece-b286-90f97ee840fc.png)
![image](https://user-images.githubusercontent.com/36747104/218107581-4e74523d-ef67-4c79-9306-a94cd31d8d64.png)

## If you need an app
- First of all, you have to extract DIST and BUILD folders, then put them in one general folder.
- Secondly, you must find file located in dist/app/tensorflow/python and called _pywrap_tensorflow_internal.7z and unpack it in dist/app/tensorflow/python.
- Run the app, using app.exe located in dist/app/.

## In case you are interested in code
- BCDNeuralNetwork.py consist of a neural network class, based on reformed SEER Breast Cancer dataset.
- MainWindow.py consist a PySide6 main window class, designed for this particular net.
- App.py duplicates BCDNeuralNetwork.py MainWindow.py. It represents the whole program itself.

## Features and complaints
- As you may notice all categorical data in original dataset were encoded by using Embedded Encoding, original net for encoding was not saved.
- BCDNeuralNetwork.py posses much more usefull methods and attributes, that did not appear in final program.
- If you want to use original dataset or my datasets, please pay attention to white spaces, since some columns might have an extra whitespace.
