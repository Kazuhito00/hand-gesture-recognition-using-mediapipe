# hand-gesture-recognition-using-mediapipe
MediaPipe(Python版)を用いて手の姿勢推定を行い、検出したキーポイントから<br>ハンドサインとフィンガージェスチャーを認識するサンプルプログラムです。
![mqlrf-s6x16](https://user-images.githubusercontent.com/37477845/102222442-c452cd00-3f26-11eb-93ec-c387c98231be.gif)

# Requirements
* mediapipe 0.8.1
* OpenCV 3.4.2 or Later
* Tensorflow 2.3.0 or Later
* scikit-learn 0.23.2 or Later (学習時に混同行列を表示したい場合のみ) 
* matplotlib 3.3.2 or Later (学習時に混同行列を表示したい場合のみ)

* tf-nightly 2.5.0.dev20201214 or later (LSTMのTFLiteを使用したい場合のみ)
