# 기상청 Data에 대한 BRITS
[원본코드](https://github.com/Doheon/TimeSeriesImputation-BRITS) <br/>
[논문](https://doheon.github.io/%EB%85%BC%EB%AC%B8%EB%B2%88%EC%97%AD/time-series/pt-brits-post/)

BRITS 모델을 기상청 Data에 맞게 설계한 방식
<br/>기상 Data의 특성에 맞게 Custom함.

### <사용방법>
1. data 폴더에 'train_data'라는 이름으로 파일을 넣는다.
<br/> 이 때, Data의 첫줄은 시간을 나타내야 하고, 첫 행은 Colmn명이다.
2. 'train-data'의 일부를 잘라 'val_data'로 만든다.
<br/> 이 때, 'val_data'는 잘라진 Data에 대한 한 줄만 존재한다.
3. Hyper-prameter를 조정한 후 실행한다.
<br.>Classification 및 Regression 을 원하면 추가 조정을 해야한다.
