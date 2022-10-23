
#이 프로젝트는 악기연주를 하고 싶어하는 사람들에게 조금이라도 도움이 되길 바라며 만드는 프로젝트이다. 
### 어떤 이유에서건 악기를 연주해보고자 하는 사람들에게, 위안과 안식이라는 작은 선물이 될 수 있기를...


#### 이 과정은 2022 데이터청년캠퍼스에서 진행했던 프로젝트 자료로, 팀원들과 협의하여 각자 자신이 바라는 방식으로 업데이트하기로 했다. 나는, 이 프로젝트를 업로드하고 추가적으로 혼자 보완하고자 여기에 올려두었다. 

* 진행 중인 악기: 기타
* 현재 업데이트 중인 과정: 기타 운지를 확인하게 모델의 정확도 향상 작업

### 이번 프로젝트에서 프로그램을 실행시키고 시스템을 제작한 파일은 music, instrument, groove2groove 폴더에 존재한다.

## setups

## 동일한 환경 설정이 요구되므로 아래 설정을 따라야 된다.
1. 다른 프로젝트에서 다른 버젼의 패키지를 사용하는 경우 충돌이 일어날 수 있기 떄문에, 가상환경이 필요함.
2. 동일한 환경을 설정하기 위해서 아래 설정을 따라야된다.
3. 이 과정은 두 단계로 나눠지며 각각 다른 환경에서 진행된다

### 실행 전에 필수적으로 설치해야될 것
1. Dockers
2. pip or miniconda3
### 가상환경 설정하는 법

* pip 환경에서 설치할 때
1. `python -m venv venv`
2. 다른 환경에서 가상환경을 실행하는 법
   * Windows: `.\venv\Scripts\activate`
   * Linux(in Windows), Mac: `source venv/bin/activate`
3. pip 환경에서 requirments.txt에 있는 패키지 설치: `pip install -r requirments.txt`

* conda 환경에서 설치할 때
1. `conda create -n <환경명> python=<버전>`
2. `conda activate <환경명>`
3. `pip install -r requirements.txt`

### 실행하기 위해 준비할 환경
한 곳에 instrument을 다른 곳에는 groove2groove와 music를 실행시키는 구조이다.
   * music를 실행할 환경에서 설치/실행 시킬 것(python 환경:python 3.6.9)

       `docker pull mctlab/omnizart:latest`
   
       `docker run -it -p 5000:5000 mctlab/omnizart:latest bash` 
   
       `pip install -r requirements.txt`
   * groove2groove(현재 모델 체크포인트를 groove2groove로 부터 다운 받아 experiment 폴더 안에 v01 안에 넣어놓은 상태)
       `conda env create -f environment.yml`
   
       `pip install -r requirements.txt`
   
       `conda activate groove2groove`
   
       `pip install './code[gpu]'`
   
       `export LOGDIR=experiments/v01`

   * instrument에서 실행할 환경에서 설치해야할 것(python 환경: python 3.8)
        
     `pip install -r requirements.txt`

   * 악보 추출 시에 해야 할 것
       설치: musescore2.3.4(https://musescore.org/ko), lilypond(http://lilypond.org/)
       확인할 것: app.py의 33~36번째 줄에서 path 설정을 맞춰서 할 것.

### 모델 생성 방법
1. instrument/related_to_data 에 간다. 
2. 이 위치에 따로 첨부된 raw data를 두고 Trainer_fineTuning.py를 실행시킨다.
3. 새로 생성된 guitar_learner.h5를 instrument 폴더로 옮긴 후 실행 방법을 따라서 작동시키면 된다.


### 실행 방법(VScode에서 실행하는 것을 추천함)
1. 두개의 Windows를 준비한다.
2. 한 Window에서 View-> Command Palette-> Remote-Containers: Attach to Running Container
-> 앞에서 설정한 cotainer 선택한다.
3. 새로 열린 Window에 music와 groove2groove 파일을 넣는다.
4. 다른 Window창에 instrument 폴더를 넣고  app.py를 실행시킨다.
5. 기존 music폴더가 열려있는 Window창으로 가서 flask_upload.py를 실행시킨다.
6. 시연 동영상에 맞춰서 따라서 하면 된다.


### fuiture_updates 는 실제로 만들다가 미비해서 제외한 파일들로, 차후 업데이트시 사용할 예정이다.
