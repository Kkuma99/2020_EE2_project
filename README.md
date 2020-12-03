# 2020년 전기전자심화설계및소프트웨어실습 과목 텀프로젝트
## ⚡️개요

- 참여자: 강민지, 우상우, 진시민
- 일시: 2020.11.24 ~ 2020.12.18
- 주제: open pose를 이용한 자세 교정

(항상 시작하기전에 git pull 하고 push 하기 전 확인!)


---
## 20.11.24

주제 관련 토의 진행

- 주제: human pose를 이용한 무엇
  - 세분화: 사람 자세를 이용한 무언가? (여기서 참신성 필요함)
- human pose: 있는 알고리즘 조사 - 사용 (업그레이드 필요)
- evaluation도 진행할 수 있으면 진행하기
- 큰 주제에 적용해서 시뮬까지 찍을수 있으면 찍기

<br>

 <details>
<summary><span style="color:blue">🚩타임 테이블(클릭해서 확인)</span></summary>

| 주차 | 월 | 화 | 수 | 목 | 금 | 토 | 일 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 11.24~11.29 | | 주제 관련 토의 | human pose 알고리즘 조사 / 정리 | 주제 픽스 / 개요 레포트 | 오픈 소스 적용 시작~ | | |
| 11.30~12.6 | 개별 진도 체크 | 진행사항 토의 / 방향성 잡기 | 지정 방향으로 실습 진행 | 진행사항 레포트 제출 / 실습 | ~ | | |
</details>

---
## 20.11.26

주제 확정

- open pose 활용
  - https://github.com/microsoft/human-pose-estimation.pytorch
  - https://github.com/ildoonet/tf-pose-estimation
  - https://github.com/facebookresearch/VideoPose3D
  - https://github.com/luvimperfection/pose-estimation
  - https://github.com/NVIDIA-AI-IOT/trt_pose
  - https://m.blog.naver.com/shino1025/221607197982
  
- 자세 교정
  - 걸을 때: 어깨와 목 자세
  - 앉을 때: 어깨, 목, 허리, 등
    - 앉을 때: 얼마나 앉아있는지, 몇 % 바른 자세로 있었는지, 바른 자세가 아니면 컴퓨터 화면에 알림 띄우기

- 알림: opencv 사용 (사용자가 젯슨을 메인으로 쓰고있다는 가정하에 - 만약 시간이 남으면 앱 혹은 서버로 알림 찾아보기)

- 최적화

- 성능 비교(mAP, evaluation)

- 발표 준비

---
## 20.12.01
### 시민
pose-estimation 실행 방법

-github : https://github.com/luvimperfection/pose-estimation

- 설치 순서
  - git clone https://github.com/spmallick/learnopencv.git
  - cd learnopencv/OpenPose
  - sudo chmod a+x getModels.sh
  - ./getModels.sh

- OpenPoseImage.py 수정 (영상:OpenPoseVideo.py)
  - MODE = "MPI" // COCO와 다르게 허리에 점
  - cv2.imshow 주석 // 원격으로 display 불가능
  - cv2.imread("파일 이름")
  
- 실행
  - python3 OpenPoseImage.py (영상:OpenPoseVideo.py)
  
- 결과
  - Output-Skeleton.jpg : 선과 점 표시
  - Output-Keypoints.jpg : 점과 번호 표시
 
---
## 20.12.02
### 민지

- 동영상에 각도 코드 적용
- 데이터 몇개 가지고 테스트 진행

- 할일:
  1. 동영상 종료 조건 진행
  2. 각도 동영상에 표시하기
  3. 각도 확실하게 지정
  4. 옳은 자세 틀린 자세 구분 코드 수정
  5. 쓰레시 값 결정
  6. gpu 사용여부 확인

---
## 20.12.03

- 회의 진행
  - 각도 확정 (12.3~12.4 시민)
  - 종료 조건 (12.3~12.4 민지)
  - thresh 값 정리 (12.3~12.4 상우)
  - 12.5: 바른/틀린 자세 각도 세팅하고 동영상에 프린트
  - 12.6: 자세 관련 기록 -> txt 파일, cv로 프린트 진행 (시민,민지) // video stream 관련 조사 (상우)
  - 12.4~12.7: 동영상 저장 / 화면에 기록 등
  - 12.7: 회의 -> map 계산, input 사이즈 줄이기 시도, 선 자세 확인
  - 마지막주: 부족한거 넣기(실시간 비디오), 발표 시연 준비, 레포트
