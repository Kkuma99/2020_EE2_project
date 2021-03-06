import cv2             # for image processing
import time            # for FPS
import numpy as np     
import argparse        # for 인자

parser = argparse.ArgumentParser(description='Run keypoint detection')
parser.add_argument("--device", default="cpu", help="Device to inference on") # 거의 픽스해서 사용
parser.add_argument("--video_file", default="sample_video.mp4", help="Input Video") # Input

args = parser.parse_args()

MODE = "MPI" # 모드 픽스하기

if MODE is "COCO":
    protoFile = "pose/coco/pose_deploy_linevec.prototxt"
    weightsFile = "pose/coco/pose_iter_440000.caffemodel"
    nPoints = 18
    POSE_PAIRS = [ [1,0],[1,2],[1,5],[2,3],[3,4],[5,6],[6,7],[1,8],[8,9],[9,10],[1,11],[11,12],[12,13],[0,14],[0,15],[14,16],[15,17]]

# MODE
elif MODE is "MPI" :
    protoFile = "pose/mpi/pose_deploy_linevec_faster_4_stages.prototxt"
    weightsFile = "pose/mpi/pose_iter_160000.caffemodel" # using trained weight model
    nPoints = 15                                         # MPI model's point
    POSE_PAIRS = [[0,1], [1,2], [2,3], [3,4], [1,5], [5,6], [6,7], [1,14], [14,8], [8,9], [9,10], [14,11], [11,12], [12,13] ]
    # head = 0, neck = 1, waist = 14, hip = 8

# for faster time -> reduce size of input
inWidth = 368
inHeight = 368
threshold = 0.1 # 적절한 쓰레시 값 찾기

#********************************* video input 처리 ************************************#

input_source = args.video_file       #사용자에게 input 받은 비디오 파일 사용
cap = cv2.VideoCapture(input_source) 
hasFrame, frame = cap.read()

vid_writer = cv2.VideoWriter('output_check.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame.shape[1],frame.shape[0])) # output 저장

net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile) # 일단 신경망은 오픈소스 그대로 사용
if args.device == "cpu":
    net.setPreferableBackend(cv2.dnn.DNN_TARGET_CPU)
    print("Using CPU device")
elif args.device == "gpu":
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    print("Using GPU device")


# 동영상 처리
while cv2.waitKey(1) < 0:
    t = time.time()             # 동영상 시작 시간
    hasFrame, frame = cap.read()
    frameCopy = np.copy(frame)
    if not hasFrame:
        cv2.waitKey()
        break

    frameWidth = frame.shape[1]
    frameHeight = frame.shape[0]

    inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (inWidth, inHeight),
                              (0, 0, 0), swapRB=False, crop=False)
    net.setInput(inpBlob)
    output = net.forward()

    H = output.shape[2]
    W = output.shape[3]
    
    # 특징점들 초기화
    point_x = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    point_y = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0] 

    # Empty list to store the detected keypoints
    points = []

    for i in range(nPoints):
        # confidence map of corresponding body's part.
        probMap = output[0, i, :, :] # map checking point

        # Find global maxima of the probMap.
        minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)
        
        # Scale the point to fit on the original image
        x = (frameWidth * point[0]) / W
        y = (frameHeight * point[1]) / H

        #  for문 동안 각 포인트들 매치 (프레임당 변화됨)
        point_x[i]=x
        point_y[i]=y

        if prob > threshold : 
            cv2.circle(frameCopy, (int(x), int(y)), 8, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
            cv2.putText(frameCopy, "{}".format(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, lineType=cv2.LINE_AA)

            # Add the point to the list if the probability is greater than the threshold
            points.append((int(x), int(y)))
        else :
            points.append(None)
    
    print("frame's point value: ") # 프레임당 확인하기 위해 포인트 프린트
    print(point_x)
    print(point_y)
    point_x[15] = (point_x[8]+point_x[11])/2
    point_y[15] = (point_y[8]+point_y[11])/2

    # line 사이 각도 판별
    def __angle_between(p1,p2):
        ang1=np.arctan2(*p1[::-1])
        ang2=np.arctan2(*p2[::-1])
        res=np.rad2deg((ang1-ang2)%(2*np.pi))
        return res

    
    def getAngle3P(p1,p2,p3):
        pt1=(point_x[p1]-point_x[p2], point_y[p1]-point_y[p2])
        pt2=(point_x[p3]-point_x[p2], point_y[p3]-point_y[p2])
        res=__angle_between(pt1,pt2)
        res=(res+360)%360
        return res

    print("\nangle1:{}".format(getAngle3P(0,1,14)))
    print("angle1:{}\n".format(getAngle3P(1,14,8)))

    # ********************** 프레임에 skeleton 그림 ******************** #
    # Draw Skeleton
    for pair in POSE_PAIRS:
        partA = pair[0]
        partB = pair[1]

        if points[partA] and points[partB]:
            cv2.line(frame, points[partA], points[partB], (0, 255, 255), 3, lineType=cv2.LINE_AA)
            cv2.circle(frame, points[partA], 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)
            cv2.circle(frame, points[partB], 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)

    cv2.putText(frame, "time taken = {:.2f} sec".format(time.time() - t), (50, 50), cv2.FONT_HERSHEY_COMPLEX, .8, (255, 50, 0), 2, lineType=cv2.LINE_AA)
    # cv2.putText(frame, "OpenPose using OpenCV", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 50, 0), 2, lineType=cv2.LINE_AA)
    # cv2.imshow('Output-Keypoints', frameCopy)
    #cv2.imshow('Output-Skeleton', frame)

    vid_writer.write(frame)

vid_writer.release()
