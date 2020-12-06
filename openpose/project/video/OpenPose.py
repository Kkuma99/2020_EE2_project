import cv2             # for image processing
import time            # for FPS
import numpy as np     
import argparse        # for 인자

parser = argparse.ArgumentParser(description='Run keypoint detection')
parser.add_argument("--device", default="cpu", help="Device to inference on") # 거의 픽스해서 사용
parser.add_argument("--video_file", default="sitting_short_right.mp4", help="Input Video") # Input
side = 1 # left:1 / right:2
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
    POSE_PAIRS_left = [ [0,1],[1,14],[14,11],[11,12] ]
    POSE_PAIRS_right = [ [0,1],[1,14],[14,8],[8,9] ]
    # head = 0, neck = 1, waist = 14, hip = 8

# for faster time -> reduce size of input
inWidth = 368
inHeight = 368
threshold = 0.34 # 적절한 쓰레시 값

#********************************* video input 처리 ************************************#

input_source = args.video_file       #사용자에게 input 받은 비디오 파일 사용
cap = cv2.VideoCapture(input_source) 
hasFrame, frame = cap.read()

vid_writer = cv2.VideoWriter('output_tot.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame.shape[1],frame.shape[0])) # output 저장

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
        #cv2.waitKey()
        print("finished detection") # 탈출 확인
        cv2.destroyAllWindows()     # 다 close 하기
        break                       # 종료

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
            if side == 1:
                if i in [0,11,12,14]:
                    cv2.circle(frameCopy, (int(x), int(y)), 8, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
                    #cv2.putText(frameCopy, "{}".format(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, lineType=cv2.LINE_AA)
            elif side == 2:
                if i in [0,8,9,14]:
                    cv2.circle(frameCopy, (int(x), int(y)), 8, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
                    #cv2.putText(frameCopy, "{}".format(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, lineType=cv2.LINE_AA)

            # Add the point to the list if the probability is greater than the threshold
            points.append((int(x), int(y)))
        else :
            points.append(None)

    def __angle_between(p1,p2):
        ang1=np.arctan2(*p1[::-1])
        ang2=np.arctan2(*p2[::-1])
        res=np.rad2deg((ang1-ang2)%(2*np.pi))
        return res

    def getAngle3P(p1,p2,p3):
        pt1=(point_x[p1]-point_x[p2], point_y[p1]-point_y[p2])
        pt2=(point_x[p3]-point_x[p2], point_y[p3]-point_y[p2])
        product=np.dot(pt1,pt2)
        normu=np.linalg.norm(pt1)
        normv=np.linalg.norm(pt2)
        cost=product/(normu*normv)
        res=np.rad2deg(np.arccos(cost))
        return res

    if side == 1:
        print("\nangle1:{}".format(getAngle3P(0,1,14)))    # head-neck-waist
        print("angle2:{}".format(getAngle3P(1,14,11)))      # neck-waist-hip
        print("angle3:{}\n".format(getAngle3P(14,11,12)))    # waist-hip-knee

    elif side == 2:
        print("\nangle1:{}".format(getAngle3P(0,1,14)))    # head-neck-waist
        print("angle2:{}".format(getAngle3P(1,14,8)))      # neck-waist-hip
        print("angle3:{}\n".format(getAngle3P(14,8,9)))    # waist-hip-knee

    # ********************** 프레임에 skeleton 그림 ******************** #
    # Draw Skeleton
    if side == 1:
        for pair in POSE_PAIRS_left:
            partA = pair[0]
            partB = pair[1]

            if points[partA] and points[partB]:
                if (160 <= getAngle3P(0,1,14) <= 180) and (165 <= getAngle3P(1,14,11) <= 180) and (100 <= getAngle3P(14,11,12) <= 120) :
                    cv2.putText(frame, "proper pose", (30,30), cv2.FONT_HERSHEY_COMPLEX, .8, (255, 50, 0), 2, lineType=cv2.LINE_AA)
                else:
                    cv2.putText(frame, "wrong pose", (30,30), cv2.FONT_HERSHEY_COMPLEX, .8, (255, 50, 0), 2, lineType=cv2.LINE_AA)
                    
                cv2.line(frame, points[partA], points[partB], (0, 255, 255), 3, lineType=cv2.LINE_AA)
                cv2.circle(frame, points[partA], 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)
                cv2.circle(frame, points[partB], 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)

    elif side == 2:
        for pair in POSE_PAIRS_right:
            partA = pair[0]
            partB = pair[1]

            if points[partA] and points[partB]:
                if (160 <= getAngle3P(0,1,14) <= 180) and (165 <= getAngle3P(1,14,8) <= 180) and (100 <= getAngle3P(14,8,9) <= 120) :
                    cv2.putText(frame, "proper pose", (30,30), cv2.FONT_HERSHEY_COMPLEX, .8, (255, 50, 0), 2, lineType=cv2.LINE_AA)
                else:
                    cv2.putText(frame, "wrong pose", (30,30), cv2.FONT_HERSHEY_COMPLEX, .8, (255, 50, 0), 2, lineType=cv2.LINE_AA)

                cv2.line(frame, points[partA], points[partB], (0, 255, 255), 3, lineType=cv2.LINE_AA)
                cv2.circle(frame, points[partA], 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)
                cv2.circle(frame, points[partB], 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)

    cv2.putText(frame, "time taken = {:.2f} sec".format(time.time() - t), (50, 150), cv2.FONT_HERSHEY_COMPLEX, .8, (255, 50, 0), 2, lineType=cv2.LINE_AA)

    # angle print
    cv2.putText(frame, "angle:{:.0f}".format(getAngle3P(0,1,14)), (int(point_x[1]), int(point_y[1])), cv2.FONT_HERSHEY_PLAIN, 0.8, (0, 0, 0), 2, lineType=cv2.LINE_AA)
    cv2.putText(frame, "angle:{:.0f}".format(getAngle3P(1,14,8)), (int(point_x[14]), int(point_y[14])), cv2.FONT_HERSHEY_PLAIN, 0.8, (0, 0, 0), 2, lineType=cv2.LINE_AA)
    cv2.putText(frame, "angle:{:.0f}".format(getAngle3P(14,8,9)), (int(point_x[8]), int(point_y[8])), cv2.FONT_HERSHEY_PLAIN, 0.8, (0, 0, 0), 2, lineType=cv2.LINE_AA)
    # cv2.putText(frame, "OpenPose using OpenCV", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 50, 0), 2, lineType=cv2.LINE_AA)
    # cv2.imshow('Output-Keypoints', frameCopy)
    #cv2.imshow('Output-Skeleton', frame)

    vid_writer.write(frame)

vid_writer.release()
