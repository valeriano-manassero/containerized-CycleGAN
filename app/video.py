import os
import cv2


SIZE = 256


def generate_frames(prefixes, augmentation):
    if os.path.exists('/dataset/frames'):
        for root, dirs, files in os.walk('/dataset/frames', topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))
    else:
        os.makedirs('/dataset/frames')
    for prefix in prefixes:
        os.makedirs('/dataset/frames/' + prefix)
        capture = cv2.VideoCapture('/dataset/' + prefix + '.mp4')
        count = 0
        while True:
            success, image = capture.read()
            if success:
                resize = cv2.resize(image, (SIZE, SIZE))
                cv2.imwrite('/dataset/frames/' + prefix + '/%05d.jpg' % count, resize)
                count += 1
                if augmentation:
                    flip = cv2.flip(resize, 1)
                    cv2.imwrite('/dataset/frames/' + prefix + '/%05d_f.jpg' % count, flip)
            else:
                break


def generate_video():
    if os.path.exists('/output/output.mp4'):
        os.remove('/output/output.mp4')
    video = cv2.VideoWriter('/output/output.mp4', cv2.VideoWriter_fourcc(*"mp4v"), 20.0, (SIZE, SIZE))

    for frame in sorted(os.listdir('/output/result/')):
        img = cv2.imread('/output/result/' + frame)
        video.write(img)

    video.release()
    cv2.destroyAllWindows()
