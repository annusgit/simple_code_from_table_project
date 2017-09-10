import os
import cv2
import moviepy
from moviepy.editor import VideoFileClip 
from transforms import table_transform as tt 
print('All imports successful!')

video_folder = 'video_folder'
clip = VideoFileClip("table.MP4")
index = 0
for frame in clip.iter_frames():
    index += 1
    string = 'on frame = {}'.format(index)
    print(string, end='')
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    this = tt(gray_frame)
    new = os.path.join(video_folder, str(index) + '.jpg')
    cv2.imwrite(new, this)
    print('\b'*len(string), end='', flush=True)

def save():
    os.system("ffmpeg -f image2 -r 30 -i video_folder/%01d.jpg -vcodec mpeg4 -y movie.mp4")

save()














