import os

# 현재 위치의 파일 목록
mask_files = os.walk('./mask').__next__()[2]
original_files = os.walk('./original').__next__()[2]
# 파일명에 번호 추가하기
count = 763
for name in range(1,451):
    os.rename("./mask/complex_right_frame_"+str(name)+"_seg.jpg", "./mask/{0}.jpg".format(count))
    os.rename("./original/complex_right_frame_"+str(name)+".jpg", "./original/{0}.jpg".format(count))
    count += 1
