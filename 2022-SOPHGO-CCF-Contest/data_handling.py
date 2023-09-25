import os
import cv2
import argparse




def reshape_imgs():
    init_maxSize = [2048, 2048]
    minSize = [576, 768]

    path = args.img_path
    output_path = args.out_path

    if not os.path.exists(output_path):
        os.mkdir(output_path)

    img_list = list(filter(lambda x: x.endswith('jpg'), os.listdir(path)))

    for file_name in img_list:

        im = cv2.imread(os.path.join(path, file_name))
        
        h, w, c = im.shape

        # determine rate, scaled h, w should not be lager than max h, w
        rate = init_maxSize[0] / h
        rate_w = w * rate
        if rate_w > init_maxSize[1]:
            rate = init_maxSize[1] / w
        tmp_h = int(h * rate / 16) * 16

        # determine rate, scaled h, w should not be smaller than min h, w
        if tmp_h < minSize[0]:
            rate = minSize[0] / h
        tmp_w = int(w * rate / 16) * 16

        if tmp_w < minSize[1]:
            rate = minSize[1] / w
        tmp_h = int(h * rate / 16) * 16
        tmp_w = int(w * rate / 16) * 16

        # real rate
        rate_h = tmp_h / h
        rate_w = tmp_w / w
        # im = imresize(im,[tmp_h,tmp_w]);
        tmp = im.shape
        im = cv2.resize(im, (tmp_w, tmp_h))

        cv2.imwrite(os.path.join(output_path, file_name), im)


if __name__ == '__main__':
        parser = argparse.ArgumentParser(description='Trace Pytorch Model.')
        parser.add_argument('--img-path', type=str, default='./',
                    help='image path')
        parser.add_argument('--out-path', type=str,
                    help='output image path')

        args = parser.parse_args()
        reshape_imgs()