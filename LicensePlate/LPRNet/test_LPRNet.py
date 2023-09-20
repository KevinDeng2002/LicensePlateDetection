'''
test pretrained model.
Author: aiboy.wei@outlook.com .
'''

from dataset.load_data import CHARS, LPRDataLoader
from PIL import Image, ImageDraw, ImageFont
from model.LPRNet import build_lprnet
# import torch.backends.cudnn as cudnn
from model.STN import STNet
from torch.autograd import Variable
from torch.utils.data import *
import numpy as np
import argparse
import torch
import time
import cv2
import os


def get_parser():
    parser = argparse.ArgumentParser(description='parameters to train net')
    parser.add_argument('--img_size', default=[94, 24], help='the image size')
    # parser.add_argument('--test_img_dirs', default="G:\plate\clipped", help='the test images path')
    # parser.add_argument('--test_img_dirs', default=r"G:\plate\LPRNet_Pytorch-master\data\teachertest", help='the test images path')
    parser.add_argument('--test_img_dirs', default="G:\plate\LicensePlate-master\data\preprocessed\lpr_test2",
                        help='the test images path')
    parser.add_argument('--dropout_rate', default=0, help='dropout rate.')
    parser.add_argument('--lpr_max_len', default=8, help='license plate number max length.')
    parser.add_argument('--test_batch_size', default=1, help='testing batch size.')
    parser.add_argument('--phase_train', default=False, type=bool, help='train or test phase flag.')
    parser.add_argument('--num_workers', default=8, type=int, help='Number of workers used in dataloading')
    parser.add_argument('--cuda', default=True, type=bool, help='Use cuda to train model')
    parser.add_argument('--show', default=False, type=bool, help='show test image and its predict result or not.')
    parser.add_argument('--pretrained_model', default='./weights/Final_LPRNet_model.pth', help='pretrained base model')

    args = parser.parse_args()

    return args


def collate_fn(batch):
    imgs = []
    labels = []
    lengths = []
    for _, sample in enumerate(batch):
        img, label, length = sample
        imgs.append(torch.from_numpy(img))
        labels.extend(label)
        lengths.append(length)
    labels = np.asarray(labels).flatten().astype(np.float32)

    return (torch.stack(imgs, 0), torch.from_numpy(labels), lengths)


def test():
    args = get_parser()

    lprnet = build_lprnet(lpr_max_len=args.lpr_max_len, phase=args.phase_train, class_num=len(CHARS),
                          dropout_rate=args.dropout_rate)
    device = torch.device("cuda:0" if args.cuda else "cpu")
    lprnet.to(device)
    print("Successful to build network!")
    Stn = STNet()
    Stn.to(device)
    print("STnloaded")

    # load pretrained model
    if args.pretrained_model:
        lprnet.load_state_dict(torch.load(args.pretrained_model))
        Stn.load_state_dict(torch.load('weights/Final_STN_model.pth', map_location=lambda storage, loc: storage))
        print("load pretrained model successful!")
    else:
        print("[Error] Can't found pretrained mode, please check!")
        return False

    test_img_dirs = os.path.expanduser(args.test_img_dirs)
    test_dataset = LPRDataLoader(test_img_dirs.split(','), args.img_size, args.lpr_max_len)
    try:
        Greedy_Decode_Eval(Stn, lprnet, test_dataset, args)
    finally:
        cv2.destroyAllWindows()


def Greedy_Decode_Eval(Net1, Net, datasets, args):
    # TestNet = Net.eval()
    epoch_size = len(datasets) // args.test_batch_size
    batch_iterator = iter(
        DataLoader(datasets, args.test_batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_fn))

    Tp = 0
    Tn_1 = 0
    Tn_2 = 0
    t1 = time.time()
    for i in range(epoch_size):
        # load train data
        images, labels, lengths = next(batch_iterator)
        start = 0
        targets = []
        for length in lengths:
            label = labels[start:start + length]
            targets.append(label)
            start += length
        targets = np.array([el.numpy() for el in targets])
        imgs = images.numpy().copy()

        if args.cuda:
            images = Variable(images.cuda())
        else:
            images = Variable(images)

        # forward
        transfer = Net1(images)
        prebs = Net(transfer)
        # greedy decode
        prebs = prebs.cpu().detach().numpy()
        preb_labels = list()
        for i in range(prebs.shape[0]):
            preb = prebs[i, :, :]
            preb_label = list()
            for j in range(preb.shape[1]):
                preb_label.append(np.argmax(preb[:, j], axis=0))
            no_repeat_blank_label = list()
            pre_c = preb_label[0]
            if pre_c != len(CHARS) - 1:
                no_repeat_blank_label.append(pre_c)
            for c in preb_label:  # dropout repeate label and blank label
                if (pre_c == c) or (c == len(CHARS) - 1):
                    if c == len(CHARS) - 1:
                        pre_c = c
                    continue
                no_repeat_blank_label.append(c)
                pre_c = c
            preb_labels.append(no_repeat_blank_label)
        for i, label in enumerate(preb_labels):
            # show image and its predict label
            if args.show:
                show(imgs[i], label, targets[i])
            # with open(r"结果.txt", "a+", encoding='utf-8') as f:
            #     with open(r'C:\Users\rekyse\Desktop\License_Plate_Detection\plate_Rec/label.json','r')as fp:
            #         testlabel = json.load(fp)
            #         # if len(label) != len(testlabel[targets[i]+'.jpg']):
            #         #     f.write('   长度错误')
            #         #     Tn_1 += 1
            #         #     f.write('\n')
            #         #     continue
            #         if (np.asarray(testlabel[targets[i]+'.jpg']) == np.asarray(label)).all():
            #             f.write('   正确')
            #             Tp += 1
            #         else:
            #             f.write('   错误！！！！')
            #             Tn_2 += 1
            with open(r"结果.txt", "a+", encoding='utf-8') as f:
                # f.write('文件名：')
                # for tww in range(0, len(np.asarray(targets[i]))):
                #     wyw = round(np.asarray(targets[i])[tww])
                #     print(CHARS[wyw], end='')
                #     f.write(CHARS[wyw])
                # print(':', end='')
                # f.write('      预测结果：')
                for tww in range(0, len(np.asarray(label))):
                    print(CHARS[np.asarray(label)[tww]], end='')
                    f.write(CHARS[np.asarray(label)[tww]])
                if len(label) != len(targets[i]):
                    f.write('   长度错误')
                    Tn_1 += 1
                    f.write('\n')
                    continue
                if (np.asarray(targets[i]) == np.asarray(label)).all():
                    # f.write('   正确')
                    Tp += 1
                else:
                    f.write('   错误！！！！')
                    Tn_2 += 1

                f.write('\n')
                print('')
                f.close()

    Acc = Tp * 1.0 / (Tp + Tn_1 + Tn_2)
    print("[Info] Test Accuracy: {} [{}:{}:{}:{}]".format(Acc, Tp, Tn_1, Tn_2, (Tp + Tn_1 + Tn_2)))
    t2 = time.time()
    print("[Info] Test Speed: {}s 1/{}]".format((t2 - t1) / len(datasets), len(datasets)))


def show(img, label, target):
    img = np.transpose(img, (1, 2, 0))
    img *= 128.
    img += 127.5
    img = img.astype(np.uint8)

    lb = ""
    for i in label:
        lb += CHARS[i]
    tg = ""
    for j in target.tolist():
        tg += CHARS[int(j)]

    flag = "F"
    if lb == tg:
        flag = "T"
    # img = cv2.putText(img, lb, (0,16), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.6, (0, 0, 255), 1)
    img = cv2ImgAddText(img, lb, (0, 0))
    cv2.imshow("test", img)
    print("target: ", tg, " ### {} ### ".format(flag), "predict: ", lb)
    cv2.waitKey()
    cv2.destroyAllWindows()


def cv2ImgAddText(img, text, pos, textColor=(255, 0, 0), textSize=12):
    if (isinstance(img, np.ndarray)):  # detect opencv format or not
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    fontText = ImageFont.truetype("data/NotoSansCJK-Regular.ttc", textSize, encoding="utf-8")
    draw.text(pos, text, textColor, font=fontText)

    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


if __name__ == "__main__":
    test()
