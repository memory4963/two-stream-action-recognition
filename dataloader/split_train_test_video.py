import os, pickle


class UCF101_splitter():
    def __init__(self, path):
        self.path = path

    def get_action_index(self):
        self.action_label = {}
        with open(self.path + 'classInd.txt') as f:
            content = f.readlines()
            content = [x.strip('\n') for x in content]
        f.close()
        for line in content:
            action, label = line.split(' ')
            # print label,action
            if action not in self.action_label.keys():
                self.action_label[action] = label

    def split_video(self):
        self.get_action_index()
        for path, subdir, files in os.walk(self.path):
            for filename in files:
                if filename.split('.')[0] == 'train_list':
                    train_video = self.file2_dic(self.path + filename)
                if filename.split('.')[0] == 'validate_list':
                    test_video = self.file2_dic(self.path + filename)
        print('==> (Training video, Validation video):(', len(train_video), len(test_video), ')')
        self.train_video = self.name_HandstandPushups(train_video)
        self.test_video = self.name_HandstandPushups(test_video)

        return self.train_video, self.test_video

    def file2_dic(self, fname):
        with open(fname) as f:
            content = f.readlines()
            content = [x.strip('\n') for x in content]
        f.close()
        dic = {}
        for line in content:
            # print line
            strs = line.split(' ')
            # video = line.split('/', 1)[1].split(' ', 1)[0]
            # key = video.split('_', 1)[1].split('.', 1)[0]
            # label = self.action_label[line.split('/')[0]]
            dic[strs[0].split('.')[0]] = int(strs[2])
            # print key,label
        return dic

    def name_HandstandPushups(self, dic):
        dic2 = {}
        for video in dic:
            n, g = video.split('_', 1)
            if n == 'HandStandPushups':
                videoname = 'HandstandPushups_' + g
            else:
                videoname = video
            dic2[videoname] = dic[video]
        return dic2


if __name__ == '__main__':
    path = '../UCF_list/'
    split = '01'
    splitter = UCF101_splitter(path=path, split=split)
    train_video, test_video = splitter.split_video()
    print(len(train_video), len(test_video))
