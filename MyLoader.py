import os, io_utils
import torch.utils.data as data
import os.path as osp
import numpy as np

from scipy.misc import imread

class MyLoader(data.Dataset):

    def __init__(self, path_to_assos):
        """
        :param the directory of color images
        :param the directory of depth images
        """
        with open( str(path) +'/2all_gt.txt' , 'r') as fileassos:
            data = fileassos.readlines()

            for i, line in enumerate((data[:-1])):
                self.color_pairs = []
                self.depth_pairs = []
                
                new_data = data[i].split()
                second_data = data[i+1].split()  
                rgb_first = new_data[1]
                rgb_second = second_data[1]  
                depth_first = new_data[1]
                depth_second = second_data[1]  
                self.color_pairs.append([
                    osp.join(str(path) + '/' +str(rgb_first)), 
                    osp.join(str(path) + '/' +str(rgb_second))
                    ] )
                self.depth_pairs.append([
                    osp.join(str(path) + '/' +str(depth_first)), 
                    osp.join(str(path) + '/' +str(depth_second))
                    ] )

    def __getitem__(self, index):

        image0_path, image1_path = self.color_pairs[index]
        depth0_path, depth1_path = self.depth_pairs[index]

        image0 = self.__load_rgb_tensor(image0_path)
        image1 = self.__load_rgb_tensor(image1_path)

        depth0 = self.__load_depth_tensor(depth0_path)
        depth1 = self.__load_depth_tensor(depth1_path)

        return image0, image1, depth0, depth1

    def __load_rgb_tensor(self, path):
        image = imread(path)
        image = image.astype(np.float32) / 255.0
        image = np.transpose(image, (2,0,1))
        return image

    def __load_depth_tensor(self, path):
        if path.endswith('.dpt'):
            depth = io_utils.depth_read(path)
        elif path.endswith('.png'):
            depth = imread(path) / 1000.0
        else: 
            raise NotImplementedError
        return depth[np.newaxis, :]
