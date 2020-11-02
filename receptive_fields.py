import time
import numpy as np
from itertools import chain, product
import torch

#  Returns a dictionary containing:
#  - the layer wise mask M(xl0, yl0) 
#  - the shared weight matrix W(xl0*yl0, xl1*yl1) for each of full / off / on RFs

c,h,w = 1,32,32
image_dim = (c,h,w)
layers = 1

height = [image_dim[-1]] * (layers+1);
width  = [image_dim[-2]] * (layers+1);
chans  = [image_dim[-3]] * (layers+1);

filter_size = [9, 3] # please keep these odd
stride = [2, 2, 2]

layer_dim = [None] * (layers+1)

l2l_mask = {'on'	: [None] * (layers+1), 
			'off'  : [None] * (layers+1), 
			'full' : [None] * (layers+1)}

obs_mask = {'on'	: [None] * (layers+1), 
			'off'  : [None] * (layers+1), 
			'full' : [None] * (layers+1)}
			


#l2l_mask = {'full' : [None] * (layers+1)}

#obs_mask = {'full' : [None] * (layers+1)}



for l in range(layers):

	height[l+1] = ((height[l] - filter_size[l]) // stride[l] ) + 1
	width[l+1]	= ((width[l]  - filter_size[l]) // stride[l] ) + 1
	layer_dim[l] = [chans[l], height[l], width[l]]

	a = [i for i in range((filter_size[l]//2)+1)]
	cntr = a[len(a)//2]
	on = a[:cntr] ; off = a[cntr:] ; full = a;
	
	on_c_rf	  = set(product([*on, *[ -i for i in on]], repeat=2))
	full_c_rf = set(product([*a,*[-i for i in a]], repeat=2)) 
	off_c_rf  = full_c_rf.difference(on_c_rf)
	rfs = {'full' : full_c_rf, 'on' : on_c_rf, 'off' : off_c_rf}
	
	# indexable im space - disregarding channel dims
	coords = np.zeros((height[l], width[l]),dtype='i,i,i').tolist()
	
	for m in range(height[l]):
		for n in range(width[l]):
			coords[m][n] = (m,n)
	
	# chain image coordinates 
	coords = list(chain.from_iterable(zip(*coords)))
	
	for rf_type in l2l_mask.keys():
		print(rf_type)
		
		# apply coordinate transforms to rf	
		rf_coords	= [None] * len(coords)
	
		weight_matrix = np.zeros((chans[l]*height[l]*width[l], chans[l+1]*height[l+1]*width[l+1]))
		mask_matrix	  = np.zeros((chans[l]*height[l]*width[l], chans[l]*height[l]*width[l]))
		# for every image coordinate
		for ind, imxy in enumerate(coords): # better plz 
	
			# calculate the receptive field in terms of real indices
			imxyrf = np.array([(imxy[0] + rf_[0], imxy[1] + rf_[1]) for rf_ in rfs[rf_type]])
	
			# filter invalid coordinates
			validation = lambda y, w, h : y[0]>=0 and y[0]<w and y[1]>=0 and y[1]<h 
			imxyrf	 = imxyrf[[validation(rfind, width[l],height[l]) for rfind in imxyrf]]
			
			# filter unique coordinates
			imxyrf	 = list(set(map(tuple, imxyrf)))
			rf_coords[ind] = imxyrf
			
			# mask of ones 
			rf_map = np.zeros((chans[l], height[l], width[l])).astype('int')
			for rcrd in rf_coords[ind]:
				rf_map[:, rcrd[0], rcrd[1]] = 1
	
			# connection map
			connect_to = np.zeros((chans[l+1],height[l+1], width[l+1]))
			for x in range(height[l+1]):
				for y in range(width[l+1]):
					connect_to[:,x,y] = rf_map[:, stride[l]*x+filter_size[l+1], stride[l]*y+filter_size[l+1]]
	
			weight_matrix[ind,:] = connect_to.flatten()
			mask_matrix[ind,:]	 = rf_map.flatten()
	
		l2l_mask[rf_type][l] = torch.from_numpy(weight_matrix).float()
		obs_mask[rf_type][l] = torch.from_numpy(mask_matrix).float()
		

layer_dim  = list(filter(None.__ne__, layer_dim))

np.set_printoptions(threshold=np.inf)

for i in obs_mask['off'][0]:
	print(i.numpy().astype(int).reshape((w, h)))
	time.sleep(0.5)
