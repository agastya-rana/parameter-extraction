"""
General, miscellaneous functions for CS decoding scripts.

Created by Nirag Kadakia at 23:30 07-31-2017
This work is licensed under the 
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 
International License. 
To view a copy of this license,
visit http://creativecommons.org/licenses/by-nc-sa/4.0/.
"""

import scipy as sp
import sys

def get_flags():
	"""
	Args:
		arg_num: the command line argument number
	
	Returns:
		data_flag: string
	"""
	
	data_flags = []
	if len(sys.argv) < 2:
		raise Exception("Need at least 1 tag for the data in command line")
		quit()
	else:
		for flag in sys.argv[1:]:
			data_flags.append(str(flag))
	return data_flags
	
def get_flag(arg_num=1):
	"""
	Args:
		arg_num: the command line argument number
	
	Returns:
		data_flag: string
	"""
	try:
		data_flag = str(sys.argv[arg_num])
	except:
		raise Exception("Need to specify a tag for the data in command line")
		
	return data_flag

def merge_two_dicts(x, y):
   	"""
	Given two dicts, merge them into a 	new dict as a shallow copy.
	"""

	z = x.copy()
	z.update(y)
	
	return z
		
def noisify(Ss, params=[0, 1]):
	"""
	Adds noise to any vector.
	"""
	
	mu, sigma = params
	size = Ss.shape
	Ss += sp.random.normal(mu, sigma, size)
	
	return Ss

def project_tensor(tensor, axes, projection_components, projected_axes):
	"""
	Project a tensor array of rank > 2 to lower dimensions
	along given axes.
	
	Args:
		tensor: numpy array whose shape has length > 2
		axes: dictionary whose keys are the names of the variables
						to be projected to and whose values are their 
						respective ranges as rank-1 numpy arrays.
		projection_components: dictionary of axes to project along, whose 
					keys are the names of the projected axis variablse and 
					whose values indicate the component along which to 
					take the projection. Index must be less than or equal 
					to the length of this axis.
		projected_axes: 2-element list indicated which indices of axes 
						are to be projected to. 
		
	Returns:
		tensor: the projected tensor of shape 1 less than the input tensor.
	"""

	assert len(tensor.shape) > 2, \
		'Cannot project a rank 1 or 2 tensor to two dimensions'
	assert len(projected_axes) == 2, 'Can only project to two dimensions'
	
	for idx, name in enumerate(axes.keys()):
		if idx == projected_axes[0]:
			pass
		elif idx == projected_axes[1]:
			pass
		else:
			proj_axis = axes.keys().index(name)
			
			try:
				print ('Setting %s fixed..' % name)
				proj_element = projection_components[name]
			except:
				print ('Need to specify iterated variable values that ' \
						'are not being plotted in projection_components ' \
						'dictionary')
				quit()
			
			assert (proj_element < len(axes[name])), \
					'Fixed index out of range, %s >= %s'\
					% (proj_element, len(axes[name]))
			proj_vec = sp.zeros(len(axes[name]))
			proj_vec[proj_element] = 1.0
			
			tensor = sp.tensordot(tensor, proj_vec, [proj_axis, 0])
	
	return tensor
	
def clip_array(array_dict, min=1e-10, max=1e10):
	"""
	Clip an array to a particular interval.
	"""
	
	for name, array in array_dict.items():
		
		lower_bound_elements = sp.sum(array < min)
		if lower_bound_elements > 0:
			print 'Clipping %s; %s lower bound elements' \
					% (name, lower_bound_elements)
			array_dict[name] = array.clip(min=min)
	
		upper_bound_elements = sp.sum(array > max)
		if upper_bound_elements > 0:
			print 'Clipping %s; %s upper bound elements' \
					% (name, upper_bound_elements)
			array_dict[name] = array.clip(max=max)
	
	return array_dict