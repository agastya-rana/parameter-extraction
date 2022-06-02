"""
Miscellaneous functions that depend on the machine or local
parameters or setups. 


Created by Nirag Kadakia at 21:30 08-14-2017
This work is licensed under the 
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 
International License. 
To view a copy of this license,
visit http://creativecommons.org/licenses/by-nc-sa/4.0/.
"""

import sys, os

scripts_path = [i for i in sys.path if 'scripts' in i][0]
## Append entire directory to system path here
#sys.path.append()

def def_data_dir():
	"""
	Define a data directory here for all 
	scripts in this project
	"""
	
	data_dir = os.path.join(os.path.dirname(scripts_path), 'swayamshree_data')
	
	return data_dir