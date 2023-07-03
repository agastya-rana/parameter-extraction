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

## Append entire directory to system path here
#sys.path.append()

def def_data_dir():
	"""
	Define a data directory here for all 
	scripts in this project
	"""
	
	data_dir = "/home/asr93/UpdatedFRETassimilation/FRET-data-assimilation/swayamshree_data"
	
	return data_dir