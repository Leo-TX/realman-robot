'''
Author: TX-Leo
Mail: tx.leo.wz@gmail.com
Date: 2024-06-20 14:43:39
Version: v1
File: 
Brief: 
'''
def primitive_unlock(T, _, __, threshold):
    ret = do_rotation(T, threshold)
    if ret == "success":
    	return 0
    elif ret == "threshold":
    	return 1
    else:
    	return -1

def do_primitive(primitive_id, primitive_params):
	ret = primitive_dict[primitive_id](*primitive_params)
	return ret

def step(obs, last_result):
	next_primitive = high_level_model(obs, last_result)
	next_primitive_params = low_level_model(obs, last_result, next_primitive)
	return next_primitive, next_primitive_params

while task_is_not_done:
	obs = get_obs()
	next_primitive, next_primitive_params = step(obs, last_result)
	this_result = do_primitive(next_primitive, next_primitive_params)
	last_result = this_result