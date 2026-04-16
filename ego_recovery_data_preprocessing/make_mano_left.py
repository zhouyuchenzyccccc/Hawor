import pickle, copy
src = '/home/ubuntu/WorkSpace/ZYC/hamer/_DATA/data/mano/MANO_RIGHT.pkl'
dst = '/home/ubuntu/WorkSpace/ZYC/hawor/_DATA/data_left/mano_left/MANO_LEFT.pkl'
with open(src, 'rb') as f:
    right = pickle.load(f, encoding='latin1')
left = copy.deepcopy(right)
# HaWoR run_mano_left applies fix_shapedirs=True (shapedirs[:,0,:] *= -1) automatically
# so we just copy the right model as a placeholder
with open(dst, 'wb') as f:
    pickle.dump(left, f)
print('MANO_LEFT.pkl created at', dst)
