from basic.util import read_dict, AverageMeter, LogCollector, getVideoId
v2f_dict = read_dict('/root/VisualSearch/vatax/FeatureData/pyresnext-101_rbps13k,flatten0_output,os/video2frames.txt')
print(len(v2f_dict.keys()))
allall = 0
for key in v2f_dict.keys():
    allall += len(v2f_dict[key])
print(allall)

# with open('/root/VisualSearch/vatax/TextData/vataxsb1.caption.txt','w') as f: 

#     for line in open('/root/VisualSearch/vatax/TextData/vatax.caption.txt'):
#         cap_id, caption = line.strip().split(' ', 1)
#         vid = getVideoId(cap_id)
#         if vid in v2f_dict.keys():
#             f.write(line)




