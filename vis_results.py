import pickle

# fname = 'somethingv2_1.pkl'
# with open(fname, 'rb') as f:
#     res = pickle.load(f)
# for i in range(len(res)):
#     if type(res[i]) is int:
#         continue
#     else:
#         for prob, cap in res[i]:
#             #print(cap)
#             if 'Spreading' in cap and prob > 0.04:
#                 print(i, prob, cap)
#

add = [0, 1, 5, 6, 7, 8, 9, 10, 15, 18, 19, 20, 21, 22, 23, 24, 29, 30, 31, 32, 38]

fname = 'moments_1.pkl'
with open(fname, 'rb') as f:
    moments_res = pickle.load(f)
add_found = []
keywords = ['Putting', 'Stuffing', 'Scooping', 'Poking', 'Piling', 'out', 'into']
fname = 'somethingv2_1.pkl'
with open(fname, 'rb') as f:
    res = pickle.load(f)
for i in range(len(res)):
    if type(res[i]) is int:
        continue
    else:
        found_mix = False
        for prob, cap in res[i]:
            print(i, cap, prob)
            # if 'Spreading' in cap and prob > 0.01:
            #     found_mix = True
        if not found_mix:
            for prob, cap in res[i]:
                for keyword in keywords:
                    if keyword in cap and prob > 0.1:
                        print('-'*10, i, prob, cap)
                        add_found.append(i)
                        break

final_add_found = []
for idx in add_found:
    found_mix = False
    for prob, cap in moments_res[idx]:
        if 'frying' in cap and prob > 0.15:
            found_mix = True
    if not found_mix:
        final_add_found.append(idx)
print('recall')
for idx in add:
    if idx not in final_add_found:
        print(idx)
print('precision')
for idx in final_add_found:
    if idx not in add:
        print(idx)
        #print(moments_res[idx])
        



# for i in range(len(res)):
#     if type(res[i]) is int:
#         continue
#     else:
#         for prob, cap in res[i]:
#             #print(cap)
#             if 'frying' in cap and prob > 0.15:
#                 print(i, prob, cap)