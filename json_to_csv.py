# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 13:11:01 2019

@author: anant.jain08
"""

import json
import csv 
#f=open(r'D:\aziz_accounting\01_02_2019\Types.json','r')
#data=[line for line in open(r'D:\aziz_accounting\01_02_2019\Types.json','r')]
def get_leaves(item,key=None):
    if isinstance(item, dict):
        leaves = []
        for i in item.keys():
            leaves.extend(get_leaves(item[i], i))
        return leaves
    elif isinstance(item, list):
        leaves = []
        for i in item:
            leaves.extend(get_leaves(i, key))
        return leaves
    else:
        return [(key, item)]


#with open(r'D:\aziz_accounting\01_02_2019\Companies.json') as f_input,
    
 
with open(r'D:\aziz_accounting\01_02_2019\Comapnies.csv', 'w+', newline='') as f_output:
    data=[line for line in open(r'D:\aziz_accounting\01_02_2019\Companies.json','r')]
    write_header = True
    for rec in data:
        entry_dict=json.loads(rec)
        print(entry_dict)
        leaf_entries = get_leaves(entry_dict)
        print(leaf_entries)
        #break
        csv_output = csv.writer(f_output)
        
        if write_header:
            csv_output.writerow([k for (k, v) in leaf_entries])
            write_header = False
        csv_output.writerow([v for k, v in leaf_entries])    
#print(data)
#s=f.readlines()
#print(s[3])
##        
#print(type(f))

#with open(r'D:\aziz_accounting\01_02_2019\Comapnies.csv','w+') as file:
#    for rec in data:
#        row=[]
#        d=json.loads(rec)
#        #print(type(d))
#        for key in d.keys():
#            if isinstance(d[key],dic
#            row.append(d[key])
#        row_final=[row]
#        writer=csv.writer(file)
#        writer.writerows(row_final)
#file.close()    