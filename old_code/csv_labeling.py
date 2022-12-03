import csv

# with open('test_hu_cap1.csv','r') as out:
#         csv_out=csv.writer(out)
#         csv_out.writerow(['hu0','hu1','hu2','hu3','hu4','hu5','hu6','label'])
#         for row in test:
#             csv_out.writerow(row)


with open('test_hu.csv',newline='') as f:
    r = csv.reader(f)
    data = [line for line in r]
with open('test_hu.csv','w',newline='') as f:
    w = csv.writer(f)
    w.writerow(['hu0','hu1','hu2','hu3','hu4','hu5','hu6','label'])
    w.writerows(data)