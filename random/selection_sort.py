def SelectionSort(num):
    for i in range (len(num)):
        lowest_index=i
        for j in range (i+1,len(num)):
            if num[lowest_index]>num[j]:
                lowest_index=j;
        num[i], num[lowest_index]=num[lowest_index], num[i]
