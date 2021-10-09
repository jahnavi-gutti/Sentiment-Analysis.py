file = open('imdb.txt', 'r')
data = file.readlines()
print(data)
file.close()
k=0
dic={"good":0.5,"great":0.8,"terrible":-0.8,"alright":0.1}
for s in data:
   s=s.split()
   c=0
   for i in s:
      if i in dic:
         c+=dic[i]
   k+=c
   print(*s, ":",c)
for i in range(int((k*10)/2)):
   print("*",end=" ")

   

