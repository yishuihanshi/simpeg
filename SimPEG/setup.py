import os
print 'Compiling TriSolve.'
os.system('f2py -c Utils/TriSolve.f -m TriSolve')
print 'TriSolve Compiled! yay.'
print 'Moving TriSolve into Utils.'
os.system('mv TriSolve.so Utils/TriSolve.so')
print 'Thats it. Well Done Computer.'

