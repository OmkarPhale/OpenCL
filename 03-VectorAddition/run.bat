cls

del VecAdd.exe

cl.exe VecAdd.cpp /c /EHsc /Fo".\VecAdd.obj" /I "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.1\include" 
link.exe VecAdd.obj opencl.lib /LIBPATH:"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.1\lib\x64"

VecAdd.exe

del VecAdd.obj
