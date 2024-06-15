cls

del DevProp.exe

cl.exe DevProp.c /c /EHsc /Fo".\DevProp.obj" /I "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.1\include" 
link.exe DevProp.obj opencl.lib /LIBPATH:"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.1\lib\x64"

DevProp.exe

del DevProp.obj
