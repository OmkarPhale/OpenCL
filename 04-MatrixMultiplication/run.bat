cls

del MatMul.exe

cl.exe MatMul.cpp /c /EHsc /Fo".\MatMul.obj" /I "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.1\include" 
link.exe MatMul.obj opencl.lib /LIBPATH:"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.1\lib\x64"

MatMul.exe

del MatMul.obj
