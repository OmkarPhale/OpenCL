cls

del EnumOpenCLDevices.exe

cl.exe EnumOpenCLDevices.c /c /EHsc /Fo".\EnumOpenCLDevices.obj" /I "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.1\include" 
link.exe EnumOpenCLDevices.obj opencl.lib /LIBPATH:"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.1\lib\x64"

EnumOpenCLDevices.exe

del EnumOpenCLDevices.obj
