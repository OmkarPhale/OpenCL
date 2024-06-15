cls

del HelloOpenCL.exe

cl.exe HelloOpenCL.c /c /EHsc /Fo".\HelloOpenCL.obj" /I "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.1\include" 
link.exe HelloOpenCL.obj opencl.lib /LIBPATH:"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.1\lib\x64"

HelloOpenCL.exe

del HelloOpenCL.obj
