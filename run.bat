@REM @echo off
start cmd /k "cd /d C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\ && .\vcvars64.bat"

cd /d C:\Users\Wu\source\buffer\Newbranch

cmake -S . -B build -D CMAKE_BUILD_TYPE=Release -D LUISA_COMPUTE_ENABLE_CPU=OFF -G Ninja.
cmake --build build

exit



.\\build\bin\luisa-render-cli -b cuda ./data/scenes/cbox-diff/cbox-diff-2.luisa